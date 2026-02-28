from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json
import threading
from typing import Optional, Any, Tuple, Dict

# ==============================
# FLASK SETUP
# ==============================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Prefer common filenames present in the repository. Try multiple fallbacks so
# deployment naming differences don't prevent model loading.
_candidate_models = ["Kidney.h5", "kidney.h5", "kidneymodels.keras", "Kidney.keras"]
MODEL_PATH: Optional[str] = None
for _n in _candidate_models:
    _p = os.path.join(BASE_DIR, _n)
    if os.path.exists(_p):
        MODEL_PATH = _p
        break
if MODEL_PATH is None:
    # default to the known model filename used in this project
    MODEL_PATH = os.path.join(BASE_DIR, "kidneymodels.keras")

# label files: try common names
_candidate_labels = ["labels.json", "label.json", "labels.txt"]
LABELS_PATH: Optional[str] = None
for _n in _candidate_labels:
    _p = os.path.join(BASE_DIR, _n)
    if os.path.exists(_p):
        LABELS_PATH = _p
        break
if LABELS_PATH is None:
    LABELS_PATH = os.path.join(BASE_DIR, "labels.json")

# ==============================
# LOAD MODEL
# ==============================
model: Optional[Any] = None
# lock to prevent concurrent model loads in multi-thread/process servers
_model_lock = threading.Lock()

def ensure_model_loaded() -> Tuple[bool, Optional[str]]:
    """Ensure the global `model` is loaded. Returns (True, None) on success,
    or (False, error_message) on failure."""
    global model
    if model is not None:
        return True, None
    with _model_lock:
        if model is not None:
            return True, None
        try:
            print(f"Attempting to load model with tf.keras from: {MODEL_PATH}")
            # Some saved models reference internal paths like `keras.src.models.functional`.
            # Create lightweight sys.modules aliases so deserialization can import them.
            try:
                import sys as _sys
                import importlib as _importlib
                # map keras.src -> keras, keras.src.models -> keras.models, etc.
                if 'keras' in _sys.modules:
                    _keras_mod = _sys.modules['keras']
                else:
                    _keras_mod = _importlib.import_module('keras')
                # ensure submodule aliases
                _sys.modules.setdefault('keras.src', _keras_mod)
                if hasattr(_keras_mod, 'models'):
                    _sys.modules.setdefault('keras.src.models', _keras_mod.models)
                    try:
                        _sys.modules.setdefault('keras.src.models.functional', _importlib.import_module('keras.models.functional'))
                    except Exception:
                        pass
            except Exception:
                pass
            model_obj = tf.keras.models.load_model(MODEL_PATH, compile=False)
            model = model_obj
            print("[OK] Model loaded successfully with tf.keras")
            return True, None
        except Exception as e_tf:
            import traceback as _tb
            tb_tf = _tb.format_exc()
            print("tf.keras failed to load model, will try standalone keras. Error:\n", tb_tf)
            try:
                import keras as _keras
                print(f"Attempting to load model with standalone keras from: {MODEL_PATH}")
                model_obj = _keras.models.load_model(MODEL_PATH, compile=False)
                model = model_obj
                print("[OK] Model loaded successfully with standalone keras")
                return True, None
            except Exception as e_ks:
                tb_ks = _tb.format_exc()
                print("standalone keras failed to load model. Error:\n", tb_ks)
                return False, f"tf_error: {e_tf}; keras_error: {e_ks}"

# try to load at import time too (best-effort)
_ok, _err = ensure_model_loaded()
if not _ok:
    print("Model not loaded at import time; will attempt on first request.")

# ==============================
# LOAD LABELS
# ==============================
labels: Optional[Dict[int, str]] = None
if os.path.exists(LABELS_PATH):
    try:
        with open(LABELS_PATH, "r") as f:
            labels = json.load(f)
            labels = {int(k): v for k, v in labels.items()}
        print("[OK] Labels loaded:", labels)
    except Exception as e:
        print("[ERROR] Failed to load labels:", e)

# ==============================
# PREPROCESS FUNCTION
# ==============================
def preprocess_image(image: Image.Image) -> Any:
    image = image.convert("RGB")
    image = image.resize((224, 224), resample=Image.BILINEAR)  # Must match training
    image = np.array(image, dtype=np.float32)
    image = image / 255.0  # Must match training normalization
    image = np.expand_dims(image, axis=0)
    return image

# ==============================
# ROUTES
# ==============================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Ensure model is loaded (attempt on-demand). If loading fails, return error.
    ok, err = ensure_model_loaded()
    if not ok:
        print("Model load error on request:", err)
        return jsonify({"error": "Model not loaded", "details": str(err)}), 500

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        try:
            # force RGB on upload to avoid RGBA/L mismatches
            image = Image.open(file).convert("RGB")
        except Exception:
            return jsonify({"error": "Invalid image file"}), 400

        processed = preprocess_image(image)

        prediction = model.predict(processed)
        print("RAW PREDICTION:", prediction)

        predicted_class = int(np.argmax(prediction, axis=1)[0])
        confidence = float(np.max(prediction))

        response = {
            "class_index": predicted_class,
            "confidence": round(confidence, 4),
            "raw": prediction.tolist()
        }

        if labels and predicted_class in labels:
            response["label"] = labels[predicted_class]

        return jsonify(response)
    except Exception as e:
        import traceback as _tb
        tb = _tb.format_exc()
        print("Exception in /predict handler:\n", tb)
        return jsonify({"error": "Internal server error", "details": str(e), "trace": tb}), 500


@app.route('/_model_debug', methods=['GET'])
def model_debug() -> Any:
    info: Dict[str, Any] = {"model_loaded": model is not None}
    try:
        info['model_path'] = MODEL_PATH
        info['model_exists'] = os.path.exists(MODEL_PATH)
        if info['model_exists']:
            info['model_size_bytes'] = os.path.getsize(MODEL_PATH)
    except Exception as e:
        info['path_error'] = str(e)

    if model is not None:
        try:
            import io as _io
            buf = _io.StringIO()
            model.summary(print_fn=lambda s: buf.write(s + "\n"))
            info['summary'] = buf.getvalue()
        except Exception as e:
            info['summary_error'] = str(e)
        try:
            info['input_shape'] = getattr(model, 'input_shape', None)
        except Exception:
            info['input_shape'] = None
    return jsonify(info)

# ==============================
# MAIN (Render Compatible)
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)