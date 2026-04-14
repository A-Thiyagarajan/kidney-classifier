from flask import Flask, request, jsonify, render_template
import os
import json
import traceback
import threading
from typing import Optional, Any, Tuple, Dict
import io

print("[INFO] App loaded - ALL model/imports lazy-loaded on first request")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Kidney.h5")
LABELS_PATH = os.path.join(BASE_DIR, "label.json")

model: Optional[Any] = None
_model_lock = threading.Lock()

def lazy_import_tf() -> Any:
    """Lazy import tensorflow"""
    import tensorflow as tf
    return tf

def lazy_import_keras():
    """Lazy import keras"""
    import keras
    return keras

def lazy_import_numpy():
    """Lazy import numpy"""
    import numpy as np
    return np

def lazy_import_pil():
    """Lazy import PIL"""
    from PIL import Image
    return Image

def ensure_model_loaded() -> Tuple[bool, Optional[str]]:
    global model
    if model is not None:
        return True, None
    with _model_lock:
        if model is not None:
            return True, None
        print(f"[DEBUG] Attempting model load - path: {MODEL_PATH}")
        print(f"[DEBUG] File exists: {os.path.exists(MODEL_PATH)}, size: {os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0}")
        try:
            tf = lazy_import_tf()
            print(f"[DEBUG] TF version: {tf.__version__}, Keras location: {tf.keras.__file__}")
            print(f"[DEBUG] Loading model with tf.keras: {MODEL_PATH}")
            import sys, importlib
            sys.modules.setdefault('keras.src', tf.keras)
            sys.modules.setdefault('keras.src.models', tf.keras.models)
            custom_objects = {'InputLayer': tf.keras.layers.InputLayer}
            model_obj = tf.keras.models.load_model(MODEL_PATH, compile=False, custom_objects=custom_objects)
            model = model_obj
            print("[OK] Model loaded with tf.keras")
            return True, None
        except Exception as e_tf:
            print(f"[ERROR] tf.keras load failed: {str(e_tf)}")
            print("[DEBUG] TF traceback:")
            print(traceback.format_exc())
            print("Trying standalone keras...")
            try:
                keras = lazy_import_keras()
                print(f"[DEBUG] Standalone Keras version: {keras.__version__}")
                custom_objects = {'InputLayer': keras.layers.InputLayer}
                model_obj = keras.models.load_model(MODEL_PATH, compile=False, custom_objects=custom_objects)
                model = model_obj
                print("[OK] Model loaded with standalone keras")
                return True, None
            except Exception as e:
                print(f"[ERROR] Standalone keras load failed: {str(e)}")
                print("[DEBUG] Keras traceback:")
                print(traceback.format_exc())
                return False, str(e)

# Labels (no deps)
labels: Optional[Dict[int, str]] = None
if os.path.exists(LABELS_PATH):
    try:
        with open(LABELS_PATH, 'r') as f:
            labels_data = json.load(f)
            labels = {int(k): v for k, v in labels_data.items()}
        print("[OK] Labels loaded:", labels)
    except Exception:
        pass

def preprocess_image(image_bytes: bytes) -> Any:
    import io
    Image = lazy_import_pil()
    np = lazy_import_numpy()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224), Image.BILINEAR)
    image = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(image, 0)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    ok, err = ensure_model_loaded()
    if not ok:
        return jsonify({"error": "Model load failed", "details": err}), 500

    np = lazy_import_numpy()
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file"}), 400
        file = request.files['file']
        if file.filename == "":
            return jsonify({"error": "No filename"}), 400

        processed = preprocess_image(file.read())
        prediction = model.predict(processed, verbose=0)
        
        pred_class = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]))
        
        resp = {
            "class_index": pred_class,
            "confidence": round(confidence, 4),
            "raw": prediction.tolist()
        }
        if labels and pred_class in labels:
            resp["label"] = labels[pred_class]
        return jsonify(resp)
    except Exception as e:
        import traceback
        return jsonify({"error": str(e)}), 500

@app.route('/healthz')
def healthz():
    return jsonify({'status': 'ok', 'model_ready': model is not None})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 7860))
    app.run(host='0.0.0.0', port=port, debug=False)
