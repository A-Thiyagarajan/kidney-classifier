from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import json
import traceback
import threading
from typing import Optional, Any, Tuple, Dict
import tensorflow as tf

print("[INFO] App loaded - FIXED: tf2.19 + LegacyInputLayer for old model")

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Kidney.h5")
LABELS_PATH = os.path.join(BASE_DIR, "label.json")

model: Optional[Any] = None
_model_lock = threading.Lock()

class LegacyInputLayer(tf.keras.layers.InputLayer):
    def __init__(self, **kwargs):
        if 'batch_shape' in kwargs:
            kwargs['input_shape'] = kwargs.pop('batch_shape')[1:]
        super().__init__(**kwargs)

def ensure_model_loaded() -> Tuple[bool, Optional[str]]:
    global model
    if model is not None:
        return True, None
    with _model_lock:
        print(f"[DEBUG] Model load: {MODEL_PATH}")
        print(f"[DEBUG] File: exists={os.path.exists(MODEL_PATH)}, size={os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0}")
        if model is not None:
            return True, None
        try:
            print(f"[DEBUG] TF: {tf.__version__}, Keras: {tf.keras.__file__}")
            custom_objects = {'InputLayer': LegacyInputLayer}
            model_obj = tf.keras.models.load_model(MODEL_PATH, compile=False, custom_objects=custom_objects)
            model = model_obj
            print("[OK] Model loaded!")
            return True, None
        except Exception as e:
            print(f"[ERROR] Load failed: {str(e)}")
            traceback.print_exc()
            return False, str(e)

labels = None
if os.path.exists(LABELS_PATH):
    try:
        with open(LABELS_PATH, 'r') as f:
            labels_data = json.load(f)
            labels = {int(k): v for k, v in labels_data.items()}
        print(f"[OK] Labels: {labels}")
    except:
        pass

def preprocess_image(image_bytes):
    import io
    from PIL import Image
    import numpy as np
    print("[DEBUG] Preprocess start")
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32) / 255.0
    processed = np.expand_dims(image, 0)
    print(f"[DEBUG] Preprocess OK, shape: {processed.shape}")
    return processed

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/favicon.ico")
def favicon():
    return "", 204

@app.route("/predict", methods=["POST"])
def predict():
    print("[DEBUG] Predict start")
    ok, err = ensure_model_loaded()
    if not ok:
        return jsonify({"error": "Model failed", "details": err}), 500

    import numpy as np
    try:
        print("[DEBUG] File check")
        if 'file' not in request.files:
            return jsonify({"error": "No file"}), 400
        file = request.files['file']
        if not file.filename:
            return jsonify({"error": "No filename"}), 400
        print(f"[DEBUG] File OK: {file.filename}")

        processed = preprocess_image(file.read())
        print("[DEBUG] Model predict")
        prediction = model.predict(processed, verbose=0)
        print(f"[DEBUG] Prediction shape: {prediction.shape}")

        pred_class = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]))
        
        resp = {
            "class_index": pred_class,
            "confidence": round(confidence, 4),
            "raw": prediction[0].tolist()
        }
        if labels:
            resp["label"] = labels.get(pred_class, "Unknown")
        print(f"[OK] PREDICTION: {resp['label']} ({confidence:.2%})")
        return jsonify(resp)
    except Exception as e:
        print(f"[ERROR] Predict failed: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/healthz')
def healthz():
    return jsonify({'status': 'ok', 'model_ready': model is not None})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 7860))
    app.run(host='0.0.0.0', port=port, debug=False)

