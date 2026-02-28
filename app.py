from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import json

app = Flask(__name__)
# limit upload size to 5MB to avoid Render free-tier request-size failures
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

# Lazy model load with fallbacks
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Kidney.h5")
model = None
labels = None
# Load labels from workspace-relative path so Render finds them reliably
LABELS_PATH = os.path.join(BASE_DIR, 'labels.json')
if os.path.exists(LABELS_PATH):
    try:
        with open(LABELS_PATH, 'r') as f:
            labels = json.load(f)
            labels = {int(k): v for k, v in labels.items()}
            print('Loaded labels.json from', LABELS_PATH)
    except Exception as e:
        print('Failed to load labels.json:', e)

def load_model_with_fallback(path=MODEL_PATH):
    global model
    if model is not None:
        return model
    last_err = None
    # Prefer tf.keras but avoid compiling (not needed for inference)
    try:
        model = tf.keras.models.load_model(path, compile=False)
        print("Loaded model with tf.keras from", path)
        return model
    except Exception as e_tf:
        last_err = e_tf
        import traceback as _tb
        tb_tf = _tb.format_exc()
        print("tf.keras failed to load model:", e_tf)
        print(tb_tf)
    try:
        # try standalone keras as a fallback
        import keras as _keras
        model = _keras.models.load_model(path, compile=False)
        print("Loaded model with standalone keras from", path)
        return model
    except Exception as e_keras:
        import traceback as _tb
        tb_keras = _tb.format_exc()
        print("standalone keras failed to load model:", e_keras)
        print(tb_keras)
        last_err = e_keras
    raise RuntimeError(f"Failed to load model from {path}. Last error: {last_err}")

def preprocess_image(image):
    image = image.convert("RGB")
    # explicit resize and dtype to match training preprocessing
    image = image.resize((224, 224), resample=Image.BILINEAR)  # change to your model input size if needed
    arr = np.asarray(image).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "no file provided"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "no file provided"}), 400
    try:
        # open directly from the uploaded file and ensure RGB mode
        image = Image.open(file).convert("RGB")
    except Exception:
        return jsonify({"error": "invalid image"}), 400

    # ensure model is loaded (load once per worker)
    try:
        load_model_with_fallback()
    except Exception as e:
        return jsonify({"error": "model load failed", "details": str(e)}), 500

    processed = preprocess_image(image)
    # force float32 and explicit batch size
    processed = np.asarray(processed, dtype=np.float32)
    prediction = model.predict(processed, batch_size=1)
    print("RAW PREDICTION:", prediction)
    result = int(np.argmax(prediction, axis=1)[0])
    out = {"prediction": result, "raw": prediction.tolist()}
    if labels and result in labels:
        out['label'] = labels[result]
    return jsonify(out)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")