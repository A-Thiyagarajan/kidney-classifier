from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import json

app = Flask(__name__)

# Lazy model load with fallbacks
MODEL_PATH = "Kidney.h5"
model = None
labels = None
if os.path.exists('labels.json'):
    try:
        with open('labels.json','r') as f:
            labels = json.load(f)
            # labels is mapping index->name, ensure keys are ints
            labels = {int(k): v for k, v in labels.items()}
            print('Loaded labels.json')
    except Exception as e:
        print('Failed to load labels.json:', e)

def load_model_with_fallback(path=MODEL_PATH):
    global model
    if model is not None:
        return model
    last_err = None
    try:
        model = tf.keras.models.load_model(path)
        print("Loaded model with tf.keras")
        return model
    except Exception as e_tf:
        last_err = e_tf
        import traceback as _tb
        tb_tf = _tb.format_exc()
        print("tf.keras failed to load model:", e_tf)
        print(tb_tf)
    try:
        # try standalone keras
        import keras as _keras
        model = _keras.models.load_model(path, compile=False)
        print("Loaded model with standalone keras")
        return model
    except Exception as e_keras:
        import traceback as _tb
        tb_keras = _tb.format_exc()
        print("standalone keras failed to load model:", e_keras)
        print(tb_keras)
        last_err = e_keras
    raise RuntimeError(f"Failed to load model from {path}. Last error: {last_err}\ntf.keras traceback:\n{tb_tf}\nstandalone keras traceback:\n{tb_keras}")

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))  # change to your model input size if needed
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "no file provided"}), 400
    file = request.files["file"]
    try:
        image = Image.open(io.BytesIO(file.read()))
    except Exception:
        return jsonify({"error": "invalid image"}), 400

    # ensure model is loaded (lazy)
    try:
        load_model_with_fallback()
    except Exception as e:
        return jsonify({"error": "model load failed", "details": str(e)}), 500

    processed = preprocess_image(image)
    prediction = model.predict(processed)
    result = int(np.argmax(prediction, axis=1)[0])
    out = {"prediction": result, "raw": prediction.tolist()}
    if labels and result in labels:
        out['label'] = labels[result]
    return jsonify(out)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")