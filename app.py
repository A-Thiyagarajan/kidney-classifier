from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Kidney.h5")
LABELS_PATH = os.path.join(BASE_DIR, "labels.json")

# Load model once at startup
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully")

# Load labels
labels = None
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)
        labels = {int(k): v for k, v in labels.items()}
    print("Labels loaded:", labels)


# ✅ EXACT SAME PREPROCESSING AS TRAINING
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))  # must match training
    image = np.array(image, dtype=np.float32)
    image = image / 255.0             # same as training
    image = np.expand_dims(image, axis=0)
    return image


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        image = Image.open(file)
    except:
        return jsonify({"error": "Invalid image"}), 400

    processed = preprocess_image(image)

    prediction = model.predict(processed)
    print("RAW PREDICTION:", prediction)

    predicted_class = int(np.argmax(prediction, axis=1)[0])
    confidence = float(np.max(prediction))

    response = {
        "class_index": predicted_class,
        "confidence": confidence,
        "raw": prediction.tolist()
    }

    if labels and predicted_class in labels:
        response["label"] = labels[predicted_class]

    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)