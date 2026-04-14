import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
import io

print(f'TF version: {tf.__version__}')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'Kidney.h5')
LABELS_PATH = os.path.join(BASE_DIR, 'label.json')

class LegacyInputLayer(tf.keras.layers.InputLayer):
    def __init__(self, **kwargs):
        if 'batch_shape' in kwargs:
            kwargs['input_shape'] = kwargs.pop('batch_shape')[1:]
        super().__init__(**kwargs)

print(f'Model path: {MODEL_PATH}, exists: {os.path.exists(MODEL_PATH)}')
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, 'r') as f:
        labels = {int(k): v for k, v in json.load(f).items()}
    print('Labels:', labels)
else:
    labels = None
    print('No label.json')

# Load model
print('Loading model...')
custom_objects = {'InputLayer': LegacyInputLayer}
model = tf.keras.models.load_model(MODEL_PATH, compile=False, custom_objects=custom_objects)
print('Model loaded OK!')

# Synthetic test image (224x224 RGB)
print('Creating synthetic test image...')
img_array = np.random.uniform(0, 1, (224, 224, 3)).astype(np.float32)
processed = np.expand_dims(img_array, 0)
print(f'Input shape: {processed.shape}')

# Predict
print('Running predict...')
prediction = model.predict(processed, verbose=0)
pred_class = int(np.argmax(prediction[0]))
confidence = float(np.max(prediction[0]))
label = labels[pred_class] if labels else f'Class {pred_class}'

print(f'Prediction: {label} (confidence: {confidence:.4f})')
print('TEST SUCCESS - Model works!')
print('Next: activate .venv, pip install requirements-fixed.txt, python app.py')
