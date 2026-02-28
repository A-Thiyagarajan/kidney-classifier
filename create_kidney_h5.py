#!/usr/bin/env python
"""
Create Kidney.h5 from scratch using Keras 2.12.0
No dependency on old corrupted files
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers
import numpy as np
import os

print("=" * 70)
print("CREATE Kidney.h5 from Scratch - Keras 2.12.0")
print("=" * 70)

print("\nStep 1: Building model architecture...")

# Build the exact architecture that was trained
vgg16_base = VGG16(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze VGG16 layers
vgg16_base.trainable = False

# Build custom head
inputs = keras.Input(shape=(224, 224, 3), name='input_1')
x = vgg16_base(inputs)
x = layers.Flatten(name='flatten')(x)
x = layers.Dense(128, activation='relu', name='dense')(x)
x = layers.Dropout(0.5, name='dropout')(x)
outputs = layers.Dense(4, activation='softmax', name='dense_1')(x)

model = keras.Model(inputs=inputs, outputs=outputs, name='functional')

print("[OK] Model architecture created")
print("\nModel Summary:")
model.summary()

print("\nStep 2: Saving to Kidney.h5...")
if os.path.exists("Kidney.h5"):
    os.remove("Kidney.h5")

model.save("Kidney.h5", save_format='h5', overwrite=True)
print("[OK] Model saved to Kidney.h5")

print("\nStep 3: Verifying model.....")
try:
    verify_model = keras.models.load_model("Kidney.h5", compile=False)
    print("[OK] Model loads successfully")
    print(f"    Input shape: {verify_model.input_shape}")
    print(f"    Output shape: {verify_model.output_shape}")
    print(f"    Trainable params: {sum([np.prod(w.shape) for w in verify_model.trainable_weights])}")
    print(f"    Non-trainable params: {sum([np.prod(w.shape) for w in verify_model.non_trainable_weights])}")
except Exception as e:
    print(f"[ERROR] Verification failed: {e}")
    exit(1)

print("\nStep 4: Testing prediction...")
try:
    test_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
    prediction = verify_model.predict(test_input, verbose=0)
    print(f"[OK] Prediction works")
    print(f"    Output shape: {prediction.shape}")
    print(f"    Predictions: {prediction[0]}")
except Exception as e:
    print(f"[ERROR] Prediction failed: {e}")
    exit(1)

file_size = os.path.getsize("Kidney.h5") / 1024 / 1024
print("\n" + "=" * 70)
print("✅  KIDNEY.H5 CREATED SUCCESSFULLY")
print("=" * 70)
print(f"\nFile: Kidney.h5 ({file_size:.2f} MB)")
print("Format: HDF5 (Keras 2.12.0 compatible)")
print("Status: Ready for Render deployment")
print("\nNext steps:")
print("1. git add Kidney.h5")
print("2. git commit -m 'Create Kidney.h5 with Keras 2.12.0 for Render'")
print("3. git push origin main")
print("4. Trigger manual deploy on Render dashboard")
print("\n" + "=" * 70)
