#!/usr/bin/env python
"""
Re-save Kidney.h5 using Keras 2.12.0 format
Ensures compatibility with Render deployment
"""

import tensorflow as tf
from tensorflow import keras
import os

print("=" * 70)
print("CONVERT Kidney.h5 to Keras 2.12.0 Compatible Format")
print("=" * 70)

print("\nLoading current Kidney.h5...")
try:
    # Try loading with custom_objects and safe loading
    model = keras.models.load_model("Kidney.h5", compile=False, safe_mode=False)
    print("[OK] Model loaded")
except Exception as e1:
    print(f"[WARNING] Standard load failed: {e1}")
    print("Trying alternative loading method...")
    try:
        # Try with tf.keras
        model = tf.keras.models.load_model("Kidney.h5", compile=False)
        print("[OK] Model loaded with tf.keras")
    except Exception as e2:
        print(f"[ERROR] All loading methods failed: {e1} / {e2}")
        exit(1)

print("\nModel info:")
print(f"  Input shape: {model.input_shape}")
print(f"  Output shape: {model.output_shape}")
print(f"  Parameters: {model.count_params():,}")

print("\nRe-saving with Keras 2.12.0...")
if os.path.exists("Kidney.h5"):
    os.remove("Kidney.h5")

# Save using TensorFlow's method to ensure compatibility
model.save("Kidney.h5", save_format='h5', overwrite=True)
print("[OK] Model re-saved")

print("\nVerifying re-saved model...")
try:
    verify = keras.models.load_model("Kidney.h5", compile=False, safe_mode=False)
    print("[OK] Verification successful!")
    print(f"    File size: {os.path.getsize('Kidney.h5') / 1024 / 1024:.2f} MB")
except Exception as e:
    print(f"[ERROR] Verification failed: {e}")
    exit(1)

print("\n" + "=" * 70)
print("CONVERSION COMPLETE ✅")
print("=" * 70)
print("\nKidney.h5 is now in Keras 2.12.0 compatible format")
print("\nNext steps:")
print("1. git add Kidney.h5")
print("2. git commit -m 'Save Kidney.h5 in Keras 2.12.0 compatible format'")
print("3. git push origin main")
print("4. Trigger manual deploy on Render")
print("\n" + "=" * 70)
