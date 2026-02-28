"""
Kidney Classification Model - Training Script (Reference Only)
Trains a VGG16-based transfer learning model for kidney disease classification

Dataset: CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone
Note: This is a reference script. The model (Kidney.h5) is already trained 
and included in this repository.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import os
import json

from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

# ==============================
# PARAMETERS
# ==============================
batch_size = 32
img_height = 224
img_width = 224
epochs = 10

# Update this path to your local dataset directory
data_dir = 'path/to/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone'
# For Kaggle notebook:
# data_dir = '/kaggle/input/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone'

# ==============================
# LOAD DATASET
# ==============================
print(f"Loading dataset from: {data_dir}")

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=(img_height, img_width),
    validation_split=0.1,
    subset='training',
    seed=123,
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=(img_height, img_width),
    validation_split=0.1,
    subset='validation',
    seed=123,
    batch_size=batch_size
)

class_names = train_ds.class_names
label_to_class_name = dict(zip(range(len(class_names)), class_names))

print("[OK] Dataset loaded")
print(f"Classes: {class_names}")
print(f"Number of classes: {len(class_names)}")

# ==============================
# NORMALIZATION
# ==============================
print("\nNormalizing images...")

train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
val_ds = val_ds.map(lambda x, y: (x / 255.0, y))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

print("[OK] Images normalized to [0, 1]")

# ==============================
# LOAD VGG16 BASE
# ==============================
print("\nLoading VGG16 base model with ImageNet weights...")

vgg16_base = VGG16(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base layers (transfer learning)
for layer in vgg16_base.layers:
    layer.trainable = False

print("[OK] VGG16 base loaded and frozen")
print(f"VGG16 parameters: {vgg16_base.count_params():,}")

# ==============================
# BUILD CUSTOM MODEL
# ==============================
print("\nBuilding custom model...")

inputs = tf.keras.Input(shape=(img_height, img_width, 3))
x = vgg16_base(inputs, training=False)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)

model = tf.keras.Model(inputs, outputs, name='kidney_classification_model')

print("[OK] Model architecture created")

# ==============================
# COMPILE
# ==============================
print("\nCompiling model...")

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("[OK] Model compiled")

# ==============================
# MODEL SUMMARY
# ==============================
model.summary()

# ==============================
# TRAIN
# ==============================
print(f"\nTraining model for {epochs} epochs...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    verbose=1
)

print("\n[OK] Training complete")

# ==============================
# EVALUATE
# ==============================
print("\nEvaluating on validation set...")
val_loss, val_accuracy = model.evaluate(val_ds)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# ==============================
# SAVE MODEL
# ==============================
print("\nSaving model...")

model.save("Kidney.h5")
print("[OK] Model saved as Kidney.h5")

# ==============================
# SAVE LABELS
# ==============================
print("Saving labels...")

with open("label.json", "w") as f:
    json.dump(label_to_class_name, f, indent=2)

print("[OK] Labels saved as label.json")
print(f"Label mapping: {label_to_class_name}")

# ==============================
# TRAINING COMPLETE
# ==============================
print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
print(f"\nModel files created:")
print(f"  - Kidney.h5 (main model)")
print(f"  - label.json (class labels)")
print(f"\nTo deploy:")
print(f"  - Ensure Kidney.h5 and label.json are in the repository")
print(f"  - Push to GitHub")
print(f"  - Trigger Render deployment")
print("="*70)
