#!/usr/bin/env python
"""
Rebuild Kidney.h5 using TensorFlow 2.12.0 + Keras 2.12.0
Ensures compatibility with Render deployment
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers
import h5py
import json
import os
import zipfile
import tempfile

print("=" * 70)
print("REBUILD Kidney.h5 - TensorFlow 2.12.0 Compatible")
print("=" * 70)

# Check if source model exists
if not os.path.exists("kidneymodels.keras"):
    print("\n[ERROR] kidneymodels.keras not found!")
    print("You need the original trained model to extract weights.")
    exit(1)

print("\nStep 1: Extract weights from kidneymodels.keras...")
with tempfile.TemporaryDirectory() as tmpdir:
    with zipfile.ZipFile("kidneymodels.keras", 'r') as z:
        z.extractall(tmpdir)
    
    weights_path = os.path.join(tmpdir, "model.weights.h5")
    
    print("Step 2: Create new model architecture...")
    # Rebuild with same architecture
    vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    vgg16_base.trainable = False
    
    inputs = keras.Input(shape=(224, 224, 3))
    x = vgg16_base(inputs)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(4, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='functional')
    print("[OK] Model architecture rebuilt")
    
    print("\nStep 3: Extract weights from source model...")
    weights_dict = {}
    with h5py.File(weights_path, 'r') as wf:
        if 'layers' in wf:
            layers_group = wf['layers']
            
            # Extract dense layer weights
            for layer_name in ['dense', 'dense_1']:
                if layer_name in layers_group:
                    layer_group = layers_group[layer_name]
                    if 'vars' in layer_group:
                        weights_dict[layer_name] = []
                        vars_group = layer_group['vars']
                        for key in sorted(vars_group.keys()):
                            item = vars_group[key]
                            if isinstance(item, h5py.Dataset):
                                weights_dict[layer_name].append(item[()])
            
            # Extract VGG16 weights
            if 'functional' in layers_group:
                functional_group = layers_group['functional']
                if 'layers' in functional_group:
                    vgg_conv_layers = functional_group['layers']
                    
                    # Mapping from conv2d_X to block*_conv*
                    conv2d_to_block = {
                        'conv2d': 'block1_conv1',
                        'conv2d_1': 'block1_conv2',
                        'conv2d_2': 'block2_conv1',
                        'conv2d_3': 'block2_conv2',
                        'conv2d_4': 'block3_conv1',
                        'conv2d_5': 'block3_conv2',
                        'conv2d_6': 'block3_conv3',
                        'conv2d_7': 'block4_conv1',
                        'conv2d_8': 'block4_conv2',
                        'conv2d_9': 'block4_conv3',
                        'conv2d_10': 'block5_conv1',
                        'conv2d_11': 'block5_conv2',
                        'conv2d_12': 'block5_conv3',
                    }
                    
                    vgg_layer = model.get_layer('vgg16')
                    
                    for conv_layer_name in vgg_conv_layers.keys():
                        if conv_layer_name in conv2d_to_block:
                            layer_group = vgg_conv_layers[conv_layer_name]
                            if 'vars' in layer_group:
                                vars_group = layer_group['vars']
                                weights = []
                                for key in sorted(vars_group.keys()):
                                    item = vars_group[key]
                                    if isinstance(item, h5py.Dataset):
                                        weights.append(item[()])
                                
                                block_layer_name = conv2d_to_block[conv_layer_name]
                                try:
                                    vgg_internal_layer = vgg_layer.get_layer(block_layer_name)
                                    if vgg_internal_layer and weights:
                                        vgg_internal_layer.set_weights(weights)
                                        print(f"  [OK] Loaded {block_layer_name}")
                                except Exception as e:
                                    print(f"  [SKIP] {block_layer_name}: {e}")
    
    print("\nStep 4: Apply extracted weights to model...")
    
    # Apply dense layer weights
    for layer_name in ['dense', 'dense_1']:
        if layer_name in weights_dict:
            layer = model.get_layer(layer_name)
            if layer:
                layer.set_weights(weights_dict[layer_name])
                print(f"  [OK] Applied {layer_name} weights")
    
    print("\nStep 5: Save to Kidney.h5...")
    # Remove old file if exists
    if os.path.exists("Kidney.h5"):
        os.remove("Kidney.h5")
    
    model.save("Kidney.h5", save_format='h5')
    print("[OK] Kidney.h5 saved")

print("\nStep 6: Verify saved model...")
try:
    verify_model = keras.models.load_model("Kidney.h5", compile=False)
    print("[OK] Model loads successfully!")
    print(f"    Input shape: {verify_model.input_shape}")
    print(f"    Output shape: {verify_model.output_shape}")
    print(f"    Total parameters: {verify_model.count_params():,}")
except Exception as e:
    print(f"[ERROR] Model verification failed: {e}")
    exit(1)

print("\n" + "=" * 70)
print("REBUILD COMPLETE ✅")
print("=" * 70)
print("\nKidney.h5 is now compatible with TensorFlow 2.12.0 + Keras 2.12.0")
print("\nNext steps:")
print("1. git add Kidney.h5")
print("2. git commit -m 'Rebuild Kidney.h5 for Keras 2.12.0 compatibility'")
print("3. git push origin main")
print("4. Manual deploy on Render dashboard")
print("\n" + "=" * 70)
