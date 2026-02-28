"""
Model Loader Module
Load and manage the Kidney.h5 model for kidney disease classification
"""

import tensorflow as tf
from tensorflow import keras
import json
import os


class KidneyModelLoader:
    """Load and manage the Kidney classification model"""
    
    def __init__(self, model_path='Kidney.h5', labels_path='label.json'):
        """
        Initialize the model loader
        
        Args:
            model_path: Path to the Kidney.h5 model file
            labels_path: Path to the labels JSON file
        """
        self.model_path = model_path
        self.labels_path = labels_path
        self.model = None
        self.labels = None
        
    def load_model(self):
        """Load the Keras model from file"""
        try:
            self.model = keras.models.load_model(self.model_path, compile=False)
            print(f"[OK] Model loaded successfully from {self.model_path}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            return False
    
    def load_labels(self):
        """Load class labels from JSON file"""
        try:
            with open(self.labels_path, 'r') as f:
                labels_dict = json.load(f)
            
            # Convert string keys to integers
            self.labels = {int(k): v for k, v in labels_dict.items()}
            print(f"[OK] Labels loaded successfully: {self.labels}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load labels: {e}")
            return False
    
    def print_model_info(self):
        """Print detailed model information"""
        if self.model is None:
            print("[ERROR] Model not loaded")
            return
        
        print("\n" + "=" * 70)
        print("MODEL INFORMATION")
        print("=" * 70)
        
        total_params = sum([tf.size(w).numpy() for w in self.model.weights])
        trainable_params = sum([tf.size(w).numpy() for w in self.model.trainable_weights])
        non_trainable_params = sum([tf.size(w).numpy() for w in self.model.non_trainable_weights])
        
        print(f"\nInput Shape: {self.model.input_shape}")
        print(f"Output Shape: {self.model.output_shape}")
        print(f"\nParameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print(f"  Non-trainable: {non_trainable_params:,}")
        
        if self.labels:
            print(f"\nClasses ({len(self.labels)}):")
            for class_id, class_name in self.labels.items():
                print(f"  {class_id}: {class_name}")
        
        print("\n" + "=" * 70)
    
    def get_model_summary(self):
        """Get model summary"""
        if self.model is None:
            return "Model not loaded"
        
        return self.model.summary()


if __name__ == '__main__':
    # Example usage
    loader = KidneyModelLoader()
    
    # Load model and labels
    if loader.load_model() and loader.load_labels():
        loader.print_model_info()
    else:
        print("[ERROR] Failed to initialize model")
