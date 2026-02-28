"""
Kidney Disease Prediction Module
Make predictions using the trained Kidney.h5 model
"""

import numpy as np
from model_loader import KidneyModelLoader
from image_preprocessing import preprocess_image, load_images_batch


class KidneyDiseasePredictor:
    """Predict kidney disease from CT scan images"""
    
    def __init__(self, model_path='Kidney.h5', labels_path='label.json'):
        """
        Initialize the predictor
        
        Args:
            model_path: Path to the Kidney.h5 model file
            labels_path: Path to the labels JSON file
        """
        self.loader = KidneyModelLoader(model_path, labels_path)
        self.loader.load_model()
        self.loader.load_labels()
        
        self.model = self.loader.model
        self.labels = self.loader.labels
    
    def predict_single(self, image_path):
        """
        Predict kidney disease from a single image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with prediction results:
            {
                'predicted_class': int (0-3),
                'class_name': str,
                'confidence': float (0-1),
                'probabilities': dict {class_name: probability, ...}
            }
        """
        try:
            # Preprocess image
            _, image_batch = preprocess_image(image_path)
            
            # Make prediction
            predictions = self.model.predict(image_batch, verbose=0)
            prediction = predictions[0]
            
            # Get predicted class
            predicted_class_id = np.argmax(prediction)
            predicted_class_name = self.labels[predicted_class_id]
            confidence = float(prediction[predicted_class_id])
            
            # Build result
            result = {
                'predicted_class': int(predicted_class_id),
                'class_name': predicted_class_name,
                'confidence': confidence,
                'probabilities': {
                    self.labels[i]: float(prediction[i]) 
                    for i in range(len(self.labels))
                }
            }
            
            return result
        
        except Exception as e:
            return {'error': str(e)}
    
    def predict_batch(self, image_paths):
        """
        Predict on multiple images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        try:
            # Load all images in batch
            image_batch, valid_paths = load_images_batch(image_paths)
            
            # Make predictions
            predictions = self.model.predict(image_batch, verbose=0)
            
            # Process each prediction
            for idx, prediction in enumerate(predictions):
                predicted_class_id = np.argmax(prediction)
                predicted_class_name = self.labels[predicted_class_id]
                confidence = float(prediction[predicted_class_id])
                
                result = {
                    'file': valid_paths[idx],
                    'predicted_class': int(predicted_class_id),
                    'class_name': predicted_class_name,
                    'confidence': confidence,
                    'probabilities': {
                        self.labels[i]: float(prediction[i]) 
                        for i in range(len(self.labels))
                    }
                }
                results.append(result)
            
            return results
        
        except Exception as e:
            return [{'error': str(e)}]
    
    def print_result(self, result, image_path=''):
        """Print prediction result in readable format"""
        if 'error' in result:
            print(f"[ERROR] {result['error']}")
            return
        
        print("\n" + "=" * 60)
        print("KIDNEY DISEASE PREDICTION RESULT")
        print("=" * 60)
        
        if image_path:
            print(f"Image: {image_path}")
        
        print(f"\nPredicted Class: {result['class_name']} (ID: {result['predicted_class']})")
        print(f"Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        
        print("\nAll Class Probabilities:")
        for class_name, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
            bar = '█' * int(prob * 40)
            print(f"  {class_name:10s}: {prob:.6f} {bar}")
        
        print("=" * 60)


if __name__ == '__main__':
    # Example usage
    predictor = KidneyDiseasePredictor()
    
    # Show model info
    predictor.loader.print_model_info()
    
    # Predict on sample image if it exists
    import os
    if os.path.exists('c.jpg'):
        print("\nMaking prediction on c.jpg...")
        result = predictor.predict_single('c.jpg')
        predictor.print_result(result, 'c.jpg')
    else:
        print("\n[INFO] No sample image found (c.jpg)")
        print("[INFO] To make a prediction, call:")
        print("       predictor.predict_single('path/to/image.jpg')")
