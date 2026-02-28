"""
Usage Examples for Kidney Classification Model
Demonstrates various ways to use the Kidney.h5 model
"""

import json
from predict import KidneyDiseasePredictor
from image_preprocessing import find_images


def example_1_single_prediction():
    """Example 1: Make a prediction on a single image"""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Single Image Prediction")
    print("=" * 70)
    
    predictor = KidneyDiseasePredictor()
    
    # Try to predict on sample image
    import os
    if os.path.exists('c.jpg'):
        print("\nPredicting on c.jpg...")
        result = predictor.predict_single('c.jpg')
        predictor.print_result(result, 'c.jpg')
    else:
        print("\n[INFO] Sample image c.jpg not found")
        print("[INFO] Usage: result = predictor.predict_single('path/to/image.jpg')")


def example_2_batch_prediction():
    """Example 2: Batch prediction on multiple images"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Batch Prediction")
    print("=" * 70)
    
    predictor = KidneyDiseasePredictor()
    
    # Find all images in current directory
    image_files = find_images('.')
    
    if not image_files:
        print("\n[INFO] No images found in current directory")
        print("[INFO] Usage: results = predictor.predict_batch(image_paths)")
        return
    
    print(f"\nFound {len(image_files)} images")
    print("\nRunning batch prediction...")
    
    try:
        results = predictor.predict_batch(image_files[:5])  # Limit to first 5
        
        print(f"\nResults ({len(results)} images):")
        print("-" * 70)
        for result in results:
            if 'error' not in result:
                filename = result['file'].split('\\')[-1] if '\\' in result['file'] else result['file']
                print(f"{filename:20s} -> {result['class_name']:10s} (confidence: {result['confidence']:.4f})")
            else:
                print(f"[ERROR] {result['error']}")
    except Exception as e:
        print(f"[ERROR] Batch prediction failed: {e}")


def example_3_json_output():
    """Example 3: Get prediction as JSON for API/storage"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: JSON Output Format")
    print("=" * 70)
    
    predictor = KidneyDiseasePredictor()
    
    import os
    if os.path.exists('c.jpg'):
        print("\nPredicting on c.jpg...")
        result = predictor.predict_single('c.jpg')
        
        if 'error' not in result:
            print("\nJSON Output (suitable for API responses):")
            print(json.dumps(result, indent=2))
        else:
            print(f"[ERROR] {result['error']}")
    else:
        print("\n[INFO] Sample image c.jpg not found")


def example_4_threshold_filtering():
    """Example 4: Filter predictions by confidence threshold"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Confidence Threshold Filtering")
    print("=" * 70)
    
    predictor = KidneyDiseasePredictor()
    threshold = 0.8  # Only accept predictions with 80%+ confidence
    
    image_files = find_images('.')
    
    if not image_files:
        print("\n[INFO] No images found")
        return
    
    print(f"\nFiltering predictions by confidence >= {threshold}")
    
    try:
        results = predictor.predict_batch(image_files[:5])
        
        high_confidence = [r for r in results if 'error' not in r and r['confidence'] >= threshold]
        low_confidence = [r for r in results if 'error' not in r and r['confidence'] < threshold]
        
        print(f"\nHigh Confidence (>= {threshold}): {len(high_confidence)}")
        for result in high_confidence:
            filename = result['file'].split('\\')[-1] if '\\' in result['file'] else result['file']
            print(f"  {filename:20s} -> {result['class_name']:10s} ({result['confidence']:.4f})")
        
        if low_confidence:
            print(f"\nLow Confidence (< {threshold}): {len(low_confidence)}")
            for result in low_confidence:
                filename = result['file'].split('\\')[-1] if '\\' in result['file'] else result['file']
                print(f"  {filename:20s} -> {result['class_name']:10s} ({result['confidence']:.4f})")
    
    except Exception as e:
        print(f"[ERROR] {e}")


def example_5_model_info():
    """Example 5: Display detailed model information"""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Model Information")
    print("=" * 70)
    
    predictor = KidneyDiseasePredictor()
    predictor.loader.print_model_info()


def example_6_programmatic_usage():
    """Example 6: Using the model programmatically in your code"""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Programmatic Usage")
    print("=" * 70)
    
    print("""
# Import the predictor class
from predict import KidneyDiseasePredictor

# Initialize the predictor (loads model automatically)
predictor = KidneyDiseasePredictor()

# Single prediction
result = predictor.predict_single('kidney_scan.jpg')
print(f"Diagnosis: {result['class_name']}")
print(f"Confidence: {result['confidence']:.2%}")

# Check specific class probabilities
probs = result['probabilities']
print(f"Normal kidney: {probs['Normal']:.2%}")
print(f"Cyst: {probs['Cyst']:.2%}")
print(f"Stone: {probs['Stone']:.2%}")
print(f"Tumor: {probs['Tumor']:.2%}")

# Batch prediction
results = predictor.predict_batch(['img1.jpg', 'img2.jpg', 'img3.jpg'])
for r in results:
    print(f"{r['file']}: {r['class_name']} ({r['confidence']:.1%})")
    """)


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("KIDNEY CLASSIFICATION MODEL - USAGE EXAMPLES")
    print("=" * 70)
    print(f"\nModel File: Kidney.h5")
    print(f"Framework: TensorFlow 2.12.0 / Keras 2.12.0")
    print(f"Classes: Cyst, Normal, Stone, Tumor")
    
    # Run examples
    example_5_model_info()
    example_1_single_prediction()
    example_2_batch_prediction()
    example_3_json_output()
    example_4_threshold_filtering()
    example_6_programmatic_usage()
    
    print("\n" + "=" * 70)
    print("END OF EXAMPLES")
    print("=" * 70)
