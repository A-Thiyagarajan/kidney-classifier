"""
API Usage Examples
Shows how to interact with the Kidney model via REST API
"""

import requests
import json
from predict import KidneyDiseasePredictor


def example_local_api():
    """Example: Call the locally running Flask API"""
    print("\n" + "=" * 70)
    print("LOCAL API USAGE")
    print("=" * 70)
    
    print("""
Step 1: Start the Flask server
    python app.py

Step 2: In another terminal, send a prediction request:

    # Using curl
    curl -X POST -F "file=@kidney_scan.jpg" http://localhost:5000/predict
    
    # Using Python requests
    import requests
    files = {'file': open('kidney_scan.jpg', 'rb')}
    response = requests.post('http://localhost:5000/predict', files=files)
    result = response.json()
    print(f"Prediction: {result['class_name']}")
    """)


def example_render_api():
    """Example: Call the deployed Render API"""
    print("\n" + "=" * 70)
    print("RENDER CLOUD API USAGE")
    print("=" * 70)
    
    api_url = 'https://kidney-classifier-1.onrender.com/predict'
    
    print(f"\nAPI Endpoint: {api_url}")
    print("\nUsing curl:")
    print(f'    curl -X POST -F "file=@kidney_scan.jpg" {api_url}')
    
    print("\nUsing Python requests:")
    print(f"""
    import requests
    
    files = {{'file': open('kidney_scan.jpg', 'rb')}}
    response = requests.post('{api_url}', files=files)
    result = response.json()
    
    print(f"Diagnosis: {{result['class_name']}}")
    print(f"Confidence: {{result['confidence']:.2%}}")
    """)


def test_prediction_format():
    """Show the format of prediction responses"""
    print("\n" + "=" * 70)
    print("PREDICTION RESPONSE FORMAT")
    print("=" * 70)
    
    # Load model to show actual format
    predictor = KidneyDiseasePredictor()
    
    import os
    if os.path.exists('c.jpg'):
        result = predictor.predict_single('c.jpg')
        
        if 'error' not in result:
            print("\nPrediction Response:")
            print(json.dumps(result, indent=2))
            
            print("\nResponse Structure:")
            print("""
{
    "predicted_class": 0-3,              # Integer class ID
    "class_name": "Cyst|Normal|Stone|Tumor",  # String class name
    "confidence": 0.0-1.0,              # Confidence score (0-100%)
    "probabilities": {                  # Probability for each class
        "Cyst": 0.0-1.0,
        "Normal": 0.0-1.0,
        "Stone": 0.0-1.0,
        "Tumor": 0.0-1.0
    }
}
            """)
    else:
        print("\n[INFO] Sample image not found")


def request_with_timeout():
    """Example: Make API request with timeout"""
    print("\n" + "=" * 70)
    print("API REQUEST WITH TIMEOUT")
    print("=" * 70)
    
    print("""
import requests
import json

api_url = 'https://kidney-classifier-1.onrender.com/predict'

try:
    files = {'file': open('kidney_scan.jpg', 'rb')}
    
    # Set timeout to 30 seconds
    response = requests.post(api_url, files=files, timeout=30)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Success: {result['class_name']} ({result['confidence']:.1%})")
    else:
        print(f"Error: HTTP {response.status_code}")
        print(response.text)

except requests.exceptions.Timeout:
    print("Request timed out - server may be slow")
except requests.exceptions.ConnectionError:
    print("Connection error - server may be down")
except Exception as e:
    print(f"Error: {e}")
    """)


def batch_api_requests():
    """Example: Make multiple API requests"""
    print("\n" + "=" * 70)
    print("BATCH API REQUESTS")
    print("=" * 70)
    
    print("""
import requests
import os
from pathlib import Path

api_url = 'https://kidney-classifier-1.onrender.com/predict'
image_dir = 'kidney_images'

# Find all images
image_files = list(Path(image_dir).glob('*.jpg')) + list(Path(image_dir).glob('*.png'))

results = []
for img_path in image_files:
    try:
        with open(img_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(api_url, files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            result['filename'] = img_path.name
            results.append(result)
            print(f"✓ {img_path.name}: {result['class_name']}")
        else:
            print(f"✗ {img_path.name}: HTTP {response.status_code}")
    
    except Exception as e:
        print(f"✗ {img_path.name}: {e}")

# Export results as CSV
import csv
with open('predictions.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'prediction', 'confidence'])
    for r in results:
        writer.writerow([
            r['filename'],
            r['class_name'],
            f"{r['confidence']:.4f}"
        ])
    """)


def error_handling():
    """Example: Proper error handling"""
    print("\n" + "=" * 70)
    print("ERROR HANDLING")
    print("=" * 70)
    
    print("""
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

api_url = 'https://kidney-classifier-1.onrender.com/predict'

# Create session with retries
session = requests.Session()
retry = Retry(connect=3, backoff_factor=0.5)
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

try:
    with open('kidney_scan.jpg', 'rb') as f:
        files = {'file': f}
        response = session.post(api_url, files=files, timeout=30)
    
    response.raise_for_status()  # Raise exception for bad status codes
    result = response.json()
    
    # Validate response structure
    required_keys = {'predicted_class', 'class_name', 'confidence', 'probabilities'}
    if not required_keys.issubset(result.keys()):
        print("Invalid response format")
    else:
        print(f"Prediction: {result['class_name']} ({result['confidence']:.1%})")

except requests.exceptions.HTTPError as e:
    print(f"HTTP Error: {e}")
except requests.exceptions.RequestException as e:
    print(f"Request Error: {e}")
except json.JSONDecodeError:
    print("Invalid JSON response")
except Exception as e:
    print(f"Unexpected Error: {e}")
    """)


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("KIDNEY MODEL - API USAGE EXAMPLES")
    print("=" * 70)
    
    example_local_api()
    example_render_api()
    test_prediction_format()
    request_with_timeout()
    batch_api_requests()
    error_handling()
    
    print("\n" + "=" * 70)
    print("END OF API EXAMPLES")
    print("=" * 70)
