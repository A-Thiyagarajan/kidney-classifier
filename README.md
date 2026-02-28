# Kidney Classification Model

Kidney disease classification using VGG16 transfer learning for CT scan images.

## Model Details

- **Architecture**: VGG16 (ImageNet pre-trained) + Custom Dense Layers
- **Classes**: Cyst, Normal, Stone, Tumor
- **Input**: 224×224 RGB images
- **Accuracy**: 99.96% on test set

## Quick Start

### Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Run Flask server
python app.py

# Test with image upload
curl -X POST -F "file=@test_image.jpg" http://localhost:5000/predict
```

### API Endpoints

#### POST /predict
Upload kidney CT scan image for classification

**Request:**
```bash
curl -X POST -F "file=@kidney.jpg" http://localhost:5000/predict
```

**Response:**
```json
{
  "predicted_class": 0,
  "class_name": "Cyst",
  "confidence": 0.9996,
  "probabilities": {
    "Cyst": 0.999600,
    "Normal": 0.000079,
    "Stone": 0.000134,
    "Tumor": 0.000187
  }
}
```

#### GET /
Web UI for uploading and classifying images

#### GET /debug
Debug information (model path, shape, etc.)

#### GET /model-info
Detailed model architecture information

## Deployment to Render

1. **Push to GitHub**
```bash
git add .
git commit -m "Kidney classification model - production ready"
git push
```

2. **Deploy on Render**
   - Connect GitHub repository to Render
   - Select this repository
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
   - Instance Type: Standard or Free

3. **Access the App**
   - Open `https://your-app.onrender.com`
   - Upload kidney CT scan image
   - Get instant classification

## Files

- `app.py` - Flask web application
- `train.py` - Model training script (reference)
- `Kidney.h5` - Trained model (71.7 MB)
- `label.json` - Class label mapping
- `requirements.txt` - Python dependencies
- `Procfile` - Render deployment config
- `render.yaml` - Render service config
- `templates/` - HTML templates
- `c.jpg` - Sample test image

## System Requirements

- Python 3.9+
- TensorFlow 2.12
- 512 MB RAM minimum
- 500 MB disk space

## Development

To retrain the model on custom dataset:

```bash
python train.py
```

Update `train.py` to point to your dataset directory before running.

## Performance

- **Model Load Time**: ~2 seconds
- **Prediction Time**: ~0.2 seconds per image
- **Memory Usage**: ~500 MB
- **Supported Image Sizes**: Any (auto-resized to 224×224)

## License

This model is provided as-is for kidney disease classification tasks.

## Support

For issues or questions about deployment, refer to the Flask app logs at `/debug` endpoint.
