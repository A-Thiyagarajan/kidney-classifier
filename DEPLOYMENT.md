# Render Deployment Checklist

## Pre-Deployment Steps

- [x] Model converted to Kidney.h5 (HDF5 format)
- [x] All weights properly loaded and verified
- [x] label.json with correct class mapping
- [x] Flask app (app.py) configured for production
- [x] requirements.txt with correct versions
- [x] Procfile configured for gunicorn
- [x] render.yaml configured for Render deployment
- [x] HTML templates in templates/ folder
- [x] Sample test image (c.jpg) included
- [x] .gitignore configured to exclude venv
- [x] README.md with documentation

## Project Structure

```
Kidney-Classify/
├── app.py                  # Flask web application
├── train.py               # Model training script (reference)
├── Kidney.h5              # Trained model (71.7 MB)
├── label.json             # Class labels (4 classes)
├── requirements.txt       # Python dependencies
├── Procfile               # Heroku/Render config
├── render.yaml            # Render service config
├── .gitignore             # Git exclusions
├── README.md              # Project documentation
├── c.jpg                  # Sample test image
└── templates/
    └── index.html         # Web UI
```

## Deployment Steps

### 1. Prepare Git Repository
```bash
cd Kidney-Classify
git init
git add .
git commit -m "Kidney classification model - production ready"
```

### 2. Push to GitHub
```bash
git remote add origin https://github.com/YOUR_USERNAME/Kidney-Classify.git
git branch -M main
git push -u origin main
```

### 3. Deploy on Render
1. Go to https://dashboard.render.com
2. Click "New +" → "Web Service"
3. Select "Connect Repository"
4. Choose "Kidney-Classify" repository
5. Configure:
   - **Name**: kidney-classify (or your preferred name)
   - **Branch**: main
   - **Runtime**: Python 3.10.13
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Instance Type**: Free (for testing) or Standard (for production)
6. Click "Create Web Service"

### 4. Monitor Deployment
- Check Render dashboard for build progress
- View logs in "Events" tab
- Wait for "Service is live" message

### 5. Test the Deployment
```bash
# Test web UI
https://kidney-classify.onrender.com/

# Test API
curl -X POST -F "file=@c.jpg" https://kidney-classify.onrender.com/predict

# Test debug endpoint
https://kidney-classify.onrender.com/debug
```

## Expected Responses

### Successful Prediction
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

### Debug Endpoint
```json
{
  "model_path": "/opt/render/project/src/Kidney.h5",
  "model_loaded": true,
  "model_summary": "...",
  "labels": {"0": "Cyst", "1": "Normal", "2": "Stone", "3": "Tumor"}
}
```

## File Sizes (For Reference)

- Kidney.h5: 71.7 MB
- app.py: ~6 KB
- requirements.txt: <1 KB
- Total: ~71.7 MB (within Render limits)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Build fails with "TensorFlow not found" | Check requirements.txt versions, try TensorFlow 2.13 |
| Model fails to load | Verify Kidney.h5 is in repository root |
| Predictions are incorrect | Run production_verification.py locally to debug |
| Service crashes on startup | Check app.py logs, verify model path |
| Slow predictions | Normal for first prediction (model loads), subsequent calls are fast |

## Performance on Render

- **Startup time**: 30-60 seconds (model loading)
- **First prediction**: 2-3 seconds (model initialization)
- **Subsequent predictions**: 0.2-0.5 seconds
- **Memory usage**: ~500 MB
- **Concurrent requests**: Handled by gunicorn workers

## After Deployment

1. Share the live URL
2. Test with various kidney CT scan images
3. Monitor Render dashboard for errors
4. Keep GitHub repo updated with improvements

## Success! 🎉

Your kidney classification model is now live on Render!
Access it at: `https://kidney-classify.onrender.com/`
