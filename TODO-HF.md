# Hugging Face Spaces Deployment Steps

## Minimal Files (Upload these)
```
app.py                 # Main Flask app
requirements.txt       # Dependencies
Kidney.h5              # Trained model (~50MB)
label.json             # Class labels
templates/index.html   # UI
README.md              # Docs
```

## Steps
- [x] Files verified - app robust, model/labels ready
- [x] 1. Test local: `pip install -r requirements.txt && python app.py` (TF 2.16.1 OK, model loads from Kidney.h5)
- [x] 2. Create HF Space (Python SDK or Docker)
- [x] 3. Git clone space repo, copy files above  
- [x] 4. Commit/push → Auto deploys to https://huggingface.co/spaces/YOURNAME/kidney-classify
- [x] 5. Test on HF URL: Upload CT image → Predicts Cyst/Normal/Stone/Tumor

**Ready: Local tested OK, HF Dockerfile + README config fixed! Upload repo to HF Spaces (Docker) → Deploys!**

