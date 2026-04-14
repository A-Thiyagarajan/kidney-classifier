---
title: Kidney Disease Classifier
emoji: 🩸
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

# Kidney Disease CT Classifier

VGG16 model classifies kidney CT scans: **Cyst**, **Normal**, **Stone**, **Tumor**.

## HF Spaces Demo
Upload CT image → Instant prediction + confidence.

**Tech:**
- TensorFlow 2.19 + Keras 3
- Flask app.py served via Dockerfile (port 7860)
- Kidney.h5 (trained on CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone)

## 🚀 Dual Deployment: HF Backend + GitHub Pages Frontend

### Backend (Hugging Face Spaces)
```
app.py          # Flask API /predict (POST, file upload)
requirements.txt
Kidney.h5       # Model
label.json      # Labels: Cyst/Normal/Stone/Tumor
Dockerfile      # (recommended for TF deps)
```
**Steps:**
1. Push to GitHub repo `A-Thiyagarajan/Kidney-Classifier`
2. Create HF Space → Import from GitHub
3. SDK: **Docker** (for TF 2.19)
4. Auto-deploys → API: `https://thiyagu2004-kidney-classifier.hf.space/predict`

### Frontend (GitHub Pages)
```
frontend/index.html  # Simple upload UI → calls HF API
```
**Steps:**
1. GitHub repo Settings → Pages
2. Source: Deploy from **main branch**, folder `/frontend`
3. Live: `https://a-thiyagarajan.github.io/Kidney-Classifier/`

### 🔄 Workflow
```
Single GitHub repo ─push→ HF Spaces (backend API)
                   └─→ GitHub Pages (frontend UI)
                              ↓
                    Frontend calls HF /predict
```

**Update:** `git add . && git commit -m "updates" && git push origin main`

**Local Test:**
- Backend: `pip install -r requirements.txt && python app.py`
- Frontend: Open `frontend/index.html` (edit HF URL if needed)

[![Duplicate Space](https://huggingface.co/spaces/Thiyagu2004/Kidney-Classifier/badge)](https://huggingface.co/spaces/Thiyagu2004/Kidney-Classifier)

