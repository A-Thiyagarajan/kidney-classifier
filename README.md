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

## Deployment Instructions
1. Create HF Space: https://huggingface.co/new-space → Name: Kidney-Classifier
2. **SDK: Docker** (required!)
3. Git clone Space repo locally
4. Copy these files: `app.py`, `requirements.txt`, `Dockerfile`, `Kidney.h5`, `label.json`, `templates/index.html`
5. `git add . && git commit -m "Fix TF deps for HF" && git push`
6. HF auto-builds → Live at https://huggingface.co/spaces/YOURNAME/Kidney-Classifier

**Local Test:** `docker build -t kidney-app . && docker run -p 7860:7860 kidney-app`

[![Duplicate Space](https://huggingface.co/spaces/Thiyagu2004/Kidney-Classifier/badge)](https://huggingface.co/spaces/Thiyagu2004/Kidney-Classifier)

