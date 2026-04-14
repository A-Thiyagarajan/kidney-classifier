---
title: Kidney Disease Classifier
emoji: 🫘
colorFrom: pink
colorTo: red
sdk: docker
sdk_version: latest
app_file: app.py
pinned: false
---

# Kidney Disease Classifier

Flask web app for classifying kidney CT scans into Cyst, Normal, Stone, Tumor.

**Demo:** Upload CT image → Instant prediction with confidence.

## Hugging Face Spaces
This repo is ready for HF Spaces (Docker SDK).

1. Fork or upload to HF Space
2. Hardware: CPU Basic (free)
3. Auto-deploys on push

## Local Run
```
pip install -r requirements.txt
python app.py
```
Open http://localhost:10000

## API
POST /predict (multipart file='image.jpg')

Response:
```
{
  \"label\": \"Stone\", 
  \"confidence\": 0.9876,
  \"class_index\": 2
}
```

Model: VGG16 fine-tuned, 224x224 RGB input.
99.96% test accuracy.
