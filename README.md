---
title: Kidney Disease Classifier
emoji: 🩸
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Kidney Disease CT Classifier

VGG16 model classifies kidney CT scans: **Cyst**, **Normal**, **Stone**, **Tumor**.

## HF Spaces Demo
Upload CT image → Instant prediction + confidence.

**Tech:**
- TensorFlow 2.13 + Keras 2.13 (NumPy 1.24 compatible)
- Flask app.py (lazy model load → no startup crash)
- Kidney.h5 (trained on CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone)

[![Duplicate Space](https://huggingface.co/spaces/Thiyagu2004/Kidney-Classifier/badge)](https://huggingface.co/spaces/Thiyagu2004/Kidney-Classifier)
