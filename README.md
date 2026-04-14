# Kidney Disease Classifier

[![Hugging Face Spaces](https://huggingface.co/spaces/Thiyagu2004/Kidney-Classifier/badge)](https://huggingface.co/spaces/Thiyagu2004/Kidney-Classifier)

VGG16-based transfer learning model for classifying kidney CT images:
- **Cyst**
- **Normal** 
- **Stone**
- **Tumor**

## HF Spaces
- Upload CT image → Get prediction + confidence
- Lazy TF 2.13 + NumPy 1.24 (HF compatible)

## Local Run
```bash
pip install -r requirements.txt
python app.py
```

Model: Kidney.h5 (224x224 RGB /255.0 norm)
Labels: label.json
