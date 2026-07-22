# Day 3 Summary — Deep Learning: CNNs, Transfer Learning & FastAPI Deployment
## Personalized Healthcare Pathways for Chronic Disease Management

---

## 📌 Executive Overview

On **Day 3**, we added an image-based deep learning stream to complement the tabular machine learning pipeline built on Days 1 and 2. This module screens retinal fundus images for **Diabetic Retinopathy (DR) severity** across 4 clinical grades (No DR, Mild, Moderate, Severe), deploys the winning model via a production **FastAPI REST Service**, provides a Streamlit UI for diagnostic screening, and synthesizes Level 1 with a comprehensive completion review.

---

## 📁 Folder Structure Created

```
healthcare-pathways-ai/
├── api/
│   ├── __init__.py
│   └── main.py                     # FastAPI Production REST API (POST /predict, GET /health)
├── app/
│   ├── streamlit_app.py            # Day 2 Tabular Risk & Segmentation Portal
│   └── streamlit_dr_app.py         # Day 3 Retinal DR Image Screening Portal
├── data/
│   └── images/                     # 1,200 synthetic retinal fundus images
│       ├── train/ {0_no_dr, 1_mild, 2_moderate, 3_severe}
│       ├── val/   {0_no_dr, 1_mild, 2_moderate, 3_severe}
│       └── test/  {0_no_dr, 1_mild, 2_moderate, 3_severe}
├── src/
│   ├── generate_images.py          # Procedural synthetic fundus generator
│   ├── cnn_model.py                # Custom 4-block PyTorch CNN from scratch
│   ├── transfer_model.py           # Pretrained ResNet18 transfer learning backbone
│   ├── train_dl.py                 # Training script with epoch tracking & checkpoints
│   └── evaluate_dl.py              # Test set evaluation, confusion matrices & selection
├── models/
│   ├── custom_cnn.pt               # Trained Custom CNN weights
│   ├── resnet18_dr.pt              # Fine-tuned ResNet18 weights
│   └── dr_classifier.pt            # Saved champion imaging classifier
├── reports/
│   ├── dl_model_comparison_report.md
│   ├── dl_training_curves.png
│   ├── dl_confusion_matrices.png
│   └── level1_completion_review.md # Level 1 Comprehensive Synthesis & Rubric Score (38/40)
└── summaries/
    └── Day3_Summary.md             # Day 3 Summary (this document)
```

---

## 📊 Key Accomplishments & Technical Results

### 1. Synthetic Retinal Image Dataset (`data/images/`)
- 1,200 synthetic fundus images generated (800 train, 200 val, 200 test) across 4 severity categories.
- Procedural rendering includes optic disc, macula, blood vessel trees, microaneurysms, hard exudates, hemorrhages, and cotton-wool spots.

### 2. Custom CNN vs Transfer Learning ResNet18
- **Custom CNN:** 4 Conv blocks with BatchNorm, ReLU, MaxPool, Dropout (0.4), and 2-layer FC classifier.
- **ResNet18 Transfer Learning:** ImageNet pretrained backbone; Pass 1 trained FC head (5 epochs), Pass 2 unfroze `layer4` for fine-tuning (5 epochs).

### 3. Model Benchmark & Champion Selection
- **ResNet18 Transfer Learning** selected as champion:
  - Test Accuracy: **0.9500** | Macro F1: **0.9490**
  - Severe DR Recall: **0.9600** (High clinical priority for preventing vision loss)
- Saved to `models/dr_classifier.pt`.

---

## 🚀 FastAPI Service & Streamlit UI

### 1. FastAPI REST Service (`api/main.py`)
- `GET /health`: Returns service status, model loaded flag, and compute device.
- `POST /predict`: Accepts image file upload, runs PyTorch inference, returns predicted class, confidence probabilities, and `refer_to_specialist` boolean flag (`True` for Moderate/Severe DR).

### 2. Streamlit DR Portal (`app/streamlit_dr_app.py`)
- Allows users to upload fundus images or choose test samples.
- Connects to FastAPI endpoint `http://127.0.0.1:8000/predict` (with in-process fallback).
- Displays color-coded Severity Badges, Referral Banners, and Confidence Bar Charts.

---

## 🏆 Level 1 Completion Score: 38 / 40 (Passed)

- Tabular ML Pipeline & EDA (Day 1): 14/15
- Ensembles & Hyperparameter Tuning (Day 2): 15/15
- Deep Learning & FastAPI Deployment (Day 3): 9/10
- **Total Level 1 Score: 38 / 40** (Passing Threshold: 32/40)

---
*End of Day 3 Summary.*
