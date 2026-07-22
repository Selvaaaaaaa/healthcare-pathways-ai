# Healthcare Pathways AI — Personalized Chronic Disease Management

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.x-orange.svg)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.x-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.33.0-FF4B4B.svg)](https://streamlit.io/)

> **Level 1 Complete — Foundational ML, Ensemble Optimization & Deep Learning**  
> 9-day GenAI & Data Science capstone project.  
> Future phases will add Sequence/NLP modeling, SLMs, RAG, and Agentic AI orchestrators.

---

## Business Problem & Multi-Modal Vision

Develop an integrated platform to generate personalized care plans for patients with chronic diseases (diabetes, hypertension, CKD) by combining:
1. **Systemic Clinical Data (Tabular):** Medical history, vitals, lab values, lifestyle, and genetic proxies.
2. **Organ-Specific Biomarkers (Imaging):** Retinal fundus scans for Diabetic Retinopathy (DR) severity screening.

---

## Level 1 Architecture Overview

```
                                +-----------------------------------+
                                |     HEALTHCARE PATHWAYS AI        |
                                |       LEVEL 1 STACK COMPLETE      |
                                +-----------------+-----------------+
                                                  |
                 +--------------------------------+--------------------------------+
                 |                                                                 |
                 v                                                                 v
     [TABULAR STREAM (Days 1-2)]                                       [IMAGING STREAM (Day 3)]
     - 6,000 Patient Records (EHR)                                     - 1,200 Retinal Fundus Images (128x128)
     - Imputation, Winsorizing, Feature Eng                            - Custom CNN vs ResNet18 Transfer Model
     - K-Means Patient Clustering (k=3)                                - Severe DR Recall: 0.96 | Acc: 0.95
     - Tuned LightGBM / LR Classifier (AUC 0.98)                       - FastAPI REST API (api/main.py)
     - Streamlit App (app/streamlit_app.py)                            - Streamlit DR App (app/streamlit_dr_app.py)
```

---

## Project Structure

```
healthcare-pathways-ai/
├── api/
│   ├── __init__.py
│   └── main.py                     # FastAPI Service (POST /predict, GET /health)
├── app/
│   ├── __init__.py
│   ├── streamlit_app.py            # Tabular Risk & Segmentation Portal
│   └── streamlit_dr_app.py         # Retinal DR Imaging Portal
├── data/
│   ├── images/                     # 1,200 synthetic retinal fundus images
│   ├── raw/                        # patients_raw.csv (6,000 synthetic patient records)
│   └── processed/                  # Scaled matrices, cluster profiles, training logs
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory Data Analysis
│   └── 02_pipeline.ipynb           # Tabular ML runnable pipeline
├── src/
│   ├── __init__.py
│   ├── generate_data.py            # Synthetic clinical data generator
│   ├── data_prep.py                # Preprocessing pipeline
│   ├── features.py                 # Feature engineering module
│   ├── train.py                    # Baseline ML & K-Means clustering
│   ├── train_ensembles.py          # Optuna ensemble hyperparameter tuning
│   ├── evaluate_ensembles.py       # Tabular model comparison & SHAP plots
│   ├── generate_images.py          # Procedural synthetic fundus generator
│   ├── cnn_model.py                # PyTorch Custom CNN architecture
│   ├── transfer_model.py           # PyTorch ResNet18 Transfer Learning model
│   ├── train_dl.py                 # Deep learning training pipeline
│   └── evaluate_dl.py              # Deep learning evaluation & benchmark
├── models/
│   ├── best_risk_model.joblib      # Champion tabular risk model
│   ├── kmeans_model.joblib         # Saved K-Means cluster model
│   ├── preprocessor.joblib         # Fitted ColumnTransformer
│   ├── custom_cnn.pt               # Trained Custom CNN weights
│   ├── resnet18_dr.pt              # Trained ResNet18 weights
│   └── dr_classifier.pt            # Saved champion imaging classifier
├── reports/
│   ├── model_evaluation_report.md  # Day 1 report
│   ├── hyperparameter_tuning_report.md # Day 2 tuning report
│   ├── model_comparison_report.md  # Day 2 benchmark report
│   ├── dl_model_comparison_report.md   # Day 3 DL benchmark report
│   └── level1_completion_review.md # Level 1 Comprehensive Synthesis & Rubric Review
├── summaries/
│   ├── Day1_Summary.md
│   ├── Day2_Summary.md
│   └── Day3_Summary.md
├── requirements.txt
└── README.md
```

---

## How to Run Level 1 (Days 1 – 3)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Tabular Pipeline Execution (Days 1 & 2)

```bash
# Generate synthetic clinical data (6,000 rows)
python src/generate_data.py

# Preprocess & feature engineer
python src/data_prep.py

# Train baseline models & K-Means clustering
python src/train.py

# Run Optuna ensemble tuning (RF, XGBoost, LightGBM)
python src/train_ensembles.py

# Evaluate ensembles & SHAP explainability
python src/evaluate_ensembles.py

# Launch Tabular Care Coordinator App
python -m streamlit run app/streamlit_app.py
```

### 3. Imaging Pipeline Execution (Day 3)

```bash
# Step 1: Generate synthetic retinal fundus images (1,200 images)
python src/generate_images.py

# Step 2: Train Custom CNN & ResNet18 Transfer Learning models
python src/train_dl.py

# Step 3: Evaluate Deep Learning models & save champion dr_classifier.pt
python src/evaluate_dl.py

# Step 4: Launch FastAPI Production Service
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# Step 5: Launch Streamlit Retinal DR Screening App (in a separate terminal)
python -m streamlit run app/streamlit_dr_app.py
```

---

## Benchmark Metrics Summary

### Tabular Risk Model (Days 1–2, $N=1,200$)
- **Champion Model:** Logistic Regression / LightGBM
- **ROC-AUC:** **0.9814** | **PR-AUC:** **0.9578** | **Recall:** **0.9472**

### Imaging DR Model (Day 3, $N=200$)
- **Champion Model:** ResNet18 Transfer Learning
- **Test Accuracy:** **0.9500** | **Macro F1:** **0.9490** | **Severe DR Recall:** **0.9600**

---

## Documentation & Level 1 Review

- [Day 1 Summary](summaries/Day1_Summary.md)
- [Day 2 Summary](summaries/Day2_Summary.md)
- [Day 3 Summary](summaries/Day3_Summary.md)
- [Level 1 Completion Review & Rubric Score (38/40)](reports/level1_completion_review.md)

---

*Educational prototype demonstration only — not intended for direct clinical diagnostic use.*
