# Healthcare Pathways AI — Personalized Chronic Disease Management

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.x-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.33.0-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.x-green.svg)](https://xgboost.readthedocs.io/)
[![Optuna](https://img.shields.io/badge/Optuna-3.6.x-blueviolet.svg)](https://optuna.org/)

> **Day 1 & Day 2 — Foundational Machine Learning & Ensemble Optimization**  
> 9-day GenAI & Data Science capstone project.  
> Future phases will add Deep Learning, NLP, SLMs, RAG, and Agentic AI layers.

---

## Business Problem

Develop an end-to-end system to create personalized care plans for patients with chronic diseases (diabetes, hypertension, CKD) using their medical history, lifestyle data, and genetic risk proxies.

**Milestones achieved so far:**
- **Day 1:** Baseline patient segmentation (K-Means clustering) and baseline risk prediction (Logistic Regression / RF / GBM).
- **Day 2:** Advanced ensemble modeling (Random Forest, XGBoost, LightGBM), Optuna hyperparameter optimization (5-Fold Stratified CV), SHAP explainability, and an interactive Streamlit Care Coordinator Web Portal.

---

## Project Structure

```
healthcare-pathways-ai/
├── app/
│   ├── __init__.py
│   └── streamlit_app.py        # Interactive Streamlit Web Application
├── data/
│   ├── raw/                    # patients_raw.csv (synthetic, 6,000 patients)
│   └── processed/              # train/test splits, scaled matrices, cluster profiles
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   └── 02_pipeline.ipynb       # End-to-end runnable pipeline
├── src/
│   ├── __init__.py
│   ├── generate_data.py        # Synthetic dataset generator
│   ├── data_prep.py            # Imputation, winsorizing, encoding, splitting
│   ├── features.py             # Derived feature engineering
│   ├── train.py                # Day 1 baseline models & K-Means clustering
│   ├── evaluate.py             # Day 1 evaluation & report generator
│   ├── train_ensembles.py      # Day 2 Optuna hyperparameter tuning (RF, XGB, LGBM)
│   └── evaluate_ensembles.py   # Day 2 PR-AUC/ROC-AUC comparison & SHAP plots
├── models/
│   ├── best_risk_model.joblib  # Saved champion risk prediction model
│   ├── kmeans_model.joblib     # Saved K-Means cluster model
│   ├── preprocessor.joblib     # Fitted ColumnTransformer pipeline
│   ├── tuned_random_forest.joblib
│   ├── tuned_xgboost.joblib
│   └── tuned_lightgbm.joblib
├── reports/
│   ├── model_evaluation_report.md
│   ├── hyperparameter_tuning_report.md
│   ├── model_comparison_report.md
│   ├── roc_comparison_day2.png
│   ├── pr_comparison_day2.png
│   └── shap_summary.png
├── summaries/
│   ├── Day1_Summary.md         # Day 1 executive summary
│   └── Day2_Summary.md         # Day 2 executive summary
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Day 1 & Day 2 Pipelines End-to-End

```bash
# Step 1: Generate synthetic patient dataset (~6,000 rows, 30% high-risk)
python src/generate_data.py

# Step 2: Preprocess, winsorize, engineer derived features, and split 80/20
python src/data_prep.py

# Step 3: Train Day 1 baseline classifiers & K-Means clustering
python src/train.py

# Step 4: Run Day 2 Optuna ensemble tuning (RF, XGBoost, LightGBM)
python src/train_ensembles.py

# Step 5: Evaluate Day 2 ensembles (ROC-AUC, PR-AUC, SHAP) & save champion model
python src/evaluate_ensembles.py
```

### 3. Launch the Interactive Streamlit Web App

```bash
streamlit run app/streamlit_app.py
```

*Access the interactive care coordinator portal at `http://localhost:8501` to evaluate patient risk scores, view K-Means cluster assignments, and inspect top 3 local risk drivers.*

---

## Day 2 Modeling & Tuning Summary

### Hyperparameter Tuning (Optuna — 5-Fold Stratified CV)
- **Class Imbalance Strategy:** Cost-sensitive weighting (`class_weight='balanced'` for RF, `scale_pos_weight` ratio $\approx 2.33$ for XGB/LGBM) to preserve probability calibration.
- **Metric Optimized:** 5-Fold Cross-Validation ROC-AUC.

### Performance Benchmark (Test Set, N=1,200)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | **PR-AUC** |
|:---|---:|---:|---:|---:|---:|---:|
| **Baseline Logistic Regression** | 0.9225 | 0.8217 | 0.9472 | 0.8800 | 0.9814 | 0.9632 |
| **Tuned Random Forest** | 0.9317 | 0.8421 | 0.9528 | 0.8940 | 0.9856 | 0.9710 |
| **Tuned XGBoost** | 0.9408 | 0.8658 | 0.9500 | 0.9059 | 0.9882 | 0.9754 |
| **Tuned LightGBM (Champion)** | **0.9450** | **0.8750** | **0.9528** | **0.9122** | **0.9895** | **0.9782** |

*The tuned ensemble models (especially LightGBM and XGBoost) achieved superior PR-AUC and Recall, minimizing costly False Negatives while reducing False Alarms.*

---

## Reports & Documentation

- [Day 1 Summary](summaries/Day1_Summary.md)
- [Day 2 Summary](summaries/Day2_Summary.md)
- [Hyperparameter Tuning Report](reports/hyperparameter_tuning_report.md)
- [Model Comparison Report](reports/model_comparison_report.md)

---

*Educational prototype demonstration only — not intended for clinical use.*
