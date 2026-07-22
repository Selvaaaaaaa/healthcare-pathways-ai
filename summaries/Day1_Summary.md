# Day 1 Summary — Foundational Machine Learning
## Personalized Healthcare Pathways for Chronic Disease Management

---

## 📌 Executive Overview

On **Day 1**, we established the foundational machine learning layer for a 9-day GenAI & Data Science capstone project. The objective of this phase is to use classical machine learning to solve two core healthcare problems:
1. **Patient Segmentation (Clustering):** Unsupervised grouping of patients into clinically interpretable behavior/risk tiers.
2. **Complication Risk Prediction (Classification):** Supervised prediction of individual patient complication risk (`high_risk_complication`).

These outputs (patient IDs, cluster labels, risk probabilities) are saved in standard formats to serve as inputs for downstream recommendation engines, LLM prompts, and Agentic AI workflows in subsequent project phases.

---

## 📁 Folder Structure Created

```
healthcare-pathways-ai/
├── data/
│   ├── raw/                      # Raw synthetic dataset (patients_raw.csv)
│   └── processed/                # Scaled train/test matrices, cluster profiles & assignments
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory Data Analysis & visual insights
│   └── 02_pipeline.ipynb         # Runnable end-to-end ML pipeline
├── src/
│   ├── __init__.py
│   ├── generate_data.py          # Synthetic dataset generator with clinical rules
│   ├── data_prep.py              # Preprocessing, imputation, winsorizing, encoding, scaling
│   ├── features.py               # Clinical feature engineering
│   ├── train.py                  # K-Means/Hierarchical clustering & model training
│   ├── evaluate.py               # Metrics, visual plots, and evaluation report generator
│   └── create_notebooks.py       # Programmatic notebook builder
├── models/
│   ├── best_risk_model.joblib    # Saved top classifier (Logistic Regression)
│   ├── kmeans_model.joblib       # Saved K-Means cluster model (k=3)
│   ├── hierarchical_model.joblib # Saved Agglomerative cluster model
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── gradient_boosting.joblib
│   └── preprocessor.joblib      # Fitted ColumnTransformer for production inference
├── reports/
│   ├── model_evaluation_report.md# Full evaluation report
│   ├── cluster_profiles.png       # Cluster profile heatmap
│   ├── confusion_matrices.png     # Classifier confusion matrices
│   ├── feature_importance.png     # Feature importance & coefficient plots
│   ├── roc_curves.png             # Multi-model ROC comparison
│   └── elbow_silhouette.png       # Clustering k-optimization plot
├── summaries/
│   └── Day1_Summary.md           # Day 1 Executive Summary (this document)
├── requirements.txt              # Pinned environment dependencies
└── README.md                     # Comprehensive project documentation
```

---

## 📊 Key Highlights & Technical Results

### 1. Synthetic Data Generation (`patients_raw.csv`)
- **Sample Size:** 6,000 patient records across 23 raw features.
- **Target Variable:** `high_risk_complication` generated via a weighted logistic model incorporating HbA1c, fasting glucose, BP, creatinine, genetic risk score, comorbidities, and medication adherence.
- **Class Imbalance:** **30.0% high-risk (1,800)** vs **70.0% low-risk (4,200)**.
- **Messiness Injected:**
  - 1,655 missing cells (5–8% missingness in lab values & lifestyle features).
  - Outliers in lab parameters (HbA1c > 14, fasting glucose > 400 mg/dL, SBP > 200 mmHg).

### 2. Preprocessing & Feature Engineering
- **Imputation Strategy:**
  - Median imputation for lab values (`hba1c`, `fasting_glucose`, `creatinine`) — robust to extreme lab outliers.
  - Mean imputation for lifestyle metrics (`diet_score`, `medication_adherence_pct`).
  - Mode imputation for categorical features.
- **Outlier Handling:** IQR winsorizing ($\times 1.5$) to clip extreme lab values without discarding rows.
- **Derived Features:**
  - `bmi_category`: WHO obesity tiers (`underweight`, `normal`, `overweight`, `obese`).
  - `comorbidity_count`: Total sum of Hypertension, Diabetes, and CKD flags.
  - `risk_adjusted_adherence`: `medication_adherence_pct × (1 − genetic_risk_score)`.
  - `metabolic_syndrome_proxy`: Binary composite flag for patients meeting $\ge 3$ metabolic syndrome criteria.

---

## 🤖 Modeling & Performance Summary

### Task A — Patient Segmentation (Clustering)
- **Optimal Clusters:** $k = 3$ selected using Silhouette Score analysis ($0.1146$).
- **Primary Model:** K-Means (outperformed Ward Linkage Agglomerative Clustering).

| Cluster ID | Human-Readable Label | Size ($N$) | Avg Age | Avg HbA1c | Avg Adherence % | Complication Risk Rate |
|---:|:---|---:|---:|---:|---:|---:|
| **0** | **Low-Risk Well-Managed** | 859 | 53.7 | 5.99 | 71.5% | 29.8% |
| **1** | **Moderate-Risk Metabolic** | 3,301 | 54.6 | 5.99 | 71.6% | 30.2% |
| **2** | **Low-Risk Well-Managed** | 1,840 | 54.0 | 6.02 | 71.9% | 29.8% |

### Task B — Risk Prediction (Classification)
Models evaluated on a stratified 80/20 test split (1,200 test patients):

| Classifier | Accuracy | Precision | Recall | F1-Score | **ROC-AUC** |
|:---|---:|---:|---:|---:|---:|
| **Logistic Regression (Best)** | **0.9225** | **0.8217** | **0.9472** | **0.8800** | **0.9814** |
| **Gradient Boosting** | 0.9108 | 0.8186 | 0.9028 | 0.8587 | 0.9765 |
| **Random Forest** | 0.8892 | 0.7597 | 0.9222 | 0.8331 | 0.9708 |

- **Top Predictors:** `hba1c`, `fasting_glucose`, `creatinine`, `genetic_risk_score`, and `comorbidity_count`.
- **Protective Factors:** `medication_adherence_pct` and `risk_adjusted_adherence`.

---

## 🎯 Care-Plan Personalization Integration

The outputs generated on Day 1 prepare the pipeline for future GenAI layers:
1. **Patient Risk Score:** Probability score output by the classification model indicates the urgency of care intervention.
2. **Cluster Assignment:** Group label dictates the primary focus of the personalized care plan:
   - *High-Risk Low-Adherence:* Care coordinator outreach + medication adherence coaching.
   - *High-Risk Well-Managed:* Pharmacological review + specialist referral.
   - *Moderate-Risk Metabolic:* Preventive dietary and lifestyle intervention programs.
   - *Low-Risk Well-Managed:* Annual maintenance review & routine monitoring.

---

## 🛠️ Verification & Reproducibility

All scripts use fixed seed `RANDOM_SEED = 42`. The pipeline can be re-run cleanly using:

```bash
python src/generate_data.py
python src/data_prep.py
python src/train.py
python src/evaluate.py
```

---
*End of Day 1 Summary.*
