# Day 2 Summary — Ensemble Methods & Hyperparameter Tuning
## Personalized Healthcare Pathways for Chronic Disease Management

---

## 📌 Executive Overview

On **Day 2**, we advanced the risk-prediction pipeline beyond Day 1 baselines by implementing state-of-the-art ensemble modeling (Random Forest, XGBoost, LightGBM), systematic hyperparameter optimization (Optuna with 5-Fold Stratified Cross-Validation), SHAP explainability, and an interactive Streamlit Care Coordinator Portal.

---

## 📁 Artifacts & Files Created

```
healthcare-pathways-ai/
├── app/
│   ├── __init__.py
│   └── streamlit_app.py           # Interactive Streamlit Web Portal
├── src/
│   ├── train_ensembles.py         # Optuna 5-Fold CV hyperparameter tuning script
│   └── evaluate_ensembles.py      # PR-AUC / ROC-AUC benchmark & SHAP explainability
├── models/
│   ├── tuned_random_forest.joblib
│   ├── tuned_xgboost.joblib
│   ├── tuned_lightgbm.joblib
│   └── best_risk_model.joblib     # Saved champion model (Tuned LightGBM)
├── reports/
│   ├── hyperparameter_tuning_report.md
│   ├── model_comparison_report.md
│   ├── roc_comparison_day2.png
│   ├── pr_comparison_day2.png
│   └── shap_summary.png
└── summaries/
    └── Day2_Summary.md            # Day 2 Summary (this document)
```

---

## 📊 Key Highlights & Technical Accomplishments

### 1. Class Imbalance Strategy
- **Mechanism:** Cost-sensitive loss weighting (`class_weight='balanced'` for RF, `scale_pos_weight = N_neg / N_pos ≈ 2.33` for XGBoost & LightGBM).
- **Rationale:** Adjusting loss gradients directly penalizes minority class errors without generating synthetic samples (like SMOTE). This maintains raw feature distributions and preserves risk probability calibration for care coordinator decisions.

### 2. Systematic Hyperparameter Tuning (Optuna)
- **Engine:** Optuna TPE Sampler with 5-Fold Stratified K-Fold CV (30 trials per algorithm).
- **Tuned Hyperparameters:**
  - *Random Forest:* `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`.
  - *XGBoost:* `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`.
  - *LightGBM:* `n_estimators`, `max_depth`, `num_leaves`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_samples`.

---

## 🤖 Model Comparison Benchmark (Test Set, N=1,200)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | **PR-AUC** |
|:---|---:|---:|---:|---:|---:|---:|
| **Baseline Logistic Regression (Day 1)** | 0.9225 | 0.8217 | 0.9472 | 0.8800 | 0.9814 | 0.9632 |
| **Tuned Random Forest** | 0.9317 | 0.8421 | 0.9528 | 0.8940 | 0.9856 | 0.9710 |
| **Tuned XGBoost** | 0.9408 | 0.8658 | 0.9500 | 0.9059 | 0.9882 | 0.9754 |
| **Tuned LightGBM (Champion)** | **0.9450** | **0.8750** | **0.9528** | **0.9122** | **0.9895** | **0.9782** |

### Clinical Selection Rationale:
- **Tuned LightGBM** selected as champion:
  - Highest **PR-AUC (0.9782)** — critical under class imbalance.
  - High **Recall (0.9528)** — minimizes costly False Negatives (unflagged high-risk patients).
  - High **Precision (0.8750)** — reduces false alarms for care coordinators.

---

## 🩺 Interactive Streamlit Web Portal (`app/streamlit_app.py`)

Key features built into the care coordinator app:
1. **Interactive Clinical Input Form:** Sidebar for Demographics, Comorbidities, Lab Vitals, and Lifestyle Scores.
2. **Real-time Risk Prediction:** Predicts complication probability and flags `HIGH RISK` or `LOW RISK`.
3. **Real-time Cohort Segmentation:** Maps input patient to Day 1 K-Means cluster (e.g. *Moderate-Risk Metabolic*, *Low-Risk Well-Managed*).
4. **Local Feature Drivers (Top 3):** Computes patient-specific top 3 risk factors driving the prediction.
5. **Care Plan Action Recommendations:** Suggests adherence coaching, endocrinology consults, or renal safeguards based on findings.
6. **Educational Banner:** Prominent prototype disclaimer banner.

---

## 🛠️ Verification Commands

```bash
# 1. Run Optuna ensemble tuning
python src/train_ensembles.py

# 2. Benchmark models & save champion
python src/evaluate_ensembles.py

# 3. Launch Streamlit portal
streamlit run app/streamlit_app.py
```

---
*End of Day 2 Summary.*
