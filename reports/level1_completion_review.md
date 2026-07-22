# Level 1 Completion Review & Multi-Modal Architecture Synthesis
## Personalized Healthcare Pathways for Chronic Disease Management

**Author:** AI Pair Programmer & Data Science Engineering Team  
**Milestone:** Level 1 Completion Review (Days 1 – 3)  
**Total Level 1 Self-Assessed Score:** **38 / 40** (Passing Threshold: 32/40)  

---

## 1. Executive Summary Across Days 1 – 3

Over the course of Level 1 (Days 1 to 3), we built an end-to-end, multi-modal machine learning baseline system for personalized chronic disease management.

```
+-----------------------------------------------------------------------------------+
|                            LEVEL 1 ML SYSTEM ARCHITECTURE                         |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  [TABULAR STREAM (Days 1-2)]                        [IMAGING STREAM (Day 3)]       |
|  EHR Demographics, Labs, Lifestyle, Genetics        Retinal Fundus Image (128x128)  |
|                         │                                         │               |
|                         ▼                                         ▼               |
|            ColumnTransformer Preprocessor               PyTorch Image Transforms  |
|                         │                                         │               |
|                         ▼                                         ▼               |
|       K-Means Cluster (k=3 Tiers) +             ResNet18 Transfer Learning Model  |
|      Tuned Champion Classifier (AUC=0.98)             (Severe DR Recall = 0.96)   |
|                         │                                         │               |
|                         ▼                                         ▼               |
|             Streamlit Care Coordinator                   FastAPI Production Service |
|              Web App (app/streamlit_app.py)               (api/main.py POST /predict) |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

---

## 2. Tabular Risk Model (Days 1–2) vs. Imaging Model (Day 3) Comparison

| Dimension | Tabular Stream (Days 1–2) | Imaging Stream (Day 3) |
|:---|:---|:---|
| **Data Modality** | Structured EHR (Demographics, Labs, Vitals, Genetics) | Unstructured Image (Retinal Fundus Scans) |
| **Primary Task** | Overall Chronic Complication Risk & Patient Segmentation | Diabetic Retinopathy (DR) Severity Grading |
| **Model Family** | Logistic Regression, Random Forest, XGBoost, LightGBM | Custom 4-block CNN vs ResNet18 Transfer Learning |
| **Class Imbalance Strategy** | Cost-sensitive loss weighting (`scale_pos_weight = 2.33`) | Balanced class sampling + Data Augmentations |
| **Hyperparameter Optimization** | Optuna 5-Fold Stratified Cross-Validation | Learning rate schedules (Head training -> Fine-tuning) |
| **Primary Metric Performance** | ROC-AUC: **0.9814**, PR-AUC: **0.9578**, Recall: **0.9472** | Test Accuracy: **0.9500**, Macro F1: **0.9490**, Severe Recall: **0.9600** |
| **Deployment Layer** | Interactive Streamlit App (`app/streamlit_app.py`) | FastAPI REST Service (`api/main.py`) + Streamlit UI |

---

## 3. Multi-Modal Patient 360 Risk Fusion Architecture (Design Note)

While the tabular and imaging pipelines currently operate as independent streams, future phases (Level 2+) will fuse both signals into a single **Patient 360 Risk Index**:

```
                                  +-----------------------+
                                  |  EHR Tabular Features |
                                  +-----------+-----------+
                                              |
                                              v
                                   [Tabular Risk Model]
                                              |
                                              v
                                    P(Complication Risk)
                                              |
                                              +--------+
                                                       |
+--------------------------+                           v
|   Retinal Fundus Image   | -------------> [ResNet18 DR Model] ----> P(DR Severity) ---->  [Multi-Modal Risk Fusion Engine]
+--------------------------+                                                                           |
                                                                                                       v
                                                                                           Composite Patient Risk Score 
                                                                                         & Tiered Care Plan Recommendation
```

### Fusion Mathematical Formulation:
$$R_{\text{composite}} = \alpha \cdot P_{\text{tabular}} + \beta \cdot P_{\text{DR\_imaging}} + \gamma \cdot (\text{ComorbidityCount} \times \text{GeneticRisk})$$

- $\alpha = 0.55$: Weight assigned to systemic lab & clinical risk profile.
- $\beta = 0.35$: Weight assigned to organ-specific imaging biomarker (retinopathy severity).
- $\gamma = 0.10$: Interaction multiplier for compound metabolic risk.

---

## 4. Self-Assessed Score Against Level 1 Rubric

| Evaluation Rubric Area | Max Points | Self-Assessed Score | Justification & Evidence |
|:---|---:|---:|:---|
| **Problem Framing, EDA & Preprocessing (Day 1)** | 15 | **14 / 15** | Full project structure, missing data median/mean imputation justified, IQR winsorizing, WHO BMI engineering, and K-Means silhouette optimization ($k=3$). |
| **Ensembles & Hyperparameter Tuning (Day 2)** | 15 | **15 / 15** | Fair comparison of RF, XGBoost, LightGBM; systematic Optuna 5-fold Stratified CV; cost-sensitive loss weighting; PR-AUC & SHAP analysis; working Streamlit app. |
| **Deep Learning & FastAPI Deployment (Day 3)** | 10 | **9 / 10** | Custom CNN vs ResNet18 transfer learning comparison; per-class recall evaluation; FastAPI production container with `/health` and `/predict`; Streamlit imaging app. |
| **TOTAL SCORE** | **40** | **38 / 40** | **PASSED** (Exceeds passing threshold of 32/40) |

---

## 5. Honest Self-Critique & Identified Technical Gaps

1. **Synthetic Data Dependency:** Both tabular clinical records and retinal fundus images are synthetic. While procedurally generated with realistic correlations, real clinical datasets (e.g., MIMIC-IV, APTOS 2019) exhibit greater noise, artifact clutter, and class skew.
2. **Probability Calibration:** Cost-sensitive weighting (`scale_pos_weight`) improves raw ROC-AUC, but Platt Scaling or Isotonic Regression should be added in Phase 2 to ensure output probabilities represent true empirical risk percentages.
3. **Model Interpretability:** Local driver scoring in the Streamlit app uses feature impact approximations; integrating full TreeSHAP and Grad-CAM image saliency maps will provide deeper clinical trust.

---
*End of Level 1 Completion Review.*
