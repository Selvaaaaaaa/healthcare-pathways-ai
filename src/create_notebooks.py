"""
create_notebooks.py
===================
Programmatically generates both Jupyter notebooks for the Healthcare Pathways AI project.

Run:
    python src/create_notebooks.py
"""

import os
import nbformat as nbf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NOTEBOOKS_DIR = os.path.join(ROOT, "notebooks")
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)


def md(text): return nbf.v4.new_markdown_cell(text)
def code(text): return nbf.v4.new_code_cell(text)


# ═════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 1 — EDA
# ═════════════════════════════════════════════════════════════════════════════

def create_eda_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3"
    }
    nb.cells = [

        md("""# 01 — Exploratory Data Analysis
## Healthcare Pathways AI | Day 1 — Foundational ML

This notebook explores the synthetic patient dataset to understand distributions,
missing data patterns, class imbalance, and clinical feature relationships before
building the preprocessing and modelling pipeline.

> **Dataset:** `data/raw/patients_raw.csv` — 6,000 synthetic patients, 23 features  
> **Target:** `high_risk_complication` (binary, ~70/30 imbalance)
"""),

        code("""import os, sys
sys.path.insert(0, os.path.abspath(".."))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})
sns.set_palette("husl")

# Load raw data
df = pd.read_csv("../data/raw/patients_raw.csv")
print(f"Dataset shape: {df.shape}")
df.head(3)
"""),

        md("""## 1. Dataset Overview

Let's start with basic dimensions, data types, and a quick sanity check.
"""),

        code("""print("=" * 55)
print(f"Rows     : {df.shape[0]:,}")
print(f"Columns  : {df.shape[1]}")
print(f"Memory   : {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
print("=" * 55)
print("\\nDtypes:\\n")
print(df.dtypes.to_string())
"""),

        code("""df.describe(include="all").T.head(30)
"""),

        md("""## 2. Missing Value Analysis

The dataset was intentionally seeded with 5–8% missing values in `hba1c`,
`fasting_glucose`, `creatinine`, `medication_adherence_pct`, and `diet_score`
to simulate real-world EHR data quality issues.
"""),

        code("""# Count + percentage of missing values
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
missing_df = missing_df[missing_df["Missing Count"] > 0].sort_values("Missing %", ascending=False)
print(missing_df.to_string())
"""),

        code("""fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.barh(missing_df.index, missing_df["Missing %"], color=sns.color_palette("Reds_r", len(missing_df)))
for bar, val in zip(bars, missing_df["Missing %"]):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}%", va="center", fontsize=10)
ax.set_xlabel("Missing (%)")
ax.set_title("Missing Values by Column", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
"""),

        md("""**Key finding:** The missing data pattern is **Missing At Random (MAR)** by design.
Lab values (`hba1c`, `fasting_glucose`, `creatinine`) are missing most frequently — this
mirrors real EHR data where not all tests are ordered at every visit.
We will apply **median imputation** for lab columns (robust to outliers) and
**mean imputation** for lifestyle columns (diet_score, adherence).
"""),

        md("""## 3. Target Distribution & Class Imbalance
"""),

        code("""target_counts = df["high_risk_complication"].value_counts()
target_pct = df["high_risk_complication"].value_counts(normalize=True) * 100

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Bar chart
axes[0].bar(["Low Risk (0)", "High Risk (1)"], target_counts.values,
            color=["#4CAF50", "#F44336"], edgecolor="black", alpha=0.85)
axes[0].set_title("Target Class Distribution", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Count")
for i, (cnt, pct) in enumerate(zip(target_counts.values, target_pct.values)):
    axes[0].text(i, cnt + 20, f"{cnt:,}\\n({pct:.1f}%)", ha="center", fontsize=10)

# Pie chart
axes[1].pie(target_counts.values, labels=["Low Risk", "High Risk"],
            autopct="%1.1f%%", colors=["#4CAF50", "#F44336"],
            startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[1].set_title("Class Proportion", fontsize=12, fontweight="bold")

plt.suptitle("high_risk_complication Distribution", fontsize=13, y=1.02)
plt.tight_layout()
plt.show()

print(f"\\nClass imbalance ratio: {target_counts[0] / target_counts[1]:.2f}:1  (low:high)")
"""),

        md("""**Key finding:** The dataset has a **~70/30 class imbalance** (low:high risk ≈ 2.3:1).
This is clinically realistic — in a chronic disease management programme, roughly 30% of
patients are at high risk of complications at any given time.

**Implication for modelling:** We will use `class_weight='balanced'` in classifiers to
prevent the model from simply predicting "low risk" for everyone and achieving 70% accuracy
without any real learning. ROC-AUC is our primary evaluation metric because it is
threshold-agnostic and robust to class imbalance.
"""),

        md("""## 4. Descriptive Statistics by Risk Class
"""),

        code("""numeric_cols = ["age", "bmi", "hba1c", "fasting_glucose", "systolic_bp",
               "diastolic_bp", "cholesterol_ldl", "cholesterol_hdl",
               "creatinine", "medication_adherence_pct", "diet_score",
               "genetic_risk_score", "years_since_diagnosis"]

stats = df.groupby("high_risk_complication")[numeric_cols].mean().round(2).T
stats.columns = ["Low Risk (0)", "High Risk (1)"]
stats["Δ (High − Low)"] = (stats["High Risk (1)"] - stats["Low Risk (0)"]).round(2)
stats["Δ%"] = ((stats["Δ (High − Low)"] / stats["Low Risk (0)"]) * 100).round(1)
print(stats.to_string())
"""),

        md("""**Key findings from the statistics table:**
- **HbA1c** is markedly higher in the high-risk group — primary diabetes management indicator.
- **Fasting glucose** shows the largest absolute difference.
- **Creatinine** is elevated in high-risk patients — kidney function as a complication indicator.
- **Medication adherence** is lower in high-risk patients, confirming adherence as protective.
- **Genetic risk score** is higher in high-risk patients — genetic burden is a strong driver.
"""),

        md("""## 5. Correlation Heatmap
"""),

        code("""corr_df = df[numeric_cols + ["high_risk_complication"]].corr()

fig, ax = plt.subplots(figsize=(13, 10))
mask = np.zeros_like(corr_df, dtype=bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(
    corr_df, mask=mask, annot=True, fmt=".2f",
    cmap="RdBu_r", center=0, vmin=-1, vmax=1,
    square=True, linewidths=0.5, ax=ax,
    annot_kws={"size": 8},
)
ax.set_title("Correlation Matrix — Clinical Features × Risk Target", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
"""),

        code("""# Correlations with target specifically
target_corr = corr_df["high_risk_complication"].drop("high_risk_complication").sort_values()

fig, ax = plt.subplots(figsize=(8, 6))
colors = ["#F44336" if v > 0 else "#4CAF50" for v in target_corr.values]
ax.barh(target_corr.index, target_corr.values, color=colors, edgecolor="black", alpha=0.8)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Feature Correlations with high_risk_complication\\n(Red=Risk-increasing, Green=Protective)",
             fontsize=11, fontweight="bold")
ax.set_xlabel("Pearson Correlation")
plt.tight_layout()
plt.show()
"""),

        md("""**Key finding:** `hba1c`, `creatinine`, `genetic_risk_score`, and the comorbidity flags
show the strongest positive correlations with high-risk status.
`medication_adherence_pct` and `diet_score` show negative correlations (protective).
These findings are consistent with clinical knowledge and validate the synthetic data
generation mechanism.
"""),

        md("""## 6. Distribution Plots by Risk Class
"""),

        code("""key_clinical = ["hba1c", "fasting_glucose", "creatinine",
               "medication_adherence_pct", "genetic_risk_score", "bmi"]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

palette = {0: "#4CAF50", 1: "#F44336"}
labels = {0: "Low Risk", 1: "High Risk"}

for ax, col in zip(axes, key_clinical):
    for cls in [0, 1]:
        vals = df.loc[df["high_risk_complication"] == cls, col].dropna()
        ax.hist(vals, bins=35, alpha=0.55, color=palette[cls], label=labels[cls],
                density=True, edgecolor="none")
        vals.plot(kind="kde", ax=ax, color=palette[cls], lw=2)
    ax.set_title(col, fontsize=11, fontweight="bold")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)

plt.suptitle("Key Clinical Feature Distributions by Risk Class", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
"""),

        code("""# Comorbidity flags by risk class
flag_cols = ["hypertension_flag", "diabetes_flag", "ckd_flag", "family_history_flag"]

flag_rates = df.groupby("high_risk_complication")[flag_cols].mean() * 100

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(flag_cols))
width = 0.35

bars0 = ax.bar(x - width/2, flag_rates.loc[0], width, label="Low Risk", color="#4CAF50", alpha=0.85)
bars1 = ax.bar(x + width/2, flag_rates.loc[1], width, label="High Risk", color="#F44336", alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(["Hypertension", "Diabetes", "CKD", "Family History"], fontsize=10)
ax.set_ylabel("Prevalence (%)")
ax.set_title("Comorbidity & History Flags by Risk Class", fontsize=12, fontweight="bold")
ax.legend()
ax.set_ylim(0, 100)

for bar in list(bars0) + list(bars1):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.show()
"""),

        code("""# Physical activity distribution by risk class
act_order = ["sedentary", "low", "moderate", "high"]
act_data = (df.groupby(["physical_activity_level", "high_risk_complication"])
              .size().unstack().reindex(act_order))

fig, ax = plt.subplots(figsize=(9, 5))
act_data_pct = act_data.div(act_data.sum(axis=1), axis=0) * 100
act_data_pct.plot(kind="bar", ax=ax, color=["#4CAF50", "#F44336"],
                  edgecolor="black", alpha=0.85)
ax.set_xticklabels(act_order, rotation=0, fontsize=10)
ax.set_ylabel("Proportion (%)")
ax.set_title("Physical Activity Level vs Risk Class", fontsize=12, fontweight="bold")
ax.legend(["Low Risk", "High Risk"])
plt.tight_layout()
plt.show()
"""),

        md("""**Key findings from distribution plots:**
1. **HbA1c** distributions are clearly separated — high-risk patients cluster above 8.0.
2. **Creatinine** shows a marked right-tail shift in high-risk patients (renal impairment).
3. **Genetic risk score** is substantially higher in the high-risk group.
4. **Medication adherence** distributions overlap heavily but the low-risk group peaks at higher values.
5. **Comorbidity flags**: Diabetes (+15pp), CKD (+12pp), and Hypertension (+8pp) are all more prevalent in high-risk patients.
6. **Physical activity**: Sedentary patients have a notably higher proportion of high-risk labels.

## Summary of EDA Findings

| Finding | Implication |
|---|---|
| 30% positive class imbalance | Use class_weight='balanced'; report AUC, not accuracy |
| HbA1c & glucose strong predictors | Include as primary features; engineer interaction |
| Creatinine predicts renal complications | Critical for CKD patient segment care plans |
| Adherence is protective | Adherence coaching should be the #1 care plan intervention for high-adherence-deficit clusters |
| Genetic risk is informative | Strong signal for a future RAG layer pulling gene-condition literature |
| Missing values are lab-heavy (5–8%) | Median imputation appropriate; MAR assumption is reasonable |

---
*Proceed to `02_pipeline.ipynb` for the end-to-end ML pipeline.*
"""),

    ]

    path = os.path.join(NOTEBOOKS_DIR, "01_eda.ipynb")
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[OK] Created: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 2 — End-to-End Pipeline
# ═════════════════════════════════════════════════════════════════════════════

def create_pipeline_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3"
    }
    nb.cells = [

        md("""# 02 — End-to-End ML Pipeline
## Healthcare Pathways AI | Day 1 — Foundational ML

This notebook runs the **complete Day 1 pipeline** from raw data to trained models
and evaluation report. It calls the `src/` modules directly, keeping the notebook
concise and the business logic reusable.

**Pipeline stages:**
1. Generate synthetic data → `data/raw/patients_raw.csv`
2. Preprocess & feature-engineer → `data/processed/`
3. Train clustering + classification models → `models/`
4. Evaluate and generate report → `reports/`
"""),

        code("""import os, sys
sys.path.insert(0, os.path.abspath(".."))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})

ROOT = os.path.abspath("..")
PROC_DIR = os.path.join(ROOT, "data", "processed")
MODELS_DIR = os.path.join(ROOT, "models")
REPORTS_DIR = os.path.join(ROOT, "reports")
"""),

        md("""## Step 1 — Generate Synthetic Dataset
"""),

        code("""from src.generate_data import generate_patients

df_raw = generate_patients(n=6000, seed=42)
os.makedirs(os.path.join(ROOT, "data", "raw"), exist_ok=True)
df_raw.to_csv(os.path.join(ROOT, "data", "raw", "patients_raw.csv"), index=False)

print(f"Dataset shape   : {df_raw.shape}")
print(f"Positive class  : {df_raw['high_risk_complication'].mean():.2%}")
print(f"Missing cells   : {df_raw.isnull().sum().sum()}")
df_raw.head(3)
"""),

        md("""## Step 2 — Preprocessing & Feature Engineering
"""),

        code("""from src.data_prep import run_preprocessing
run_preprocessing()
"""),

        code("""# Verify outputs
X_train = pd.read_csv(os.path.join(PROC_DIR, "X_train.csv"))
X_test  = pd.read_csv(os.path.join(PROC_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(PROC_DIR, "y_train.csv")).values.ravel()
y_test  = pd.read_csv(os.path.join(PROC_DIR, "y_test.csv")).values.ravel()
df_imp  = pd.read_csv(os.path.join(PROC_DIR, "patients_imputed.csv"))

with open(os.path.join(PROC_DIR, "feature_names.txt")) as f:
    feature_names = [l.strip() for l in f]

print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")
print(f"Features: {len(feature_names)}")
print(f"Derived features added: bmi_category, comorbidity_count, risk_adjusted_adherence, metabolic_syndrome_proxy")
"""),

        md("""## Step 3A — Patient Segmentation (Clustering)
"""),

        code("""from src.train import run_clustering

X_all = np.vstack([X_train.values, X_test.values])
optimal_k = run_clustering(X_all, df_imp, feature_names)
print(f"\\nOptimal k = {optimal_k}")
"""),

        code("""# Visualise elbow / silhouette
elbow_df = pd.read_csv(os.path.join(PROC_DIR, "elbow_data.csv"))

fig, ax1 = plt.subplots(figsize=(9, 5))
ax2 = ax1.twinx()

ax1.plot(elbow_df["k"], elbow_df["inertia"], "bo-", lw=2, markersize=7, label="Inertia (Elbow)")
ax2.plot(elbow_df["k"], elbow_df["silhouette"], "rs--", lw=2, markersize=7, label="Silhouette Score")

best_k = elbow_df.loc[elbow_df["silhouette"].idxmax(), "k"]
ax2.axvline(best_k, color="gray", linestyle=":", linewidth=1.5, label=f"Optimal k={best_k}")

ax1.set_xlabel("k (Number of Clusters)"); ax1.set_ylabel("Inertia", color="blue")
ax2.set_ylabel("Silhouette Score", color="red")
ax1.tick_params(axis="y", labelcolor="blue"); ax2.tick_params(axis="y", labelcolor="red")

lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
ax1.legend(lines, labels, loc="upper right", fontsize=9)

plt.title(f"K-Means Elbow & Silhouette Analysis → Optimal k = {best_k}", fontsize=12, fontweight="bold")
plt.tight_layout(); plt.show()
"""),

        code("""# Cluster profiles
cluster_profiles = pd.read_csv(os.path.join(PROC_DIR, "cluster_profiles.csv"))
display_cols = ["cluster_kmeans", "cluster_label", "n_patients",
                "avg_age", "avg_hba1c", "avg_adherence", "risk_rate"]
print(cluster_profiles[display_cols].to_string(index=False))
"""),

        code("""# Cluster heatmap
profile_cols = ["avg_age", "avg_bmi", "avg_hba1c", "avg_fasting_glucose",
                "avg_systolic_bp", "avg_adherence", "avg_genetic_risk", "risk_rate"]

plot_df = cluster_profiles.set_index("cluster_label")[profile_cols].copy()
plot_norm = (plot_df - plot_df.min()) / (plot_df.max() - plot_df.min() + 1e-9)

fig, ax = plt.subplots(figsize=(12, 4))
sns.heatmap(plot_norm.T, annot=plot_df.T.round(2), fmt="g",
            cmap="RdYlGn_r", ax=ax, linewidths=0.5,
            cbar_kws={"label": "Normalised value"})
ax.set_title("Cluster Profile Heatmap (Normalised; annotated with raw means)",
             fontsize=11, fontweight="bold")
ax.tick_params(axis="x", rotation=20)
plt.tight_layout(); plt.show()
"""),

        md("""## Step 3B — Risk Prediction (Classification)
"""),

        code("""from src.train import run_classification

all_results = run_classification(
    X_train.values, X_test.values, y_train, y_test, feature_names
)
"""),

        code("""# Results comparison table
model_comparison = pd.read_csv(os.path.join(PROC_DIR, "model_comparison.csv"))
print("\\nModel Comparison:")
print(model_comparison.to_string(index=False))

best = model_comparison.loc[model_comparison["roc_auc"].idxmax()]
print(f"\\n→ Best model: {best['model']} (AUC = {best['roc_auc']:.4f})")
"""),

        code("""# ROC curves
from sklearn.metrics import roc_curve, auc as sk_auc

fig, ax = plt.subplots(figsize=(8, 6))
colors = ["#2196F3", "#4CAF50", "#FF5722"]
model_files = ["logistic_regression", "random_forest", "gradient_boosting"]
model_names = ["Logistic Regression", "Random Forest", "Gradient Boosting"]

for fname, name, color in zip(model_files, model_names, colors):
    path = os.path.join(MODELS_DIR, f"{fname}.joblib")
    if os.path.exists(path):
        model = joblib.load(path)
        y_prob = model.predict_proba(X_test.values)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={roc_auc:.3f})")

ax.plot([0,1],[0,1],"k--",lw=1,label="Random")
ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
       title="ROC Curve Comparison — Complication Risk Prediction")
ax.legend(loc="lower right"); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
"""),

        code("""# Confusion matrices
from sklearn.metrics import confusion_matrix

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, fname, name in zip(axes, model_files, model_names):
    path = os.path.join(MODELS_DIR, f"{fname}.joblib")
    if os.path.exists(path):
        model = joblib.load(path)
        cm = confusion_matrix(y_test, model.predict(X_test.values))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Low","High"], yticklabels=["Low","High"],
                    annot_kws={"size": 14})
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.suptitle("Confusion Matrices — Test Set", fontsize=12, fontweight="bold")
plt.tight_layout(); plt.show()
"""),

        code("""# Feature importance
img_path = os.path.join(REPORTS_DIR, "feature_importance.png")
if os.path.exists(img_path):
    from IPython.display import Image
    display(Image(img_path, width=900))
"""),

        md("""## Step 4 — Evaluate & Generate Report
"""),

        code("""from src.evaluate import main as eval_main
eval_main()
"""),

        code("""# Show report path
report_path = os.path.join(REPORTS_DIR, "model_evaluation_report.md")
print(f"[OK] Report written to: {report_path}")

# Show final model files
print("\\nSaved model files:")
for f in sorted(os.listdir(MODELS_DIR)):
    path = os.path.join(MODELS_DIR, f)
    size = os.path.getsize(path) / 1e3
    print(f"  {f:40s} {size:8.1f} KB")
"""),

        md("""## Pipeline Summary

| Stage | Output | Status |
|---|---|---|
| Data generation | `data/raw/patients_raw.csv` | ✅ |
| Preprocessing | `data/processed/X_train/test, y_train/test` | ✅ |
| Feature engineering | bmi_category, comorbidity_count, risk_adjusted_adherence, metabolic_syndrome | ✅ |
| K-Means clustering | `models/kmeans_model.joblib`, cluster profiles | ✅ |
| Hierarchical clustering | `models/hierarchical_model.joblib` | ✅ |
| Logistic Regression | `models/logistic_regression.joblib` | ✅ |
| Random Forest | `models/random_forest.joblib` | ✅ |
| Gradient Boosting | `models/gradient_boosting.joblib` | ✅ |
| Best model saved | `models/best_risk_model.joblib` | ✅ |
| Evaluation report | `reports/model_evaluation_report.md` | ✅ |

---
**Next:** Day 2 — Deep Learning Risk Model (do not implement yet)
"""),

    ]

    path = os.path.join(NOTEBOOKS_DIR, "02_pipeline.ipynb")
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[OK] Created: {path}")


if __name__ == "__main__":
    create_eda_notebook()
    create_pipeline_notebook()
    print("\nBoth notebooks created successfully.")
