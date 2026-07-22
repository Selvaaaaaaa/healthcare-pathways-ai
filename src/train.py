"""
train.py
========
Model Training Script for Healthcare Pathways AI (Day 1).

This script orchestrates two parallel ML tasks:

TASK A — Patient Segmentation (Clustering)
  - Loads scaled feature matrix (X_train + X_test combined for unsupervised task)
  - K-Means: elbow method (k=2–10) + silhouette score analysis to select optimal k
  - Agglomerative Hierarchical Clustering (Ward linkage): alternative comparison
  - Cluster profiling: mean age, HbA1c, adherence, risk rate per cluster
  - Human-readable cluster labels based on risk + adherence profiles
  - Saves: models/kmeans_model.joblib, data/processed/cluster_assignments.csv

TASK B — Risk Prediction (Classification)
  - Trains three models on (X_train, y_train):
      1. Logistic Regression (L2, class_weight='balanced')
      2. Random Forest (n=200, class_weight='balanced')
      3. Gradient Boosting (GradientBoostingClassifier)
  - Class imbalance handled via class_weight='balanced' (LR, RF)
    Note on SMOTE: SMOTE is an alternative; class_weight is preferred here
    because it avoids inflating the dataset with synthetic minority samples
    that could introduce subtle distribution shift in later pipeline phases.
  - Data leakage check: target never included in feature matrix; patient_id
    dropped before any modelling step; preprocessing fitted on train set only.
  - Saves: models/best_risk_model.joblib + models/all_classifiers.joblib

Usage:
    python src/train.py

Outputs:
    models/kmeans_model.joblib
    models/hierarchical_model.joblib
    models/logistic_regression.joblib
    models/random_forest.joblib
    models/gradient_boosting.joblib
    models/best_risk_model.joblib
    data/processed/cluster_assignments.csv
    data/processed/cluster_profiles.csv
    data/processed/elbow_data.csv
"""

import os
import sys
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROC_DIR = os.path.join(ROOT, "data", "processed")
MODELS_DIR = os.path.join(ROOT, "models")
REPORTS_DIR = os.path.join(ROOT, "reports")

RANDOM_SEED = 42

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ── Cluster labels ─────────────────────────────────────────────────────────────
# Assigned after profiling — thresholds based on cluster means

def assign_cluster_label(row: pd.Series) -> str:
    """
    Assign a human-readable clinical label to a cluster profile row.

    Logic:
      - "High-Risk Low-Adherence"   : high risk rate AND low adherence
      - "High-Risk High-Adherence"  : high risk rate but good adherence (genetic/comorbid)
      - "Moderate-Risk Metabolic"   : moderate risk with metabolic syndrome indicators
      - "Low-Risk Well-Managed"     : low risk rate with good lifestyle metrics
      - "Young Low-Risk"            : young + low risk (default catch-all for low-risk)
    """
    risk = row.get("risk_rate", 0)
    adherence = row.get("avg_adherence", 100)
    age = row.get("avg_age", 50)

    if risk > 0.50 and adherence < 70:
        return "High-Risk Low-Adherence"
    elif risk > 0.50 and adherence >= 70:
        return "High-Risk Well-Managed"
    elif risk > 0.30:
        return "Moderate-Risk Metabolic"
    elif age < 45:
        return "Young Low-Risk"
    else:
        return "Low-Risk Well-Managed"


# ═════════════════════════════════════════════════════════════════════════════
# TASK A — Patient Segmentation
# ═════════════════════════════════════════════════════════════════════════════

def run_clustering(X_all: np.ndarray, df_imputed: pd.DataFrame, feature_names: list[str]) -> int:
    """
    K-Means + Hierarchical clustering with elbow / silhouette analysis.

    Parameters
    ----------
    X_all : np.ndarray
        Full scaled feature matrix (train + test combined for unsupervised task).
    df_imputed : pd.DataFrame
        Imputed (unscaled) patient dataframe for cluster profiling.
    feature_names : list[str]
        Ordered feature names matching X_all columns.

    Returns
    -------
    int
        Optimal k chosen by silhouette score.
    """
    print("\n" + "=" * 65)
    print("TASK A — Patient Segmentation (Clustering)")
    print("=" * 65)

    # ── Elbow + Silhouette analysis (k = 2 … 10) ──────────────────────────────
    k_range = range(2, 11)
    inertias, sil_scores = [], []

    print("\n  Running K-Means for k = 2 … 10 ...")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(X_all)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_all, labels, sample_size=2000, random_state=RANDOM_SEED)
        sil_scores.append(sil)
        print(f"    k={k:2d}  inertia={km.inertia_:10.0f}  silhouette={sil:.4f}")

    # Save elbow data for notebook visualisation
    elbow_df = pd.DataFrame({"k": list(k_range), "inertia": inertias, "silhouette": sil_scores})
    elbow_df.to_csv(os.path.join(PROC_DIR, "elbow_data.csv"), index=False)

    # Select optimal k: highest silhouette score
    best_idx = int(np.argmax(sil_scores))
    optimal_k = list(k_range)[best_idx]
    print(f"\n  [OK] Optimal k = {optimal_k} (silhouette = {sil_scores[best_idx]:.4f})")

    # ── Fit final K-Means ─────────────────────────────────────────────────────
    km_final = KMeans(n_clusters=optimal_k, random_state=RANDOM_SEED, n_init=20)
    km_labels = km_final.fit_predict(X_all)
    joblib.dump(km_final, os.path.join(MODELS_DIR, "kmeans_model.joblib"))
    print(f"  Saved: models/kmeans_model.joblib")

    # ── Agglomerative Hierarchical Clustering (alternative) ───────────────────
    print(f"\n  Running Agglomerative Clustering (Ward, k={optimal_k})...")
    agg = AgglomerativeClustering(n_clusters=optimal_k, linkage="ward")
    # Use a subsample for memory efficiency with large n
    subsample = min(3000, len(X_all))
    idx_sub = np.random.RandomState(RANDOM_SEED).choice(len(X_all), subsample, replace=False)
    agg.fit(X_all[idx_sub])
    agg_labels_sub = agg.labels_
    agg_sil = silhouette_score(X_all[idx_sub], agg_labels_sub, random_state=RANDOM_SEED)
    print(f"  Agglomerative silhouette (subsample={subsample}): {agg_sil:.4f}")

    # Full agglomerative assignment
    agg_full = AgglomerativeClustering(n_clusters=optimal_k, linkage="ward")
    agg_full_labels = agg_full.fit_predict(X_all)
    joblib.dump(agg_full, os.path.join(MODELS_DIR, "hierarchical_model.joblib"))
    print(f"  Saved: models/hierarchical_model.joblib")
    agg_sil_full = silhouette_score(X_all, agg_full_labels,
                                    sample_size=2000, random_state=RANDOM_SEED)
    print(f"  K-Means sil={sil_scores[best_idx]:.4f} | Agglomerative sil={agg_sil_full:.4f}")
    print(f"  -> {'K-Means' if sil_scores[best_idx] >= agg_sil_full else 'Agglomerative'} "
          f"selected as primary segmentation model.")

    # ── Cluster profiling ─────────────────────────────────────────────────────
    print(f"\n  Profiling {optimal_k} clusters...")

    # Map scaled assignments back to imputed dataframe (same row order)
    profile_df = df_imputed.copy()
    profile_df["cluster_kmeans"] = km_labels
    profile_df["cluster_hierarchical"] = agg_full_labels

    profile_cols = {
        "avg_age": ("age", "mean"),
        "avg_bmi": ("bmi", "mean"),
        "avg_hba1c": ("hba1c", "mean"),
        "avg_fasting_glucose": ("fasting_glucose", "mean"),
        "avg_systolic_bp": ("systolic_bp", "mean"),
        "avg_adherence": ("medication_adherence_pct", "mean"),
        "avg_genetic_risk": ("genetic_risk_score", "mean"),
        "risk_rate": ("high_risk_complication", "mean"),
        "n_patients": ("age", "count"),
    }

    agg_dict = {new_col: pd.NamedAgg(column=src, aggfunc=fn)
                for new_col, (src, fn) in profile_cols.items()}

    cluster_profile = (
        profile_df.groupby("cluster_kmeans")
        .agg(**agg_dict)
        .round(3)
        .reset_index()
    )

    # Assign human-readable labels
    cluster_profile["cluster_label"] = cluster_profile.apply(assign_cluster_label, axis=1)

    print("\n  Cluster Profile Summary:")
    print(cluster_profile[["cluster_kmeans", "cluster_label",
                            "avg_age", "avg_hba1c", "avg_adherence",
                            "risk_rate", "n_patients"]].to_string(index=False))

    cluster_profile.to_csv(os.path.join(PROC_DIR, "cluster_profiles.csv"), index=False)
    print(f"\n  Saved: data/processed/cluster_profiles.csv")

    # ── Save cluster assignments per patient ──────────────────────────────────
    assignments = profile_df[["patient_id", "cluster_kmeans", "cluster_hierarchical"]].copy()
    assignments = assignments.merge(
        cluster_profile[["cluster_kmeans", "cluster_label"]], on="cluster_kmeans", how="left"
    )
    assignments.to_csv(os.path.join(PROC_DIR, "cluster_assignments.csv"), index=False)
    print(f"  Saved: data/processed/cluster_assignments.csv")

    return optimal_k


# ═════════════════════════════════════════════════════════════════════════════
# TASK B — Risk Prediction (Classification)
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_classifier(name: str, model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Compute and return full classification metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob) if y_prob is not None else None,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return metrics


def run_classification(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
) -> dict:
    """
    Train, evaluate, and compare three classifiers.

    Class imbalance strategy:
      - Logistic Regression & Random Forest: class_weight='balanced'
        (scales the loss function inversely proportional to class frequencies)
      - Gradient Boosting: sample_weight applied during fit (GBM doesn't have
        class_weight param in sklearn's GradientBoostingClassifier)
      - Note: SMOTE (Synthetic Minority Oversampling) is an alternative but is
        NOT applied here to keep the Day 1 pipeline simple and avoid
        distribution shift between train and test folds.

    Data leakage check:
      - Target (high_risk_complication) was dropped from feature matrix in data_prep.py
      - patient_id was dropped before any modelling
      - Preprocessor was fitted ONLY on X_train; transform applied separately to X_test
      - No future information (cluster labels) is used as input to the classifier

    Returns
    -------
    dict
        Results dict with metrics for all three models.
    """
    print("\n" + "=" * 65)
    print("TASK B — Risk Prediction (Classification)")
    print("=" * 65)

    # ── Class weight for Gradient Boosting (manual sample weights) ─────────────
    classes, counts = np.unique(y_train, return_counts=True)
    class_weight_dict = {c: len(y_train) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    sample_weights = np.array([class_weight_dict[y] for y in y_train])

    # ── Define models ─────────────────────────────────────────────────────────
    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_SEED,
            solver="lbfgs",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            max_depth=10,
            min_samples_leaf=5,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=RANDOM_SEED,
        ),
    }

    all_results = {}

    for name, model in models.items():
        print(f"\n  Training {name}...")
        if name == "Gradient Boosting":
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)

        results = evaluate_classifier(name, model, X_test, y_test)
        all_results[name] = results

        print(f"    Accuracy : {results['accuracy']:.4f}")
        print(f"    Precision: {results['precision']:.4f}")
        print(f"    Recall   : {results['recall']:.4f}")
        print(f"    F1       : {results['f1']:.4f}")
        print(f"    ROC-AUC  : {results['roc_auc']:.4f}")

        # Save individual model
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(model, os.path.join(MODELS_DIR, f"{safe_name}.joblib"))
        print(f"    Saved: models/{safe_name}.joblib")

    # ── Select best model (by ROC-AUC) ───────────────────────────────────────
    best_name = max(all_results, key=lambda k: all_results[k]["roc_auc"])
    best_model = {
        "Logistic Regression": models["Logistic Regression"],
        "Random Forest": models["Random Forest"],
        "Gradient Boosting": models["Gradient Boosting"],
    }[best_name]

    print(f"\n  [OK] Best model: {best_name} (ROC-AUC = {all_results[best_name]['roc_auc']:.4f})")
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_risk_model.joblib"))
    print(f"  Saved: models/best_risk_model.joblib")

    # ── Feature importance / coefficients ─────────────────────────────────────
    print("\n  Extracting feature importance...")
    _save_feature_importance(models, feature_names, best_name)

    # Save results summary
    summary = []
    for name, res in all_results.items():
        row = {k: v for k, v in res.items() if k != "confusion_matrix"}
        summary.append(row)
    pd.DataFrame(summary).round(4).to_csv(
        os.path.join(PROC_DIR, "model_comparison.csv"), index=False
    )
    print(f"  Saved: data/processed/model_comparison.csv")

    # Save confusion matrices
    cm_data = {name: res["confusion_matrix"] for name, res in all_results.items()}
    with open(os.path.join(PROC_DIR, "confusion_matrices.json"), "w") as f:
        json.dump(cm_data, f, indent=2)

    return all_results


def _save_feature_importance(models: dict, feature_names: list[str], best_name: str) -> None:
    """Save feature importance / coefficient plots."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Feature Importance / Model Coefficients", fontsize=14, fontweight="bold")

    # ── Random Forest importance ───────────────────────────────────────────────
    rf = models["Random Forest"]
    rf_imp = pd.Series(rf.feature_importances_, index=feature_names).nlargest(20)
    axes[0].barh(rf_imp.index[::-1], rf_imp.values[::-1], color="#2196F3", alpha=0.8)
    axes[0].set_title("Random Forest — Top 20 Feature Importances")
    axes[0].set_xlabel("Importance")
    axes[0].tick_params(axis="y", labelsize=8)

    # ── Logistic Regression coefficients ──────────────────────────────────────
    lr = models["Logistic Regression"]
    lr_coef = pd.Series(lr.coef_[0], index=feature_names)
    top_lr = pd.concat([lr_coef.nlargest(10), lr_coef.nsmallest(10)])
    colors = ["#F44336" if c > 0 else "#4CAF50" for c in top_lr.values]
    axes[1].barh(top_lr.index[::-1], top_lr.values[::-1], color=colors[::-1], alpha=0.8)
    axes[1].axvline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_title("Logistic Regression — Top Coefficients (Red=Risk↑, Green=Protective)")
    axes[1].set_xlabel("Coefficient Value")
    axes[1].tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: reports/feature_importance.png")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("Healthcare Pathways AI — Model Training")
    print("=" * 65)

    # ── Load processed data ───────────────────────────────────────────────────
    print("\nLoading processed data...")
    X_train = pd.read_csv(os.path.join(PROC_DIR, "X_train.csv")).values
    X_test = pd.read_csv(os.path.join(PROC_DIR, "X_test.csv")).values
    y_train = pd.read_csv(os.path.join(PROC_DIR, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(PROC_DIR, "y_test.csv")).values.ravel()
    df_imputed = pd.read_csv(os.path.join(PROC_DIR, "patients_imputed.csv"))

    with open(os.path.join(PROC_DIR, "feature_names.txt")) as f:
        feature_names = [line.strip() for line in f.readlines()]

    print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")
    print(f"  Features: {len(feature_names)}")

    # ── Combine train + test for clustering (unsupervised — no label leakage) ──
    X_all = np.vstack([X_train, X_test])

    # ── Run tasks ─────────────────────────────────────────────────────────────
    optimal_k = run_clustering(X_all, df_imputed, feature_names)
    all_results = run_classification(X_train, X_test, y_train, y_test, feature_names)

    print("\n" + "=" * 65)
    print("Training complete. Summary:")
    print("=" * 65)
    print(f"  Optimal clusters (K-Means): {optimal_k}")
    print("\n  Classification Results:")
    for name, res in all_results.items():
        print(f"    {name:25s} | AUC={res['roc_auc']:.3f}  F1={res['f1']:.3f}")

    best = max(all_results, key=lambda k: all_results[k]["roc_auc"])
    print(f"\n  Best model: {best}")
    print("\nAll artefacts saved. Run evaluate.py for full evaluation report.")


if __name__ == "__main__":
    sys.path.insert(0, ROOT)
    main()
