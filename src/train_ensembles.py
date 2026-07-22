"""
train_ensembles.py
==================
Day 2: Ensemble Modeling & Systematic Hyperparameter Tuning using Optuna.

Models trained and tuned:
  1. Random Forest (RandomForestClassifier)
  2. XGBoost (XGBClassifier)
  3. LightGBM (LGBMClassifier)

Class Imbalance Handling Strategy:
  - Imbalance Ratio: ~70% negative (low risk), ~30% positive (high risk)
  - Random Forest: `class_weight='balanced'`
  - XGBoost & LightGBM: `scale_pos_weight = n_neg / n_pos` (~2.33)
  - Justification: Cost-sensitive loss weighting directly adjusts gradients/penalties
    during training without injecting synthetic samples (as SMOTE does), preserving
    natural feature distributions and probability calibration for downstream risk scoring.

Cross-Validation & Optimization:
  - 5-Fold Stratified K-Fold CV (preserves 70/30 class balance in every fold)
  - Optimization Engine: Optuna (30 trials per model, optimizing ROC-AUC)
  - Fixed Random Seed: 42

Outputs:
  models/tuned_random_forest.joblib
  models/tuned_xgboost.joblib
  models/tuned_lightgbm.joblib
  reports/hyperparameter_tuning_report.md
  data/processed/tuning_summary.csv
"""

import os
import sys
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import optuna

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score

import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Path configuration ─────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROC_DIR = os.path.join(ROOT, "data", "processed")
MODELS_DIR = os.path.join(ROOT, "models")
REPORTS_DIR = os.path.join(ROOT, "reports")

RANDOM_SEED = 42
N_TRIALS = 30
N_FOLDS = 5

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ── Load Processed Data ────────────────────────────────────────────────────────

def load_data():
    """Load Day 1 preprocessed dataset."""
    X_train = pd.read_csv(os.path.join(PROC_DIR, "X_train.csv")).values
    X_test = pd.read_csv(os.path.join(PROC_DIR, "X_test.csv")).values
    y_train = pd.read_csv(os.path.join(PROC_DIR, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(PROC_DIR, "y_test.csv")).values.ravel()

    with open(os.path.join(PROC_DIR, "feature_names.txt")) as f:
        feature_names = [line.strip() for line in f.readlines()]

    return X_train, X_test, y_train, y_test, feature_names


# ── Optuna Objective Functions ─────────────────────────────────────────────────

def objective_rf(trial, X, y, cv):
    """Optuna search space for Random Forest."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
        "max_depth": trial.suggest_int("max_depth", 5, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    }
    clf = RandomForestClassifier(**params)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return float(scores.mean())


def objective_xgb(trial, X, y, cv, scale_pos_weight):
    """Optuna search space for XGBoost."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        "scale_pos_weight": scale_pos_weight,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "eval_metric": "logloss",
    }
    clf = xgb.XGBClassifier(**params)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return float(scores.mean())


def objective_lgb(trial, X, y, cv, scale_pos_weight):
    """Optuna search space for LightGBM."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
        "scale_pos_weight": scale_pos_weight,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbosity": -1,
    }
    clf = lgb.LGBMClassifier(**params)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return float(scores.mean())


# ── Main Training & Tuning Function ──────────────────────────────────────────

def train_and_tune_ensembles():
    print("=" * 65)
    print("Healthcare Pathways AI — Day 2: Ensemble Tuning")
    print("=" * 65)

    X_train, X_test, y_train, y_test, feature_names = load_data()
    print(f"\n[1/4] Loaded train shape: {X_train.shape} | test shape: {X_test.shape}")
    print(f"      Feature count: {len(feature_names)}")

    # Class ratio calculation
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos
    print(f"      Class balance: {n_pos} pos ({n_pos/len(y_train):.1%}) | {n_neg} neg ({n_neg/len(y_train):.1%})")
    print(f"      Calculated scale_pos_weight: {scale_pos_weight:.3f}")

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    tuning_results = []

    # ── 1. Random Forest Tuning ───────────────────────────────────────────────
    print(f"\n[2/4] Optuna Tuning: Random Forest ({N_TRIALS} trials, 5-Fold Stratified CV)...")
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study_rf = optuna.create_study(direction="maximize", sampler=sampler)
    study_rf.optimize(lambda trial: objective_rf(trial, X_train, y_train, cv), n_trials=N_TRIALS)

    best_params_rf = study_rf.best_params
    best_params_rf["class_weight"] = "balanced"
    best_params_rf["random_state"] = RANDOM_SEED
    best_params_rf["n_jobs"] = -1

    print(f"      Best CV ROC-AUC: {study_rf.best_value:.4f}")
    print(f"      Best Params: {study_rf.best_params}")

    model_rf = RandomForestClassifier(**best_params_rf)
    model_rf.fit(X_train, y_train)
    joblib.dump(model_rf, os.path.join(MODELS_DIR, "tuned_random_forest.joblib"))

    tuning_results.append({
        "model": "Tuned Random Forest",
        "best_cv_roc_auc": round(study_rf.best_value, 4),
        "n_trials": N_TRIALS,
        "best_params": json.dumps(study_rf.best_params),
    })

    # ── 2. XGBoost Tuning ─────────────────────────────────────────────────────
    print(f"\n[3/4] Optuna Tuning: XGBoost ({N_TRIALS} trials, 5-Fold Stratified CV)...")
    study_xgb = optuna.create_study(direction="maximize", sampler=sampler)
    study_xgb.optimize(
        lambda trial: objective_xgb(trial, X_train, y_train, cv, scale_pos_weight),
        n_trials=N_TRIALS,
    )

    best_params_xgb = study_xgb.best_params
    best_params_xgb["scale_pos_weight"] = scale_pos_weight
    best_params_xgb["random_state"] = RANDOM_SEED
    best_params_xgb["n_jobs"] = -1
    best_params_xgb["eval_metric"] = "logloss"

    print(f"      Best CV ROC-AUC: {study_xgb.best_value:.4f}")
    print(f"      Best Params: {study_xgb.best_params}")

    model_xgb = xgb.XGBClassifier(**best_params_xgb)
    model_xgb.fit(X_train, y_train)
    joblib.dump(model_xgb, os.path.join(MODELS_DIR, "tuned_xgboost.joblib"))

    tuning_results.append({
        "model": "Tuned XGBoost",
        "best_cv_roc_auc": round(study_xgb.best_value, 4),
        "n_trials": N_TRIALS,
        "best_params": json.dumps(study_xgb.best_params),
    })

    # ── 3. LightGBM Tuning ────────────────────────────────────────────────────
    print(f"\n[4/4] Optuna Tuning: LightGBM ({N_TRIALS} trials, 5-Fold Stratified CV)...")
    study_lgb = optuna.create_study(direction="maximize", sampler=sampler)
    study_lgb.optimize(
        lambda trial: objective_lgb(trial, X_train, y_train, cv, scale_pos_weight),
        n_trials=N_TRIALS,
    )

    best_params_lgb = study_lgb.best_params
    best_params_lgb["scale_pos_weight"] = scale_pos_weight
    best_params_lgb["random_state"] = RANDOM_SEED
    best_params_lgb["n_jobs"] = -1
    best_params_lgb["verbosity"] = -1

    print(f"      Best CV ROC-AUC: {study_lgb.best_value:.4f}")
    print(f"      Best Params: {study_lgb.best_params}")

    model_lgb = lgb.LGBMClassifier(**best_params_lgb)
    model_lgb.fit(X_train, y_train)
    joblib.dump(model_lgb, os.path.join(MODELS_DIR, "tuned_lightgbm.joblib"))

    tuning_results.append({
        "model": "Tuned LightGBM",
        "best_cv_roc_auc": round(study_lgb.best_value, 4),
        "n_trials": N_TRIALS,
        "best_params": json.dumps(study_lgb.best_params),
    })

    # ── Save Tuning Summary CSV & Report Markdown ─────────────────────────────
    summary_df = pd.DataFrame(tuning_results)
    summary_df.to_csv(os.path.join(PROC_DIR, "tuning_summary.csv"), index=False)

    generate_tuning_report(study_rf, study_xgb, study_lgb, scale_pos_weight)

    print("\n" + "=" * 65)
    print("[OK] Ensemble tuning complete! Saved tuned models to models/")
    print("     Generated: reports/hyperparameter_tuning_report.md")
    print("=" * 65)


# ── Report Generation ─────────────────────────────────────────────────────────

def generate_tuning_report(study_rf, study_xgb, study_lgb, scale_pos_weight):
    """Write the hyperparameter tuning report."""

    report_md = f"""# Hyperparameter Tuning Report
## Healthcare Pathways AI — Day 2 Ensemble Optimization

**Generated by:** `src/train_ensembles.py`  
**Optimization Framework:** Optuna (TPE Sampler, Seed=42)  
**Cross-Validation:** 5-Fold Stratified K-Fold CV  
**Optimization Metric:** ROC-AUC  

---

## 1. Class Imbalance Strategy

- **Imbalance Ratio:** ~70% Negative (Low Risk) / 30% Positive (High Risk)
- **Strategy Selected:** Cost-sensitive Loss Weighting
  - **Random Forest:** `class_weight='balanced'`
  - **XGBoost & LightGBM:** `scale_pos_weight = {scale_pos_weight:.3f}` ($N_{{neg}} / N_{{pos}}$)
- **Rationale:** Cost-sensitive weighting directly scales the loss gradient for minority class errors without resampling the dataset. This preserves real feature distributions, maintains natural data size, and leads to superior probability calibration compared to SMOTE.

---

## 2. Optuna Search Spaces & Tuning Results

### Summary Table

| Model | Optimization Metric (5-Fold CV ROC-AUC) | Trials | Key Best Parameters |
|:---|---:|---:|:---|
| **Tuned Random Forest** | **{study_rf.best_value:.4f}** | {N_TRIALS} | `n_estimators`: {study_rf.best_params.get('n_estimators')}, `max_depth`: {study_rf.best_params.get('max_depth')}, `min_samples_split`: {study_rf.best_params.get('min_samples_split')} |
| **Tuned XGBoost** | **{study_xgb.best_value:.4f}** | {N_TRIALS} | `n_estimators`: {study_xgb.best_params.get('n_estimators')}, `max_depth`: {study_xgb.best_params.get('max_depth')}, `learning_rate`: {study_xgb.best_params.get('learning_rate'):.4f} |
| **Tuned LightGBM** | **{study_lgb.best_value:.4f}** | {N_TRIALS} | `n_estimators`: {study_lgb.best_params.get('n_estimators')}, `num_leaves`: {study_lgb.best_params.get('num_leaves')}, `learning_rate`: {study_lgb.best_params.get('learning_rate'):.4f} |

---

## 3. Best Parameter Configurations

### Tuned Random Forest
```json
{json.dumps(study_rf.best_params, indent=2)}
```

### Tuned XGBoost
```json
{json.dumps(study_xgb.best_params, indent=2)}
```

### Tuned LightGBM
```json
{json.dumps(study_lgb.best_params, indent=2)}
```

---

## 4. Next Steps
Run `python src/evaluate_ensembles.py` to evaluate tuned ensembles against the Day 1 baseline using test set ROC-AUC, PR-AUC, and SHAP explainability.
"""

    with open(os.path.join(REPORTS_DIR, "hyperparameter_tuning_report.md"), "w", encoding="utf-8") as f:
        f.write(report_md)


if __name__ == "__main__":
    train_and_tune_ensembles()
