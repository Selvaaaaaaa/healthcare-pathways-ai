"""
data_prep.py
============
Data Preprocessing Pipeline for Healthcare Pathways AI (Day 1).

Responsibilities:
  1. Load raw patient CSV
  2. Impute missing values (per-column strategy with clinical rationale)
  3. Winsorize outliers in lab values using IQR fence
  4. Apply feature engineering (from src/features.py)
  5. Encode categorical variables
  6. Scale numeric features with StandardScaler
  7. Stratified 80/20 train/test split
  8. Save processed artefacts to data/processed/

Every preprocessing decision is documented in the function docstrings below.

Usage:
    python src/data_prep.py

Outputs (data/processed/):
    patients_imputed.csv       — after imputation + outlier handling + feature eng
    X_train.csv, X_test.csv    — scaled feature matrices
    y_train.csv, y_test.csv    — target vectors
    feature_names.txt          — ordered feature names for model interpretation
    preprocessor.joblib        — fitted ColumnTransformer (for future inference)
"""

import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ── Path configuration ─────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_PATH = os.path.join(ROOT, "data", "raw", "patients_raw.csv")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
MODELS_DIR = os.path.join(ROOT, "models")

RANDOM_SEED = 42
TEST_SIZE = 0.20

# ── Column definitions ─────────────────────────────────────────────────────────
TARGET = "high_risk_complication"
ID_COL = "patient_id"

# Lab / continuous columns where median imputation is used
# Rationale: Median is robust to the injected outliers; mean would be biased upward.
LAB_NUMERIC_COLS = ["hba1c", "fasting_glucose", "creatinine"]

# Lifestyle numeric columns where mean imputation is used
# Rationale: These are more normally distributed, so mean is appropriate.
LIFESTYLE_NUMERIC_COLS = ["diet_score", "medication_adherence_pct"]

# Remaining numeric columns (no missing values expected)
OTHER_NUMERIC_COLS = [
    "age", "bmi", "years_since_diagnosis",
    "systolic_bp", "diastolic_bp",
    "cholesterol_ldl", "cholesterol_hdl",
    "genetic_risk_score",
    "comorbidity_count", "risk_adjusted_adherence",
    "metabolic_syndrome_proxy",
]

# Binary flag columns (no imputation needed; generated without NaNs)
BINARY_COLS = [
    "hypertension_flag", "diabetes_flag", "ckd_flag", "family_history_flag",
]

# Nominal categoricals → One-Hot Encoding
# Rationale: No natural ordering; OHE prevents false ordinal relationships.
NOMINAL_CATS = ["sex", "smoking_status", "alcohol_use", "bmi_category"]

# Ordinal categoricals → Ordinal Encoding with explicit order
# Rationale: Physical activity has a meaningful progression; ordinal encoding
#            preserves this ordering for linear models.
ORDINAL_CATS = ["physical_activity_level"]
ACTIVITY_ORDER = [["sedentary", "low", "moderate", "high"]]

# IQR multiplier for winsorizing — 1.5 is standard; captures extreme outliers
IQR_MULTIPLIER = 1.5


# ── Imputation ────────────────────────────────────────────────────────────────

def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values per column type.

    Strategy summary:
      - Lab columns (HbA1c, glucose, creatinine): MEDIAN imputation
        → Robust to outliers injected in the raw data. Median is the clinically
          conservative choice; it won't overestimate average lab values due to
          extreme cases.
      - Lifestyle columns (diet_score, adherence): MEAN imputation
        → These columns have near-normal distributions; mean is unbiased.
      - Categorical columns: MODE imputation
        → Preserves the most frequent clinical category without introducing
          spurious new categories.

    Parameters
    ----------
    df : pd.DataFrame
        Raw patient dataframe (may contain NaNs).

    Returns
    -------
    pd.DataFrame
        Dataframe with all NaNs filled.
    """
    df = df.copy()

    for col in LAB_NUMERIC_COLS:
        if col in df.columns:
            median_val = df[col].median()
            n_missing = df[col].isna().sum()
            df[col] = df[col].fillna(median_val)
            if n_missing:
                print(f"  [Impute] {col:35s}: {n_missing:4d} missing -> filled with median ({median_val:.3f})")

    for col in LIFESTYLE_NUMERIC_COLS:
        if col in df.columns:
            mean_val = df[col].mean()
            n_missing = df[col].isna().sum()
            df[col] = df[col].fillna(mean_val)
            if n_missing:
                print(f"  [Impute] {col:35s}: {n_missing:4d} missing -> filled with mean ({mean_val:.3f})")

    for col in NOMINAL_CATS + ORDINAL_CATS:
        if col in df.columns and df[col].isna().any():
            mode_val = df[col].mode()[0]
            n_missing = df[col].isna().sum()
            df[col] = df[col].fillna(mode_val)
            if n_missing:
                print(f"  [Impute] {col:35s}: {n_missing:4d} missing -> filled with mode ('{mode_val}')")

    return df


# -- Outlier Winsorizing -------------------------------------------------------

def winsorize_outliers(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Cap extreme values using the IQR fence method.

    For each column:
        lower fence = Q1 − IQR_MULTIPLIER × IQR
        upper fence = Q3 + IQR_MULTIPLIER × IQR

    Values beyond these fences are clipped (winsorized), NOT dropped.

    Rationale:
      - Dropping rows would reduce dataset size and potentially bias the sample
        if outliers are correlated with clinical severity.
      - Winsorizing preserves the row while removing extreme influence on model
        parameters, especially for distance-based methods (K-Means) and
        regularised models (Logistic Regression).

    Parameters
    ----------
    df : pd.DataFrame
    cols : list[str]
        Columns to winsorize.

    Returns
    -------
    pd.DataFrame
        Dataframe with outliers capped.
    """
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - IQR_MULTIPLIER * iqr
        upper = q3 + IQR_MULTIPLIER * iqr
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        if n_outliers:
            print(f"  [Winsor] {col:35s}: {n_outliers:4d} values capped to [{lower:.2f}, {upper:.2f}]")
    return df


# ── Build Preprocessing Pipeline ──────────────────────────────────────────────

def build_preprocessor(all_numeric_cols: list[str]) -> ColumnTransformer:
    """
    Construct a ColumnTransformer that:
      1. StandardScales all numeric features
         Rationale: Required for K-Means (Euclidean distance) and Logistic
         Regression (regularisation penalty). RF/GBM don't need it but it
         doesn't hurt and keeps the pipeline uniform.
      2. OneHotEncodes nominal categoricals (sex, smoking, alcohol, bmi_category)
         Rationale: Avoids imposing false ordinal structure.
      3. OrdinalEncodes physical activity level
         Rationale: Preserves the sedentary→high ordering.

    Parameters
    ----------
    all_numeric_cols : list[str]
        All numeric columns to scale (in order).

    Returns
    -------
    ColumnTransformer
    """
    numeric_transformer = Pipeline([("scaler", StandardScaler())])

    nominal_transformer = Pipeline(
        [("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )

    ordinal_transformer = Pipeline(
        [("oe", OrdinalEncoder(categories=ACTIVITY_ORDER))]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, all_numeric_cols),
            ("nom", nominal_transformer, NOMINAL_CATS),
            ("ord", ordinal_transformer, ORDINAL_CATS),
        ],
        remainder="passthrough",
    )


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_preprocessing() -> None:
    """
    End-to-end preprocessing pipeline.

    Reads raw CSV → imputes → winsorizes → engineers features →
    encodes/scales → splits → saves processed artefacts.
    """
    from src.features import engineer_features  # local import to keep module standalone-runnable

    print("=" * 65)
    print("Healthcare Pathways AI — Data Preprocessing Pipeline")
    print("=" * 65)

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"\n[1/7] Loading raw data from {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)
    print(f"      Shape: {df.shape} | Missing cells: {df.isnull().sum().sum()}")

    # ── Impute ────────────────────────────────────────────────────────────────
    print("\n[2/7] Imputing missing values...")
    df = impute_missing(df)
    print(f"      Missing cells after imputation: {df.isnull().sum().sum()}")

    # ── Winsorize ─────────────────────────────────────────────────────────────
    winsor_cols = LAB_NUMERIC_COLS + ["systolic_bp", "diastolic_bp", "bmi"]
    print("\n[3/7] Winsorizing outliers in lab/vitals columns...")
    df = winsorize_outliers(df, winsor_cols)

    # ── Feature engineering ───────────────────────────────────────────────────
    print("\n[4/7] Engineering derived features...")
    df = engineer_features(df)
    print(f"      Added: bmi_category, comorbidity_count, risk_adjusted_adherence, metabolic_syndrome_proxy")
    print(f"      Shape after feature engineering: {df.shape}")

    # Save imputed + feature-engineered frame (before scaling) for EDA / profiling
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(os.path.join(PROCESSED_DIR, "patients_imputed.csv"), index=False)
    print("      -> Saved patients_imputed.csv")

    # ── Prepare feature matrix ─────────────────────────────────────────────────
    print("\n[5/7] Preparing feature matrix and target vector...")
    drop_cols = [ID_COL, TARGET]
    feature_df = df.drop(columns=drop_cols)
    y = df[TARGET]

    # Collect all numeric columns after feature engineering
    all_numeric = (
        LAB_NUMERIC_COLS
        + LIFESTYLE_NUMERIC_COLS
        + OTHER_NUMERIC_COLS
        + BINARY_COLS
    )
    # Keep only columns that actually exist
    all_numeric = [c for c in all_numeric if c in feature_df.columns]

    # ── Build & fit preprocessor ───────────────────────────────────────────────
    print("\n[6/7] Fitting ColumnTransformer (scale + encode)...")
    preprocessor = build_preprocessor(all_numeric)
    X_all = preprocessor.fit_transform(feature_df)

    # Reconstruct feature names
    num_names = all_numeric
    nom_names = list(
        preprocessor.named_transformers_["nom"]["ohe"].get_feature_names_out(NOMINAL_CATS)
    )
    ord_names = ORDINAL_CATS
    feature_names = num_names + nom_names + ord_names
    print(f"      Total features after encoding: {len(feature_names)}")

    # ── Train / test split ─────────────────────────────────────────────────────
    print("\n[7/7] Stratified 80/20 train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"      Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"      Train positive rate: {y_train.mean():.3f} | Test: {y_test.mean():.3f}")

    # ── Save artefacts ─────────────────────────────────────────────────────────
    pd.DataFrame(X_train, columns=feature_names).to_csv(
        os.path.join(PROCESSED_DIR, "X_train.csv"), index=False
    )
    pd.DataFrame(X_test, columns=feature_names).to_csv(
        os.path.join(PROCESSED_DIR, "X_test.csv"), index=False
    )
    pd.DataFrame(y_train).to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
    pd.DataFrame(y_test).to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"), index=False)

    with open(os.path.join(PROCESSED_DIR, "feature_names.txt"), "w") as f:
        f.write("\n".join(feature_names))

    # Save patient IDs aligned with split for cluster assignment later
    ids = df[ID_COL].values
    X_train_ids, X_test_ids = train_test_split(
        ids, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    pd.DataFrame({"patient_id": X_train_ids}).to_csv(
        os.path.join(PROCESSED_DIR, "train_ids.csv"), index=False
    )
    pd.DataFrame({"patient_id": X_test_ids}).to_csv(
        os.path.join(PROCESSED_DIR, "test_ids.csv"), index=False
    )

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.joblib"))

    print(f"\n[OK] All processed files saved to {PROCESSED_DIR}/")
    print("  X_train.csv, X_test.csv, y_train.csv, y_test.csv")
    print("  feature_names.txt, train_ids.csv, test_ids.csv")
    print(f"  preprocessor.joblib -> {MODELS_DIR}/")
    print("\nPreprocessing complete.")


if __name__ == "__main__":
    # Allow running from project root or src/
    sys.path.insert(0, ROOT)
    run_preprocessing()
