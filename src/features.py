"""
features.py
===========
Feature engineering for Healthcare Pathways AI (Day 1).

Derives clinically meaningful composite features from the raw patient dataset.
All derived features are designed to be interpretable by clinicians and to
improve model signal without introducing data leakage.

Derived features:
    1. bmi_category            — WHO obesity classification (ordinal)
    2. comorbidity_count       — cumulative disease burden (0–3)
    3. risk_adjusted_adherence — adherence discounted by genetic risk
    4. metabolic_syndrome_proxy — binary flag for metabolic syndrome cluster

These features are added AFTER imputation but BEFORE scaling, so that
categorical bmi_category can be one-hot encoded alongside other nominals.

Usage:
    from src.features import engineer_features
    df_feat = engineer_features(df_imputed)
"""

import numpy as np
import pandas as pd


# ── WHO BMI thresholds ────────────────────────────────────────────────────────
BMI_BINS = [0, 18.5, 25.0, 30.0, np.inf]
BMI_LABELS = ["underweight", "normal", "overweight", "obese"]


def _bmi_category(bmi_series: pd.Series) -> pd.Series:
    """
    Classify BMI into WHO obesity categories.

    Rationale: Raw BMI is numeric but its clinical meaning is categorical;
    threshold-based categories may capture non-linear risk increments that
    a linear model would otherwise miss.
    """
    return pd.cut(bmi_series, bins=BMI_BINS, labels=BMI_LABELS, right=False)


def _comorbidity_count(df: pd.DataFrame) -> pd.Series:
    """
    Sum of hypertension, diabetes, and CKD binary flags.

    Rationale: Multimorbidity has a multiplicative effect on complication risk
    beyond individual conditions; a single count captures cumulative burden
    and is directly interpretable by clinicians.
    """
    return (
        df["hypertension_flag"].fillna(0)
        + df["diabetes_flag"].fillna(0)
        + df["ckd_flag"].fillna(0)
    ).astype(int)


def _risk_adjusted_adherence(df: pd.DataFrame) -> pd.Series:
    """
    Adherence penalised by genetic risk.

    Formula: medication_adherence_pct × (1 − genetic_risk_score)

    Rationale: A patient with perfect adherence but high genetic risk still
    faces elevated complication probability; this composite feature reflects
    the net protective effect of adherence after accounting for genetic burden.
    """
    return df["medication_adherence_pct"] * (1 - df["genetic_risk_score"])


def _metabolic_syndrome_proxy(df: pd.DataFrame) -> pd.Series:
    """
    Binary proxy for metabolic syndrome (simplified).

    Criteria (≥3 of the following):
      - BMI ≥ 30 (obese)
      - Fasting glucose ≥ 100 mg/dL
      - Systolic BP ≥ 130 mmHg
      - LDL ≥ 130 mg/dL
      - HDL < 40 mg/dL

    Rationale: Metabolic syndrome dramatically increases cardiovascular and
    diabetic complication risk; this composite flag encodes a clinically
    established risk cluster.
    """
    criteria = pd.DataFrame(
        {
            "obese": (df["bmi"] >= 30).astype(int),
            "high_glucose": (df["fasting_glucose"] >= 100).astype(int),
            "high_sbp": (df["systolic_bp"] >= 130).astype(int),
            "high_ldl": (df["cholesterol_ldl"] >= 130).astype(int),
            "low_hdl": (df["cholesterol_hdl"] < 40).astype(int),
        }
    )
    return (criteria.sum(axis=1) >= 3).astype(int)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features to an imputed patient dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Patient dataframe after imputation (no NaNs in key columns).

    Returns
    -------
    pd.DataFrame
        Original dataframe with four additional columns:
        ['bmi_category', 'comorbidity_count',
         'risk_adjusted_adherence', 'metabolic_syndrome_proxy']
    """
    df = df.copy()

    df["bmi_category"] = _bmi_category(df["bmi"])
    df["comorbidity_count"] = _comorbidity_count(df)
    df["risk_adjusted_adherence"] = _risk_adjusted_adherence(df)
    df["metabolic_syndrome_proxy"] = _metabolic_syndrome_proxy(df)

    return df


if __name__ == "__main__":
    # Quick smoke test
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    sample_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "patients_imputed.csv")
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
        df_feat = engineer_features(df)
        print(f"[OK] Feature engineering complete. Shape: {df_feat.shape}")
        print(df_feat[["bmi_category", "comorbidity_count",
                        "risk_adjusted_adherence", "metabolic_syndrome_proxy"]].describe(include="all"))
    else:
        print(f"⚠ Processed data not found at {sample_path}. Run data_prep.py first.")
