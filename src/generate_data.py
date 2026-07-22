"""
generate_data.py
================
Synthetic Patient Dataset Generator for Healthcare Pathways AI (Day 1).

Generates a realistic, clinically plausible synthetic dataset of ~6,000 patients
with demographics, medical history, vitals/labs, lifestyle factors, and a
genetic risk proxy.

Target variable ``high_risk_complication`` (binary, ~30% positive) is derived from
a weighted logistic sigmoid over clinical features + Gaussian noise, ensuring the
label is learnable but not trivially separable.

Intentional messiness injected:
  - 5–8% missing values in selected lab/lifestyle columns
  - ~1% outliers in HbA1c, fasting glucose, and systolic BP
  - ~70/30 class imbalance in the target

Run:
    python src/generate_data.py

Output:
    data/raw/patients_raw.csv
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.special import expit  # numerically stable sigmoid

# ── Configuration ────────────────────────────────────────────────────────────
RANDOM_SEED = 42
N_PATIENTS = 6_000
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "patients_raw.csv")

# Missing-value injection: {column: fraction_missing}
MISSING_CONFIG = {
    "hba1c": 0.06,
    "fasting_glucose": 0.05,
    "creatinine": 0.07,
    "medication_adherence_pct": 0.05,
    "diet_score": 0.06,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _activity_penalty(levels: np.ndarray) -> np.ndarray:
    """Convert physical activity labels to log-odds penalty."""
    mapping = {"sedentary": 0.9, "low": 0.4, "moderate": -0.2, "high": -0.7}
    return np.array([mapping[lvl] for lvl in levels])


# ── Main generation function ──────────────────────────────────────────────────

def generate_patients(n: int = N_PATIENTS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate a synthetic patient dataset with realistic correlations.

    Parameters
    ----------
    n : int
        Number of patient records to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Raw patient dataframe (before any preprocessing).
    """
    rng = np.random.default_rng(seed)

    # ── Demographics ──────────────────────────────────────────────────────────
    age = rng.normal(55, 15, n).clip(18, 85).astype(int)
    sex = rng.choice(["M", "F"], n, p=[0.48, 0.52])
    # BMI: log-normal gives a realistic right-skewed body-weight distribution
    bmi = np.exp(rng.normal(np.log(27), 0.2, n)).clip(16, 45)
    smoking_status = rng.choice(
        ["never", "former", "current"], n, p=[0.50, 0.30, 0.20]
    )
    alcohol_use = rng.choice(
        ["none", "moderate", "heavy"], n, p=[0.45, 0.40, 0.15]
    )

    # ── Medical history ───────────────────────────────────────────────────────
    years_since_diagnosis = rng.exponential(8, n).clip(0, 30)
    hypertension_flag = (rng.random(n) < 0.40).astype(int)
    diabetes_flag = (rng.random(n) < 0.35).astype(int)
    ckd_flag = (rng.random(n) < 0.15).astype(int)
    family_history_flag = (rng.random(n) < 0.45).astype(int)

    # ── Vitals / Lab values (correlated with medical history) ─────────────────
    # HbA1c: 5.0 baseline + 2.5 bump if diabetic
    hba1c = (5.0 + diabetes_flag * 2.5 + rng.normal(0, 1.2, n)).clip(4.0, 14.0)

    # Fasting glucose: 90 baseline + 70 bump if diabetic
    fasting_glucose = (90 + diabetes_flag * 70 + rng.normal(0, 30, n)).clip(70, 400)

    # Blood pressure: higher baseline if hypertensive
    systolic_bp = (115 + hypertension_flag * 25 + rng.normal(0, 15, n)).clip(90, 200)
    diastolic_bp = (75 + hypertension_flag * 10 + rng.normal(0, 10, n)).clip(60, 130)

    # Cholesterol (no strong flag correlation — diet-driven)
    cholesterol_ldl = rng.normal(120, 35, n).clip(50, 250)
    cholesterol_hdl = rng.normal(50, 15, n).clip(20, 100)

    # Creatinine: higher if CKD; exponential right tail mimics real renal labs
    creatinine = (0.9 + ckd_flag * 2.0 + rng.exponential(0.3, n)).clip(0.4, 10.0)

    # ── Lifestyle factors ─────────────────────────────────────────────────────
    physical_activity_level = rng.choice(
        ["sedentary", "low", "moderate", "high"], n, p=[0.30, 0.35, 0.25, 0.10]
    )
    # Diet score: slight positive correlation with physical activity
    diet_score = rng.normal(5.5, 2.0, n).clip(0, 10)
    # Medication adherence: Beta(5,2) → most patients are moderate-to-high adherent
    medication_adherence_pct = (rng.beta(5, 2, n) * 100).round(1)

    # ── Genetic risk proxy ────────────────────────────────────────────────────
    # Beta(2, 5): right-skewed — most patients have low genetic risk
    genetic_risk_score = rng.beta(2, 5, n).round(4)

    # ── Target: high_risk_complication ───────────────────────────────────────
    # Weighted logistic model with clinically plausible coefficients + noise.
    # Positive coefficients → increases risk; negative → protective.
    activity_vals = _activity_penalty(physical_activity_level)

    log_odds = (
        -5.0                                          # base intercept (controls prevalence)
        + 0.60 * (hba1c - 6.5)                       # HbA1c above target = higher risk
        + 0.02 * (fasting_glucose - 100)              # elevated glucose = higher risk
        + 0.03 * (systolic_bp - 120)                  # elevated BP = higher risk
        + 0.08 * (bmi - 25)                           # excess BMI = higher risk
        + 1.20 * creatinine                           # renal burden = strong risk factor
        + 2.00 * genetic_risk_score                   # genetic predisposition
        + 0.50 * hypertension_flag                    # comorbidity penalty
        + 0.80 * diabetes_flag                        # comorbidity penalty
        + 0.60 * ckd_flag                             # comorbidity penalty
        + 0.30 * family_history_flag                  # family history modest effect
        - 0.02 * medication_adherence_pct             # adherence is protective
        - 0.20 * diet_score                           # good diet is protective
        + 0.40 * activity_vals                        # sedentary = penalty, active = benefit
        + rng.normal(0, 0.5, n)                       # irreducible noise
    )

    prob = expit(log_odds)
    # Threshold calibrated to produce exactly ~30% positive class (~70/30 imbalance)
    threshold = np.percentile(prob, 70)  # Top 30% are high risk
    high_risk_complication = (prob >= threshold).astype(int)

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame(
        {
            "patient_id": [f"P{i:05d}" for i in range(1, n + 1)],
            "age": age,
            "sex": sex,
            "bmi": bmi.round(2),
            "smoking_status": smoking_status,
            "alcohol_use": alcohol_use,
            "years_since_diagnosis": years_since_diagnosis.round(1),
            "hypertension_flag": hypertension_flag,
            "diabetes_flag": diabetes_flag,
            "ckd_flag": ckd_flag,
            "family_history_flag": family_history_flag,
            "hba1c": hba1c.round(2),
            "fasting_glucose": fasting_glucose.round(1),
            "systolic_bp": systolic_bp.round(1),
            "diastolic_bp": diastolic_bp.round(1),
            "cholesterol_ldl": cholesterol_ldl.round(1),
            "cholesterol_hdl": cholesterol_hdl.round(1),
            "creatinine": creatinine.round(3),
            "physical_activity_level": physical_activity_level,
            "diet_score": diet_score.round(2),
            "medication_adherence_pct": medication_adherence_pct,
            "genetic_risk_score": genetic_risk_score,
            "high_risk_complication": high_risk_complication,
        }
    )

    # ── Inject missing values ─────────────────────────────────────────────────
    for col, rate in MISSING_CONFIG.items():
        mask = rng.random(n) < rate
        df.loc[mask, col] = np.nan

    # ── Inject outliers (~1%) ─────────────────────────────────────────────────
    # Use a separate RNG draw to avoid displacing main seed sequence
    outlier_rng = np.random.RandomState(seed + 1)
    n_out = max(1, int(n * 0.01))
    idx = outlier_rng.choice(df.index, size=n_out * 3, replace=False)

    # HbA1c above physiological maximum (~14)
    df.loc[idx[:n_out], "hba1c"] = outlier_rng.uniform(14.5, 18.0, n_out)
    # Fasting glucose > 400 (extreme hyperglycaemia)
    df.loc[idx[n_out : n_out * 2], "fasting_glucose"] = outlier_rng.uniform(401, 520, n_out)
    # Systolic BP > 200 (hypertensive crisis territory)
    df.loc[idx[n_out * 2 :], "systolic_bp"] = outlier_rng.uniform(201, 240, n_out)

    return df


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Healthcare Pathways AI — Synthetic Dataset Generator")
    print("=" * 60)

    df = generate_patients()

    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(OUTPUT_PATH))
    os.makedirs(out_dir, exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n[OK] Dataset saved to: {os.path.abspath(OUTPUT_PATH)}")
    print(f"  Shape          : {df.shape}")
    print(f"  Missing values : {df.isnull().sum().sum()} total cells")
    pos = df['high_risk_complication'].sum()
    print(f"  Class balance  : {pos} high-risk ({100*pos/len(df):.1f}%) | "
          f"{len(df)-pos} low-risk ({100*(len(df)-pos)/len(df):.1f}%)")
    print(f"  Columns        : {list(df.columns)}")
    print("\nDone.")
