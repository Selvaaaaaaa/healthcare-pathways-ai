"""
streamlit_app.py
================
Healthcare Pathways AI — Care Coordinator Interactive Web Application (Day 2).

Simulates a clinical care coordinator portal allowing real-time patient risk evaluation:
  1. Interactive form for entering demographics, medical history, lab values, and lifestyle factors.
  2. End-to-end preprocessing via fitted ColumnTransformer (data_prep & features).
  3. Real-time complication risk prediction probability & label using the tuned champion model.
  4. Real-time patient segmentation (K-Means cluster & clinical profile mapping).
  5. Patient-specific local risk factor analysis (Top 3 contributing drivers).
  6. Educational disclaimer banner.

Run:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Add project root to sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.features import engineer_features

warnings.filterwarnings("ignore")

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Healthcare Pathways AI — Care Coordinator Portal",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .disclaimer-banner {
        background-color: #EFF6FF;
        border-left: 5px solid #3B82F6;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        color: #1E40AF;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }
    .metric-card-high {
        background-color: #FEF2F2;
        border: 1px solid #FCA5A5;
        border-left: 6px solid #EF4444;
        padding: 1.2rem;
        border-radius: 8px;
    }
    .metric-card-low {
        background-color: #F0FDF4;
        border: 1px solid #86EFAC;
        border-left: 6px solid #22C55E;
        padding: 1.2rem;
        border-radius: 8px;
    }
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1F2937;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_allowed_html=True,
)


# ── Load Model Artefacts ──────────────────────────────────────────────────────

@st.cache_resource
def load_artifacts():
    models_dir = os.path.join(ROOT, "models")
    proc_dir = os.path.join(ROOT, "data", "processed")

    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    best_model_path = os.path.join(models_dir, "best_risk_model.joblib")
    kmeans_path = os.path.join(models_dir, "kmeans_model.joblib")
    feature_names_path = os.path.join(proc_dir, "feature_names.txt")
    cluster_profiles_path = os.path.join(proc_dir, "cluster_profiles.csv")

    preprocessor = joblib.load(preprocessor_path) if os.path.exists(preprocessor_path) else None
    model = joblib.load(best_model_path) if os.path.exists(best_model_path) else None
    kmeans = joblib.load(kmeans_path) if os.path.exists(kmeans_path) else None

    feature_names = []
    if os.path.exists(feature_names_path):
        with open(feature_names_path) as f:
            feature_names = [line.strip() for line in f.readlines()]

    cluster_profiles = pd.read_csv(cluster_profiles_path) if os.path.exists(cluster_profiles_path) else None

    return preprocessor, model, kmeans, feature_names, cluster_profiles


# ── Calculate Local Top 3 Drivers ─────────────────────────────────────────────

def get_local_top_drivers(transformed_row, feature_names, model):
    """
    Compute patient-specific top 3 risk factors based on feature magnitude
    multiplied by model feature importances.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        importances = np.ones(len(feature_names))

    # Element-wise contribution score = |scaled feature value| * global importance
    local_impact = np.abs(transformed_row[0]) * importances
    top_indices = np.argsort(local_impact)[::-1][:3]

    top_drivers = []
    for idx in top_indices:
        fname = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
        val = transformed_row[0][idx]
        impact = local_impact[idx]
        top_drivers.append({"feature": fname, "value": round(val, 2), "impact": round(impact, 3)})

    return top_drivers


# ── App Layout ────────────────────────────────────────────────────────────────

def main():
    st.markdown('<div class="main-title">🩺 Healthcare Pathways AI — Care Coordinator Portal</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Personalized Chronic Disease Risk Scoring & Patient Segmentation Engine</div>', unsafe_allow_html=True)

    # Educational Disclaimer Banner
    st.markdown(
        """
        <div class="disclaimer-banner">
            ⚠️ <strong>PROTOTYPE DISCLAIMER:</strong> This application is a clinical decision support prototype developed strictly for educational and research demonstration purposes. It is <strong>not a certified medical device</strong> and should not be used for direct clinical diagnosis or treatment planning.
        </div>
        """,
        unsafe_allow_html=True,
    )

    preprocessor, model, kmeans, feature_names, cluster_profiles = load_artifacts()

    if preprocessor is None or model is None or kmeans is None:
        st.error("⚠ Required model artefacts not found in `models/`. Please run `python src/train_ensembles.py` and `python src/evaluate_ensembles.py` first.")
        st.stop()

    # Sidebar: Patient Information Input Form
    st.sidebar.header("📋 Patient Vitals & Clinical Data")

    with st.sidebar.form("patient_form"):
        st.subheader("1. Demographics")
        age = st.slider("Age (years)", 18, 85, 55)
        sex = st.selectbox("Sex", ["M", "F"], index=1)
        bmi = st.number_input("BMI (kg/m²)", 16.0, 45.0, 27.5, step=0.1)
        smoking_status = st.selectbox("Smoking Status", ["never", "former", "current"], index=0)
        alcohol_use = st.selectbox("Alcohol Use", ["none", "moderate", "heavy"], index=0)

        st.subheader("2. Medical History & Comorbidities")
        years_diag = st.number_input("Years Since Diagnosis", 0.0, 30.0, 6.0, step=0.5)
        htn = st.checkbox("Hypertension Flag", value=True)
        db = st.checkbox("Diabetes Flag", value=True)
        ckd = st.checkbox("Chronic Kidney Disease (CKD) Flag", value=False)
        fam_hx = st.checkbox("Family History Flag", value=True)

        st.subheader("3. Lab Values & Vitals")
        hba1c = st.number_input("HbA1c (%)", 4.0, 14.0, 7.8, step=0.1)
        glucose = st.number_input("Fasting Glucose (mg/dL)", 70.0, 400.0, 145.0, step=1.0)
        sys_bp = st.number_input("Systolic BP (mmHg)", 90.0, 200.0, 138.0, step=1.0)
        dia_bp = st.number_input("Diastolic BP (mmHg)", 60.0, 130.0, 85.0, step=1.0)
        ldl = st.number_input("LDL Cholesterol (mg/dL)", 50.0, 250.0, 135.0, step=1.0)
        hdl = st.number_input("HDL Cholesterol (mg/dL)", 20.0, 100.0, 45.0, step=1.0)
        creatinine = st.number_input("Serum Creatinine (mg/dL)", 0.4, 10.0, 1.4, step=0.1)

        st.subheader("4. Lifestyle & Genetic Proxy")
        activity = st.selectbox("Physical Activity Level", ["sedentary", "low", "moderate", "high"], index=1)
        diet = st.slider("Diet Score (0=Poor, 10=Optimal)", 0.0, 10.0, 5.0, step=0.5)
        adherence = st.slider("Medication Adherence (%)", 0.0, 100.0, 65.0, step=1.0)
        genetic_risk = st.slider("Genetic Risk Score Proxy (0-1)", 0.0, 1.0, 0.35, step=0.05)

        submit_btn = st.form_submit_button("🔍 Evaluate Patient Risk", use_container_width=True)

    # Main Body: Evaluation Output
    if submit_btn:
        # Construct DataFrame matching raw schema
        raw_input = pd.DataFrame([{
            "patient_id": "P_DEMO",
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "smoking_status": smoking_status,
            "alcohol_use": alcohol_use,
            "years_since_diagnosis": years_diag,
            "hypertension_flag": int(htn),
            "diabetes_flag": int(db),
            "ckd_flag": int(ckd),
            "family_history_flag": int(fam_hx),
            "hba1c": hba1c,
            "fasting_glucose": glucose,
            "systolic_bp": sys_bp,
            "diastolic_bp": dia_bp,
            "cholesterol_ldl": ldl,
            "cholesterol_hdl": hdl,
            "creatinine": creatinine,
            "physical_activity_level": activity,
            "diet_score": diet,
            "medication_adherence_pct": adherence,
            "genetic_risk_score": genetic_risk,
        }])

        # Apply Feature Engineering
        feat_input = engineer_features(raw_input)
        feature_df = feat_input.drop(columns=["patient_id"])

        # Re-order columns matching preprocessor expectation
        all_numeric = [
            "hba1c", "fasting_glucose", "creatinine",
            "diet_score", "medication_adherence_pct",
            "age", "bmi", "years_since_diagnosis",
            "systolic_bp", "diastolic_bp",
            "cholesterol_ldl", "cholesterol_hdl",
            "genetic_risk_score",
            "comorbidity_count", "risk_adjusted_adherence",
            "metabolic_syndrome_proxy",
            "hypertension_flag", "diabetes_flag", "ckd_flag", "family_history_flag",
        ]
        # Keep features present in preprocessor
        feature_df = feature_df[[col for col in feature_df.columns if col in all_numeric or col in ["sex", "smoking_status", "alcohol_use", "bmi_category", "physical_activity_level"]]]

        # Transform using ColumnTransformer
        transformed_row = preprocessor.transform(feature_df)

        # Predict Risk Score
        risk_prob = float(model.predict_proba(transformed_row)[0][1])
        is_high_risk = risk_prob >= 0.45

        # Predict Cluster Assignment
        cluster_id = int(kmeans.predict(transformed_row)[0])
        cluster_label = f"Cluster {cluster_id}"

        if cluster_profiles is not None and "cluster_label" in cluster_profiles.columns:
            matching_row = cluster_profiles[cluster_profiles["cluster_kmeans"] == cluster_id]
            if not matching_row.empty:
                cluster_label = matching_row["cluster_label"].values[0]

        # Extract Top 3 Drivers
        top_drivers = get_local_top_drivers(transformed_row, feature_names, model)

        # ── Display Output Cards ──────────────────────────────────────────────
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-header">🎯 Predicted Complication Risk</div>', unsafe_allow_html=True)
            if is_high_risk:
                st.markdown(
                    f"""
                    <div class="metric-card-high">
                        <h2 style="color:#DC2626; margin:0;">HIGH RISK ({risk_prob:.1%})</h2>
                        <p style="margin:5px 0 0 0; color:#991B1B;">Patient exhibits elevated probability of chronic disease complications.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="metric-card-low">
                        <h2 style="color:#16A34A; margin:0;">LOW RISK ({risk_prob:.1%})</h2>
                        <p style="margin:5px 0 0 0; color:#166534;">Patient is currently classified within manageable risk boundaries.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with col2:
            st.markdown('<div class="section-header">📊 Patient Segment / Cluster</div>', unsafe_allow_html=True)
            st.info(f"**Assigned Cohort:** {cluster_label} (Cluster ID #{cluster_id})\n\n**Clinical Focus:** Personalized care plan routing based on Day 1 K-Means segmentation.")

        st.divider()

        # ── Top Contributing Factors ──────────────────────────────────────────
        st.markdown('<div class="section-header">🔍 Top 3 Contributing Risk Drivers (Local Feature Importance)</div>', unsafe_allow_html=True)

        driver_cols = st.columns(3)
        for idx, (d_col, driver) in enumerate(zip(driver_cols, top_drivers)):
            with d_col:
                st.metric(
                    label=f"Rank #{idx+1}: {driver['feature']}",
                    value=f"{driver['value']:+.2f} std dev",
                    delta=f"Impact: {driver['impact']:.3f}",
                    delta_color="inverse" if driver['value'] > 0 else "normal",
                )

        st.divider()

        # ── Care Plan Recommendation Preview ──────────────────────────────────
        st.markdown('<div class="section-header">📋 Recommended Care Plan Interventions</div>', unsafe_allow_html=True)

        if "Low-Adherence" in cluster_label or adherence < 70:
            st.warning("👉 **Intervention 1 — Adherence Coaching:** Assign dedicated care coordinator for weekly medication adherence follow-ups.")
        if "Metabolic" in cluster_label or hba1c >= 8.0 or sys_bp >= 140:
            st.error("👉 **Intervention 2 — Metabolic Review:** Schedule immediate endocrinology consult & dietary intervention plan.")
        if ckd or creatinine >= 1.5:
            st.error("👉 **Intervention 3 — Renal Safeguard:** Nephrology monitoring recommended due to elevated creatinine burden.")
        if not is_high_risk:
            st.success("👉 **Maintenance Plan:** Continue routine 6-month preventive checkups and health literacy reinforcement.")

    else:
        st.info("👈 Enter clinical parameters in the sidebar and click **'Evaluate Patient Risk'** to view real-time predictions.")


if __name__ == "__main__":
    main()
