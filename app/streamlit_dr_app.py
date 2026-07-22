"""
streamlit_dr_app.py
===================
Streamlit Retinal Diabetic Retinopathy Screening App (Day 3).

Allows clinicians/users to upload a retinal fundus image or select a sample image,
connects to the FastAPI endpoint (http://127.0.0.1:8000/predict) for deep learning
inference, and presents:
  - Fundus Image Visualizer
  - Predicted DR Severity Badge (No DR / Mild / Moderate / Severe)
  - Class Probability Confidence Distribution Chart
  - Specialist Referral Recommendation Alert Banner
  - Educational Disclaimer Banner

Run:
    streamlit run app/streamlit_dr_app.py
"""

import io
import os
import sys
import requests
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from torchvision import transforms
import streamlit as st

# Add project root to sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.transfer_model import build_transfer_resnet18
from src.cnn_model import CustomRetinalCNN

st.set_page_config(
    page_title="Healthcare Pathways AI — Retinal DR Screening",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://127.0.0.1:8000/predict"

# Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F766E;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .disclaimer-banner {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        color: #92400E;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }
    .badge-no-dr {
        background-color: #DCFCE7; color: #166534; padding: 0.8rem 1.2rem;
        border-radius: 8px; border-left: 6px solid #22C55E; font-weight: bold; font-size: 1.3rem;
    }
    .badge-mild {
        background-color: #FEF9C3; color: #854D0E; padding: 0.8rem 1.2rem;
        border-radius: 8px; border-left: 6px solid #EAB308; font-weight: bold; font-size: 1.3rem;
    }
    .badge-moderate {
        background-color: #FFEDD5; color: #9A3412; padding: 0.8rem 1.2rem;
        border-radius: 8px; border-left: 6px solid #F97316; font-weight: bold; font-size: 1.3rem;
    }
    .badge-severe {
        background-color: #FEE2E2; color: #991B1B; padding: 0.8rem 1.2rem;
        border-radius: 8px; border-left: 6px solid #EF4444; font-weight: bold; font-size: 1.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Direct Inference Fallback ──────────────────────────────────────────────────

def run_direct_inference(image: Image.Image):
    """Fallback in-process inference if FastAPI server is offline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = os.path.join(ROOT, "models")
    champion_path = os.path.join(models_dir, "dr_classifier.pt")
    resnet_path = os.path.join(models_dir, "resnet18_dr.pt")

    model_path = champion_path if os.path.exists(champion_path) else resnet_path
    if not os.path.exists(model_path):
        return None

    try:
        model = build_transfer_resnet18(num_classes=4, freeze_base=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception:
        model = CustomRetinalCNN(num_classes=4)
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.to(device)
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    tensor = tf(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    pred_id = int(np.argmax(probs))
    classes = ["No DR (Healthy)", "Mild DR", "Moderate DR", "Severe DR"]

    return {
        "predicted_class_id": pred_id,
        "predicted_class_name": classes[pred_id],
        "confidence_scores": {classes[i]: float(probs[i]) for i in range(4)},
        "refer_to_specialist": pred_id >= 2,
        "recommendation": "URGENT: Ophthalmologist referral required." if pred_id >= 2 else "Routine annual screening.",
    }


def main():
    st.markdown('<div class="main-title">👁️ Healthcare Pathways AI — DR Screening Portal</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Deep Learning Image-Based Retinal Diabetic Retinopathy Classification & Referral Engine</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="disclaimer-banner">
            ⚠️ <strong>PROTOTYPE DISCLAIMER:</strong> This AI screening tool is built strictly for educational and research demonstration purposes. It is <strong>not a certified diagnostic medical device</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("📁 Image Selection")
    source_option = st.sidebar.radio("Select Image Source", ["Upload Retinal Image", "Select Synthetic Test Sample"])

    selected_image = None
    image_name = ""

    if source_option == "Upload Retinal Image":
        uploaded_file = st.sidebar.file_uploader("Upload Retinal Fundus Image (PNG/JPG)", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            selected_image = Image.open(uploaded_file)
            image_name = uploaded_file.name
    else:
        sample_dir = os.path.join(ROOT, "data", "images", "test")
        sample_images = []
        if os.path.exists(sample_dir):
            for root_d, _, files in os.walk(sample_dir):
                for f in files:
                    if f.endswith(".png"):
                        sample_images.append(os.path.join(root_d, f))

        if sample_images:
            chosen_path = st.sidebar.selectbox("Choose Synthetic Fundus Sample", sample_images, format_func=lambda p: os.path.basename(p))
            if chosen_path:
                selected_image = Image.open(chosen_path)
                image_name = os.path.basename(chosen_path)
        else:
            st.sidebar.warning("No test samples found. Run `python src/generate_images.py` first.")

    col_left, col_right = st.columns([1, 1.2])

    if selected_image is not None:
        with col_left:
            st.subheader("🖼️ Retinal Fundus Image View")
            st.image(selected_image, use_container_width=True, caption=f"Selected Image: {image_name}")

        with col_right:
            st.subheader("🩺 Diagnostic Screening Results")

            # Call API or Direct Fallback
            result = None
            try:
                img_byte_arr = io.BytesIO()
                selected_image.convert("RGB").save(img_byte_arr, format="PNG")
                img_byte_arr.seek(0)
                response = requests.post(API_URL, files={"file": ("image.png", img_byte_arr, "image/png")}, timeout=3)
                if response.status_code == 200:
                    result = response.json()
            except Exception:
                result = run_direct_inference(selected_image)

            if result is not None:
                class_name = result["predicted_class_name"]
                pred_id = result["predicted_class_id"]
                scores = result["confidence_scores"]
                referral = result["refer_to_specialist"]

                # Severity Badge
                badge_class = ["badge-no-dr", "badge-mild", "badge-moderate", "badge-severe"][pred_id]
                st.markdown(f'<div class="{badge_class}">PREDICTED SEVERITY: {class_name.upper()}</div>', unsafe_allow_html=True)

                st.write("")

                # Referral Alert Banner
                if referral:
                    st.error(f"🚨 **REFERRAL ALERT:** {result['recommendation']}")
                else:
                    st.success(f"✅ **ROUTINE SCREENING:** {result['recommendation']}")

                st.write("")
                st.subheader("📊 Class Confidence Probabilities")

                # Probability Bar Chart
                df_scores = pd.DataFrame(list(scores.items()), columns=["Severity Class", "Probability"])
                fig, ax = plt.subplots(figsize=(6, 3))
                colors = ["#22C55E", "#EAB308", "#F97316", "#EF4444"]
                ax.barh(df_scores["Severity Class"], df_scores["Probability"], color=colors, alpha=0.85)
                ax.set_xlim(0, 1.0)
                ax.set_xlabel("Confidence Probability")
                for i, v in enumerate(df_scores["Probability"]):
                    ax.text(v + 0.02, i, f"{v:.1%}", va="center", fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.error("⚠ Inference error: Model artefacts or API service unavailable.")
    else:
        st.info("👈 Upload an image or select a sample in the sidebar to perform DR screening.")


if __name__ == "__main__":
    main()
