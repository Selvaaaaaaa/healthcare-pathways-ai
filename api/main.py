"""
main.py
=======
FastAPI Production API for Retinal Diabetic Retinopathy (DR) Screening.

Endpoints:
  - GET  /health  : Health check, model status, and device info.
  - POST /predict : Accepts an uploaded image file, processes image via PyTorch,
                    returns predicted severity class, class probabilities, and
                    specialist referral recommendation.

Run via Uvicorn:
    python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

import io
import os
import sys
from contextlib import asynccontextmanager

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# Add project root to sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.transfer_model import build_transfer_resnet18
from src.cnn_model import CustomRetinalCNN

# Global state for loaded model
model_state = {}

CLASSES = [
    {"id": 0, "name": "No DR (Healthy)", "referral": False},
    {"id": 1, "name": "Mild DR", "referral": False},
    {"id": 2, "name": "Moderate DR", "referral": True},
    {"id": 3, "name": "Severe DR", "referral": True},
]

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model ONCE at application startup."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = os.path.join(ROOT, "models")
    champion_path = os.path.join(models_dir, "dr_classifier.pt")
    resnet_path = os.path.join(models_dir, "resnet18_dr.pt")
    cnn_path = os.path.join(models_dir, "custom_cnn.pt")

    model_path = champion_path if os.path.exists(champion_path) else resnet_path

    if os.path.exists(model_path):
        model = build_transfer_resnet18(num_classes=4, freeze_base=False)
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception:
            model = CustomRetinalCNN(num_classes=4)
            model.load_state_dict(torch.load(model_path, map_location=device))

        model.to(device)
        model.eval()
        model_state["model"] = model
        model_state["device"] = device
        model_state["status"] = "loaded"
        print(f"[OK] FastAPI Startup: Successfully loaded model from {model_path} on {device}")
    else:
        model_state["status"] = "not_found"
        print("⚠ FastAPI Startup: Model file not found. Place model in models/dr_classifier.pt")

    yield
    model_state.clear()


app = FastAPI(
    title="Healthcare Pathways AI — DR Image Screening API",
    description="Deep Learning Production Service for Retinal Diabetic Retinopathy Screening",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def read_root():
    return {
        "service": "Healthcare Pathways AI — DR Screening API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict (POST)",
    }


@app.get("/health")
def health_check():
    status = model_state.get("status", "unknown")
    if status != "loaded":
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "Model not loaded. Run train_dl.py and evaluate_dl.py first."},
        )
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(model_state.get("device")),
        "num_classes": len(CLASSES),
    }


@app.post("/predict")
async def predict_dr_severity(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file, computes inference, and returns predicted DR severity class and referral alert.
    """
    if model_state.get("status") != "loaded":
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File uploaded must be an image (JPEG/PNG).")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(model_state["device"])

        with torch.no_grad():
            logits = model_state["model"](tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        pred_id = int(torch.argmax(logits, dim=1).item())
        pred_meta = CLASSES[pred_id]

        confidence_scores = {CLASSES[i]["name"]: float(round(probs[i], 4)) for i in range(len(CLASSES))}

        referral_required = pred_meta["referral"]
        recommendation = (
            "URGENT: Refer patient to an ophthalmologist for detailed retinal examination."
            if referral_required
            else "Routine annual diabetic eye screening recommended."
        )

        return {
            "predicted_class_id": pred_id,
            "predicted_class_name": pred_meta["name"],
            "confidence_scores": confidence_scores,
            "refer_to_specialist": referral_required,
            "recommendation": recommendation,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
