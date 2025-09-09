# api.py
import io
import json
import os
import threading
import logging
from pathlib import Path
from typing import Optional

import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torchvision.models import mobilenet_v2  # type: ignore
from torchvision import transforms  # type: ignore

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kidney-api")

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths and config via env
MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(MODELS_DIR / "kidney_mobilenet_v2.pt")))
LABELS_PATH = Path(os.getenv("LABELS_PATH", str(MODELS_DIR / "labels.json")))
API_TOKEN = os.getenv("API_TOKEN", "mysecrettoken")

# Globals for model
_model_lock = threading.Lock()
model: Optional[torch.nn.Module] = None
idx_to_class = {}

def load_labels():
    global idx_to_class
    if not LABELS_PATH.exists():
        logger.warning(f"Labels file not found at {LABELS_PATH}")
        return False
    with open(LABELS_PATH, "r") as f:
        idx_to_class = {int(k): v for k, v in json.load(f).items()}
    return True

def load_model():
    global model, idx_to_class
    with _model_lock:
        if model is not None:
            return True
        if not MODEL_PATH.exists():
            logger.warning(f"Model file not found at {MODEL_PATH}")
            return False
        # ensure labels loaded
        if not idx_to_class:
            ok = load_labels()
            if not ok:
                logger.warning("Labels not loaded, aborting model load")
                return False
        num_classes = len(idx_to_class)
        net = mobilenet_v2(weights=None)  # no pretrained
        in_feats = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_feats, num_classes)
        state = torch.load(MODEL_PATH, map_location="cpu")
        net.load_state_dict(state["state_dict"])
        net.eval()
        # move to DEVICE
        net.to(DEVICE)
        model = net
        logger.info("Model loaded into memory")
        return True

# Preprocessing
preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Auth
auth_scheme = HTTPBearer()
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid or missing token")
    return True

app = FastAPI(title="Kidney Stone Detector API")

@app.on_event("startup")
def startup_event():
    logger.info("Starting up — attempting to load model & labels")
    load_labels()
    # try loading model but don't crash if missing
    load_model()

@app.get("/healthz")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), authorized: bool = Depends(verify_token)):
    if model is None:
        ok = load_model()
        if not ok:
            raise HTTPException(status_code=503, detail="Model not available. Contact admin.")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_t = preprocess(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(img_t)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = int(torch.argmax(probs))
            pred_class = idx_to_class.get(pred_idx, "Unknown")
            confidence = float(probs[pred_idx])
        return JSONResponse({"class": pred_class, "confidence": round(confidence, 4)})
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(e))
