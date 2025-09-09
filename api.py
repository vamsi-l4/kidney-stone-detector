# api.py
import io
import json
from pathlib import Path

import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torchvision.models import mobilenet_v2  # type: ignore
from torchvision import transforms  # type: ignore
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends  # type: ignore
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials  # type: ignore
from fastapi.responses import JSONResponse  # type: ignore
from PIL import Image  # type: ignore

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "kidney_mobilenet_v2.pt"
LABELS_PATH = MODELS_DIR / "labels.json"

# API Token
API_TOKEN = "mysecrettoken"

# Load labels
if not LABELS_PATH.exists():
    raise RuntimeError(f"Labels file not found: {LABELS_PATH}")
with open(LABELS_PATH, "r") as f:
    idx_to_class = json.load(f)
idx_to_class = {int(k): v for k, v in idx_to_class.items()}

# Build model
num_classes = len(idx_to_class)
model = mobilenet_v2(weights=None)  # no pretrained weights
in_feats = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_feats, num_classes)
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state["state_dict"])
model.eval()
model.to(DEVICE)

# Preprocessing
preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Auth
auth_scheme = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid or missing token")
    return True

# App
app = FastAPI(title="Kidney Stone Detector API")

@app.get("/healthz")
def health_check():
    """Health check endpoint for Render"""
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), authorized: bool = Depends(verify_token)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_t = preprocess(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(img_t)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = int(torch.argmax(probs))
            pred_class = idx_to_class[pred_idx]
            confidence = float(probs[pred_idx])

        return JSONResponse({
            "class": pred_class,
            "confidence": round(confidence, 4)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
