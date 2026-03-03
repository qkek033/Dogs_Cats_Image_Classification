from fastapi import APIRouter, UploadFile, File
from app.models.inference import predict_image

import time
import uuid

router = APIRouter()


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    start = time.time()

    image_bytes = await file.read()

    label, confidence, cam = predict_image(image_bytes)

    return {
        "label": label,
        "confidence": confidence,
        "cam": cam.tolist(),
        "request_id": str(uuid.uuid4()),
        "latency_ms": int((time.time() - start) * 1000)
    }