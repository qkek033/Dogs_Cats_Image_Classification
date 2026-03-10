import json
import logging
from logging.handlers import TimedRotatingFileHandler
import time
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, File, Header, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError

from app.models.inference import predict_image

router = APIRouter()

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
LOG_DIR = Path("logs")
LOG_PATH = LOG_DIR / "upload_audit.log"
LOG_BACKUP_DAYS = 30

LOG_DIR.mkdir(parents=True, exist_ok=True)
audit_logger = logging.getLogger("upload_audit")
if not audit_logger.handlers:
    audit_logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    file_handler = TimedRotatingFileHandler(
        filename=str(LOG_PATH),
        when="midnight",
        interval=1,
        backupCount=LOG_BACKUP_DAYS,
        encoding="utf-8",
    )
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    audit_logger.addHandler(console_handler)
    audit_logger.addHandler(file_handler)
    audit_logger.propagate = False


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _audit_log(payload: dict) -> None:
    audit_logger.info(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


@router.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    x_user_name: str | None = Header(default=None),
):
    start = time.time()
    request_id = str(uuid.uuid4())
    actor = (x_user_name or "anonymous").strip() or "anonymous"
    client_ip = _client_ip(request)
    content_type = (file.content_type or "").lower() if file else ""
    file_name = file.filename if file and file.filename else "unknown"

    base_log = {
        "ts": int(time.time()),
        "ts_iso": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "actor": actor,
        "client_ip": client_ip,
        "file_name": file_name,
        "content_type": content_type,
    }

    if not file:
        _audit_log({**base_log, "status": "rejected", "reason": "no_file"})
        raise HTTPException(status_code=400, detail="No file uploaded.")

    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        _audit_log(
            {
                **base_log,
                "status": "rejected",
                "reason": "unsupported_content_type",
            }
        )
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Use jpg/png/webp.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        _audit_log({**base_log, "status": "rejected", "reason": "empty_file"})
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(image_bytes) > MAX_UPLOAD_SIZE:
        _audit_log(
            {
                **base_log,
                "status": "rejected",
                "reason": "file_too_large",
                "size_bytes": len(image_bytes),
            }
        )
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max allowed size is {MAX_UPLOAD_SIZE // (1024 * 1024)}MB.",
        )

    try:
        img = Image.open(BytesIO(image_bytes))
        img.verify()
    except (UnidentifiedImageError, OSError, ValueError):
        _audit_log(
            {
                **base_log,
                "status": "rejected",
                "reason": "invalid_or_corrupted_image",
                "size_bytes": len(image_bytes),
            }
        )
        raise HTTPException(status_code=400, detail="Invalid or corrupted image file.")

    label, confidence, cam, rejected_unknown, reject_reason = predict_image(image_bytes)
    latency_ms = int((time.time() - start) * 1000)

    if rejected_unknown:
        _audit_log(
            {
                **base_log,
                "status": "rejected",
                "reason": "not_dog_or_cat",
                "sub_reason": reject_reason,
                "size_bytes": len(image_bytes),
                "confidence": round(float(confidence), 6),
                "latency_ms": latency_ms,
            }
        )
        raise HTTPException(
            status_code=422,
            detail="Uploaded image does not look like a dog/cat photo.",
        )

    _audit_log(
        {
            **base_log,
            "status": "accepted",
            "size_bytes": len(image_bytes),
            "prediction": label,
            "confidence": round(float(confidence), 6),
            "latency_ms": latency_ms,
        }
    )

    return {
        "label": label,
        "confidence": confidence,
        "cam": cam.tolist(),
        "request_id": request_id,
        "latency_ms": latency_ms,
    }
