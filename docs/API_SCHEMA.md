# API 스키마 문서

## 1. 기본 정보
- 기본 URL (로컬): `http://127.0.0.1:8000`
- Content-Type:
  - 요청: `multipart/form-data` (`POST /predict`)
  - 응답: `application/json`
- Swagger UI: `GET /docs`
- OpenAPI JSON: `GET /openapi.json`

## 2. 공통 스키마
### 2.1 Health 응답
```json
{
  "type": "object",
  "required": ["status", "model_loaded"],
  "properties": {
    "status": { "type": "string", "example": "ok" },
    "model_loaded": { "type": "boolean", "example": true }
  }
}
```

### 2.2 Predict 성공 응답
```json
{
  "type": "object",
  "required": ["label", "confidence", "cam", "request_id", "latency_ms"],
  "properties": {
    "label": { "type": "string", "enum": ["dog", "cat"] },
    "confidence": { "type": "number" },
    "cam": {
      "type": "array",
      "description": "2D heatmap array (224x224)",
      "items": { "type": "array", "items": { "type": "number" } }
    },
    "request_id": { "type": "string" },
    "latency_ms": { "type": "integer" }
  }
}
```

### 2.3 에러 응답
FastAPI 기본 에러 포맷(`HTTPException`)을 사용합니다.
```json
{
  "detail": "Invalid or corrupted image file."
}
```

## 3. 엔드포인트 명세
### 3.1 GET /health
응답 예시:
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### 3.2 POST /predict
설명: 이미지 파일 1개를 업로드해 강아지/고양이 분류 및 CAM을 반환합니다.

요청:
- Content-Type: `multipart/form-data`
- Form field:
  - `file` (required): `jpg/png/jpeg/webp`
- Header (optional):
  - `X-User-Name`: 업로드 사용자 식별용 문자열 (감사 로그 기록)

성공 응답 예시:
```json
{
  "label": "dog",
  "confidence": 0.7342,
  "cam": [[0.0, 0.01], [0.02, 0.03]],
  "request_id": "9b76dca6-739b-4c6e-a10f-6d2b9a3d72c8",
  "latency_ms": 57
}
```

에러 코드:
- `400` `No file uploaded.`
- `400` `Uploaded file is empty.`
- `400` `Invalid or corrupted image file.`
- `413` `File too large. Max allowed size is 10MB.`
- `415` `Unsupported file type: ... Use jpg/png/webp.`

## 4. 업로드 감사 로그
로그 파일:
- `logs/upload_audit.log`
- 일별 회전 파일: `logs/upload_audit.log.YYYY-MM-DD`

로그 필드:
- `ts`, `request_id`, `actor`, `client_ip`
- `file_name`, `content_type`, `size_bytes`
- `status` (`accepted` / `rejected`)
- `reason` (거절 사유)
- `prediction`, `confidence`, `latency_ms` (성공 시)

로그 예시:
```json
{"ts": 1772700705, "request_id": "...", "actor": "hyeon", "client_ip": "127.0.0.1", "file_name": "dog.jpg", "content_type": "image/jpeg", "status": "accepted", "size_bytes": 182311, "prediction": "dog", "confidence": 0.7342, "latency_ms": 57}
```

## 5. OpenAI 호환 API
현재 버전에서는 아래 엔드포인트를 제공하지 않습니다.
- `GET /v1/models`
- `POST /v1/chat/completions`

필요 시 추후 별도 라우터로 확장 예정입니다.
