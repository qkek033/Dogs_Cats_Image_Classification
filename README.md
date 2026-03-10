# Dogs vs Cats Classifier

Streamlit으로 배포된 강아지/고양이 이미지 분류 AI 모델입니다.

**배포 링크**: https://dogscatsimageclassification-tc8hscjvvx7xrtfy7wtbcy.streamlit.app/

## 특징
- SimpleCNN 모델 기반
- HuggingFace Hub에서 모델 자동 다운로드
- Grad-CAM 시각화로 모델의 주목 영역 표시
- GPU 지원 (CUDA 사용 가능 시)

## 문서
- 요구사항 문서: [docs/requirements.md](docs/requirements.md)
- API 스키마 문서: [docs/API_SCHEMA.md](docs/API_SCHEMA.md)
- OpenAPI(JSON): `GET /openapi.json`

## 프로젝트 구조
```text
dogs_cats/
├── app/
│   ├── streamlit_app.py        # Streamlit UI
│   ├── api/
│   │   ├── main.py             # FastAPI 앱 엔트리
│   │   └── routes/
│   │      ├── health.py        # /health
│   │      └── predict.py       # /predict + 업로드 검증/감사로그
│   ├── data/
│   │   ├── dataset.py
│   │   ├── download.py
│   │   └── preprocess.py
│   └── models/
│      ├── model.py             # SimpleCNN / EfficientNetB7 / ViT 모델 정의
│      ├── inference.py         # 모델 로딩/추론/Unknown 거절 로직
│      ├── gradcam.py           # CAM 생성
│      └── train.py
├── data/                       # 원본/압축 데이터
├── models/                     # 체크포인트(.pth)
├── docs/
│   ├── requirements.md         # 요구사항 명세서
│   └── API_SCHEMA.md           # API 스키마 문서
├── logs/
│   └── upload_audit.log        # 감사 로그(JSON line)
├── tests/
│   ├── test_api.py
│   └── eval_model.py
├── requirements.txt
└── .env.example
```

## 0) 사전 준비
- Python 3.10+
- (권장) 가상환경

### 0-1. Python 확인
```bash
python --version
```

### 0-2. 가상환경 및 의존성 설치 (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 1) 로컬 실행
### 1-1. FastAPI 실행
```powershell
python -m uvicorn app.api.main:app --reload
```

### 1-2. Streamlit 실행
```powershell
streamlit run app/streamlit_app.py
```

## 2) API 테스트
### 2-1. Health
```bash
curl -s http://127.0.0.1:8000/health
```

### 2-2. Predict (multipart)
```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "X-User-Name: demo-user" \
  -F "file=@./sample.jpg"
```

## 3) 환경변수
`inference.py` 기준 주요 옵션:

```env
# 모델 선택: simplecnn | efficientnet | vit | auto
MODEL_BACKBONE=simplecnn

# auto 선택 시 사용
AUTO_SELECT_MODEL=false

# 시작 시 sanity-check 실패 시 중단 여부
STRICT_SANITY_CHECK=false

# 분류 임계값
PRED_THRESHOLD=0.5
SIMPLECNN_THRESHOLD=0.32

# Unknown(비반려동물/불확실) 거절
REJECT_UNKNOWN=true
STRICT_PET_GUARD=true
PET_GUARD_MODE=balanced
SIMPLECNN_UNCERTAIN_BAND=0.08
BINARY_UNCERTAIN_BAND=0.10
```

## 4) 로그
- 파일: `logs/upload_audit.log`
- 일별 회전: `logs/upload_audit.log.YYYY-MM-DD`
- 콘솔/파일 동일 JSON line 포맷
- 기록 필드 예: `ts`, `ts_iso`, `request_id`, `actor`, `client_ip`, `file_name`, `status`, `reason`, `prediction`, `confidence`, `latency_ms`

실시간 확인 (PowerShell):
```powershell
Get-Content logs/upload_audit.log -Wait
```

## 5) 체크포인트
- `GET /health` 응답 정상
- `POST /predict` 성공/실패 포맷 계약대로 동작
- 잘못된 입력(빈 파일/손상/타입/용량) 검증 동작
- Unknown 거절(422) 동작
- 로그가 파일/콘솔에 동일 JSON으로 기록
