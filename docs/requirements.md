# 요구사항 명세서 (Requirements Specification)

## 1. 목적
본 프로젝트는 강아지/고양이 이미지 분류를 위한 FastAPI + Streamlit 기반 추론 서비스를 제공한다.
업로드 이미지에 대해 분류 결과와 CAM(시각화)을 반환하며, 잘못된 입력과 비반려동물 이미지를 거절하고 요청 감사 로그(JSON line)를 기록한다.

## 2. 범위
- API 엔드포인트: `GET /health`, `POST /predict`
- Streamlit 웹 UI를 통한 파일 업로드/예측
- 입력 검증(파일 타입/용량/손상 여부)
- Unknown(비반려동물) 거절 로직
- 요청 단위 `request_id`/`latency_ms` 반환
- 구조화 감사 로그(JSON line) + 일별 로그 회전
- 모델 백본 선택(`simplecnn`/`efficientnet`/`vit`/`auto`)

## 3. 기능 요구사항 (Functional Requirements)
### FR-01 헬스 체크
- `GET /health`는 `200 OK`를 반환해야 한다.
- 응답에는 `status`, `model_loaded`가 포함되어야 한다.

### FR-02 예측 API
- `POST /predict`는 이미지 파일(`multipart/form-data`)을 입력받는다.
- 정상 응답에는 `label`, `confidence`, `cam`, `request_id`, `latency_ms`를 포함해야 한다.

### FR-03 입력 검증
- 파일 누락 시 `400 No file uploaded.`
- 빈 파일 시 `400 Uploaded file is empty.`
- 비허용 MIME 타입 시 `415 Unsupported file type`
- 최대 업로드 용량 초과 시 `413 File too large`
- 손상/비이미지 파일 시 `400 Invalid or corrupted image file.`

### FR-04 비반려동물(Unknown) 거절
- PET Guard가 활성화된 경우, 비반려동물로 판단되는 이미지는 `422`로 거절해야 한다.
- 거절 메시지는 "Uploaded image does not look like a dog/cat photo."를 사용한다.

### FR-05 CAM 제공
- 예측 성공 시 `cam`(2D heatmap array)을 반환해야 한다.
- 모델 경로에 따라 Grad-CAM 적용 또는 안전한 대체 CAM을 반환해야 한다.

### FR-06 감사 로그
- 요청 결과(성공/실패/거절)를 JSON line으로 기록해야 한다.
- 로그 항목에는 `ts`, `ts_iso`, `request_id`, `actor`, `client_ip`, `file_name`, `content_type`, `status`, `reason`(실패 시), `prediction`/`confidence`(성공 시) 등이 포함되어야 한다.
- 로그는 파일과 콘솔에 동일 JSON 포맷으로 출력되어야 한다.

### FR-07 로그 회전
- 감사 로그는 일 단위로 회전되어야 한다.
- 기본 보관 기간은 최근 30일로 유지한다.

## 4. 비기능 요구사항 (Non-Functional Requirements)
### NFR-01 재현성
- 로컬 개발환경에서 동일한 예측 API 동작을 재현할 수 있어야 한다.

### NFR-02 관측성
- 요청 단위 `request_id`와 `latency_ms`를 제공해야 한다.
- 감사 로그를 통해 입력/결과/거절 사유를 추적할 수 있어야 한다.

### NFR-03 안정성
- 잘못된 파일 입력으로 서비스 프로세스가 비정상 종료되지 않아야 한다.

### NFR-04 단순성
- 실습 목적에 맞는 최소 구조를 유지한다(FastAPI + Streamlit + 단일 추론 경로).

### NFR-05 설정 가능성
- 모델 선택, 임계값, 가드 강도를 환경변수로 조정 가능해야 한다.

## 5. 환경변수 요구사항
| 변수명 | 기본값 | 설명 |
|---|---|---|
| `MODEL_BACKBONE` | `simplecnn` | 추론 모델 선택 (`simplecnn`, `efficientnet`, `vit`, `auto`) |
| `AUTO_SELECT_MODEL` | `false` | true 시 후보 백본 중 자동 선택 |
| `STRICT_SANITY_CHECK` | `false` | 시작 시 sanity-check 실패 시 서버 중단 여부 |
| `PRED_THRESHOLD` | `0.5` | EfficientNet/ViT 이진 분류 임계값 |
| `SIMPLECNN_THRESHOLD` | `0.32` | SimpleCNN dog 확률 임계값 |
| `REJECT_UNKNOWN` | `true` | 불확실/비반려동물 거절 로직 활성화 |
| `STRICT_PET_GUARD` | `true` | ImageNet 기반 PET Guard 활성화 |
| `PET_GUARD_MODE` | `balanced` | PET Guard 강도 (`lenient`, `balanced`, `strict`) |
| `SIMPLECNN_UNCERTAIN_BAND` | `0.08` | SimpleCNN 경계 근처 거절 폭 |
| `BINARY_UNCERTAIN_BAND` | `0.10` | EfficientNet/ViT 경계 근처 거절 폭 |

## 6. 수용 기준 (Acceptance Criteria)
- `GET /health`가 `200` 및 필수 필드(`status`, `model_loaded`)를 반환한다.
- `POST /predict`가 정상 이미지에 대해 계약된 응답 포맷을 반환한다.
- 잘못된 입력(빈 파일/손상 파일/비허용 타입/용량 초과)에 대해 계약된 상태코드로 실패한다.
- 비반려동물 이미지가 PET Guard 기준에서 거절되며 `422`를 반환한다.
- 감사 로그가 파일/콘솔 모두 동일 JSON 포맷으로 기록된다.
- 로그 파일이 일별로 회전되고(`upload_audit.log.YYYY-MM-DD`) 최근 보관 개수 정책이 적용된다.
