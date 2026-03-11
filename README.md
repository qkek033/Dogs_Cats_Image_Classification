# Dogs vs Cats Classifier

Streamlit으로 배포된 강아지/고양이 이미지 분류 AI 모델입니다.

**배포 링크**: https://dogscatsimageclassification-tc8hscjvvx7xrtfy7wtbcy.streamlit.app/

## 특징

- **SimpleCNN 모델**: 가벼우면서도 정확한 CNN 기반 분류 모델
- **HuggingFace Hub 통합**: 모델을 클라우드에서 자동으로 다운로드
- **Grad-CAM 시각화**: 모델이 주목한 이미지 영역을 시각화로 표시
- **GPU 지원**: CUDA가 설치되어 있으면 GPU에서 실행
- **Streamlit 배포**: 클라우드 환경에서 즉시 사용 가능

## 빠른 시작 (로컬)

### 1. 환경 설정

```powershell
# 가상환경 생성
python -m venv .venv

# 가상환경 활성화
.\.venv\Scripts\Activate.ps1

# 의존성 설치
pip install -r requirements.txt
```

### 2. Streamlit 실행

```powershell
streamlit run app/streamlit_app.py
```

그 후 브라우저에서 `http://localhost:8501` 접속

## 모델 재학습 (선택사항)

### 1. 데이터 준비

1. [Kaggle Dogs vs Cats Redux](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition) 데이터 다운로드
2. 다음과 같이 정렬:
```
data/train/
├── cats/
│   ├── cat.1.jpg
│   ├── cat.2.jpg
│   └── ...
└── dogs/
    ├── dog.1.jpg
    ├── dog.2.jpg
    └── ...
```

### 2. 모델 학습

```powershell
python train_model.py
```

학습 완료 후 `models/model.pth` 생성됨

### 3. HuggingFace Hub에 업로드

```powershell
# 토큰 설정 (Windows PowerShell)
$env:HF_TOKEN = 'your_huggingface_token'

# 업로드
python upload_to_hub.py
```

## 사용 방법

1. **이미지 선택**: "이미지 파일 선택" 버튼으로 jpg/png/jpeg/webp 파일 업로드
2. **분석 실행**: "Predict" 버튼 클릭
3. **결과 확인**: 분류 결과(강아지/고양이)와 신뢰도(0~100%) 확인
4. **Grad-CAM**: 모델이 주목한 이미지 영역을 히트맵으로 표시

## 기술 스택

- **Framework**: Streamlit (Web UI)
- **Model**: PyTorch SimpleCNN
- **Model Storage**: HuggingFace Hub
- **Visualization**: Grad-CAM, PIL
- **Deployment**: Streamlit Cloud

## 프로젝트 구조

```
dogs_cats/
├── app/
│   ├── streamlit_app.py    # Streamlit 메인 애플리케이션
│   └── models/
│       └── inference.py    # 모델 로딩, 예측, Grad-CAM
├── docs/                    # 문서
├── notebooks/              # 학습 및 분석 노트북
├── requirements.txt        # Python 의존성
├── README.md              # 이 파일
└── .gitignore            # Git 제외 파일
```

## 요구사항

- Python 3.8 이상
- PyTorch
- Streamlit
- 기타 의존성: `requirements.txt` 참고

## 모델 정보

- **모델명**: SimpleCNN
- **입력 크기**: 128x128 RGB 이미지
- **출력**: 강아지 또는 고양이 분류
- **가중치 저장소**: [HuggingFace Hub](https://huggingface.co/qkek033/Dogs_Cats_Image_Classification)

## 참고 사항

- 모델이 처음 실행될 때 HuggingFace Hub에서 다운로드됩니다 (시간 소요)
- 신뢰도가 50% 미만인 경우 예측 결과에 경고 표시
- GPU가 없으면 CPU에서 실행됩니다 (속도 저하)

## 라이선스

MIT License
