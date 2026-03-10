# Streamlit Cloud 배포 가이드

## 1단계: 저장소 확인

GitHub 저장소: https://github.com/qkek033/Dogs_Cats_Image_Classification

다음 파일들이 있는지 확인하세요:
- ✅ `app/streamlit_app.py` (메인 앱)
- ✅ `app/models/inference.py` (모델 로드)
- ✅ `requirements.txt` (의존성)
- ✅ `.streamlit/config.toml` (설정)

---

## 2단계: Streamlit Cloud 연결

1. https://streamlit.io/cloud 접속
2. **Sign in with GitHub** 클릭
3. GitHub 계정으로 로그인
4. **Create app** 클릭

---

## 3단계: 앱 배포 설정

배포 설정:
- **Repository**: qkek033/Dogs_Cats_Image_Classification
- **Branch**: main
- **Main file path**: app/streamlit_app.py

---

## 4단계: 배포 후 확인

배포가 완료되면:
1. 자동으로 모델이 HuggingFace Hub에서 다운로드됨 (첫 실행시 5-10분)
2. 이미지 업로드 후 예측 기능 테스트

---

## 주의사항

⚠️ **HuggingFace Hub 모델이 public이어야 합니다**
- https://huggingface.co/qkek033/Dogs_Cats_Image_Classification 확인

✅ **모델 첫 다운로드는 시간이 걸립니다**
- 약 300MB 크기
- 이후 캐시에서 빠르게 로드

---

## 배포 후 이슈 해결

### 모델 다운로드 실패

```
FileNotFoundError: models not found
```

해결책:
1. HuggingFace Hub 저장소 public 확인
2. `model.pth` 파일 존재 확인
3. Streamlit Cloud 재배포

### CUDA 관련 경고

CPU에서도 정상 작동합니다. 무시해도 됩니다.

---

## 로컬 테스트

배포 전에 로컬에서 테스트하려면:

```bash
cd c:\Users\Hyeonseong\Desktop\프로젝트모음\dogs_cats
streamlit run app/streamlit_app.py
```

---

## 추가 정보

- Streamlit 공식 문서: https://docs.streamlit.io/
- 배포 가이드: https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app
