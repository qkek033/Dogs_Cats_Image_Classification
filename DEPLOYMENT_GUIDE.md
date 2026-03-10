# 배포 완료 가이드

## 현재까지 완료된 것

[완료] 모델 준비
- EfficientNet-B7 모델 (models/model.pth)
- HuggingFace Hub 통합 코드 완성

[완료] Git 정리
- 큰 파일 GitHub에서 제거
- 배포용 코드 준비 완료

[완료] Streamlit 앱 준비
- 아름다운 UI 구현
- 한글 지원
- Grad-CAM 시각화
- GPU/CPU 자동 감지

[완료] GitHub 푸시
- 모든 코드 저장소에 업로드

---

## 다음 해야 할 것 (3단계)

### 단계 1: HuggingFace Hub에 모델 업로드

PowerShell에서:

```bash
cd "c:\Users\Hyeonseong\Desktop\프로젝트모음\dogs_cats"

# 토큰 환경변수 설정
$env:HF_TOKEN='your_token_here'

# 모델 업로드
python upload_to_hub_simple.py
```

**토큰 생성 방법:**
1. https://huggingface.co/settings/tokens 접속
2. Create new token -> Name: `dogs_cats_model` -> Role: `write`
3. 토큰 복사

---

### 단계 2: Streamlit Cloud에 배포

1. https://streamlit.io/cloud 접속
2. **Sign in with GitHub**
3. **Create app** -> 다음 설정:
   - Repository: `qkek033/Dogs_Cats_Image_Classification`
   - Branch: `main`
   - Main file path: `app/streamlit_app.py`
4. **Deploy** 클릭

---

### 단계 3: 배포 확인

배포 후:
1. 모델 자동 다운로드 (5-10분)
2. 이미지 업로드 테스트
3. 예측 결과 확인

---

## 필요한 토큰 정보

### HuggingFace Token
- 목적: 모델 업로드
- 생성: https://huggingface.co/settings/tokens
- Role: `write`
- 형식: `hf_` 로 시작

### GitHub Token (이미 설정됨)
- Streamlit Cloud가 자동으로 사용

---

## 문제 해결

### 모델 업로드 실패

```
HF_TOKEN 환경변수가 설정되지 않았습니다
```

**해결책:**
```bash
$env:HF_TOKEN='your_token_here'
```

### Streamlit 앱 에러

```
FileNotFoundError: models not found
```

**해결책:**
1. HuggingFace Hub 저장소 확인
2. `model.pth` 파일 존재 확인
3. 저장소가 public인지 확인

---

## 배포 후 URL

배포 완료 후 앱 URL:
```
https://share.streamlit.io/qkek033/Dogs_Cats_Image_Classification/main/app/streamlit_app.py
```

---

## 로컬 테스트 (선택사항)

배포 전 로컬에서 테스트:

```bash
cd "c:\Users\Hyeonseong\Desktop\프로젝트모음\dogs_cats"
streamlit run app/streamlit_app.py
```

http://localhost:8501 에서 확인

---

준비 완료! 이제 HuggingFace에 모델 업로드하고 Streamlit Cloud에 배포하면 됩니다! 
