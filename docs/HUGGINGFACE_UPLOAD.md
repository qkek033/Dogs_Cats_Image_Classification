# HuggingFace Hub에 모델 업로드 가이드

## 1단계: HuggingFace 계정으로 로그인

PowerShell에서:
```bash
huggingface-cli login
```

계정 정보를 입력하세요. (토큰은 https://huggingface.co/settings/tokens 에서 생성 가능)

## 2단계: 저장소 생성 (필요한 경우)

```bash
huggingface-cli repo create Dogs_Cats_Image_Classification --type model
```

## 3단계: 모델 업로드

### 방법 A: Python 스크립트 사용 (추천)

```bash
cd c:\Users\Hyeonseong\Desktop\프로젝트모음\dogs_cats
python scripts/upload_model_to_hub.py
```

### 방법 B: 직접 업로드

```bash
huggingface-cli upload qkek033/Dogs_Cats_Image_Classification models/model.pth model.pth
```

### 방법 C: git-lfs 사용

```bash
cd models
git lfs install
git lfs track "*.pth"
git add .gitattributes model.pth
git commit -m "Add model file"
git push origin main
```

## 업로드 확인

https://huggingface.co/qkek033/Dogs_Cats_Image_Classification 에서 확인하세요.

## 환경변수 설정 (선택사항)

PowerShell에서:
```bash
$env:HF_TOKEN='your_token_here'
```

또는 시스템 환경변수에 `HF_TOKEN` 추가
