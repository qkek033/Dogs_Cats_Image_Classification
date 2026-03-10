"""
HuggingFace Hub에 모델 업로드 스크립트
"""
from huggingface_hub import HfApi, HfFolder
from pathlib import Path

# HuggingFace 토큰 설정 (먼저 huggingface-cli login 실행)
# 또는 환경변수 HF_TOKEN 설정

api = HfApi()

# 업로드할 모델 파일
model_file = Path("models/model.pth")

if not model_file.exists():
    print(f" 모델 파일을 찾을 수 없습니다: {model_file}")
    exit(1)

repo_id = "qkek033/Dogs_Cats_Image_Classification"

print(f"🚀 업로드 시작: {model_file} -> {repo_id}")

try:
    api.upload_file(
        path_or_fileobj=str(model_file),
        path_in_repo="model.pth",
        repo_id=repo_id,
        repo_type="model",
    )
    print(f" 업로드 완료!")
except Exception as e:
    print(f" 업로드 실패: {e}")
