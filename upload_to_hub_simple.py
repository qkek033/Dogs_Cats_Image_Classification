"""
HuggingFace Hub에 모델 업로드
환경변수 HF_TOKEN을 사용합니다.
"""
import os
from pathlib import Path
from huggingface_hub import upload_file

# 환경변수에서 토큰 읽기
token = os.getenv('HF_TOKEN')

if not token:
    print("❌ HF_TOKEN 환경변수가 설정되지 않았습니다.")
    print("다음 명령으로 설정하세요:")
    print('$env:HF_TOKEN="your_token_here"')
    exit(1)

# 모델 파일
model_file = Path("models/model.pth")

if not model_file.exists():
    print(f"❌ 모델 파일을 찾을 수 없습니다: {model_file}")
    exit(1)

repo_id = "qkek033/Dogs_Cats_Image_Classification"

print(f"📤 업로드 시작...")
print(f"   파일: {model_file}")
print(f"   크기: {model_file.stat().st_size / (1024*1024):.2f} MB")
print(f"   저장소: {repo_id}")

try:
    result = upload_file(
        path_or_fileobj=str(model_file),
        path_in_repo="model.pth",
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )
    print(f"✅ 업로드 완료!")
    print(f"   URL: {result}")
except Exception as e:
    print(f"❌ 업로드 실패: {e}")
    exit(1)
