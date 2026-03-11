"""
학습된 모델을 HuggingFace Hub에 업로드하는 스크립트
"""

import os
from pathlib import Path
from huggingface_hub import HfApi

MODEL_PATH = "models/model.pth"
REPO_ID = "qkek033/Dogs_Cats_Image_Classification"
MODEL_NAME = "model.pth"

def upload_model():
    """모델 파일을 HuggingFace Hub에 업로드"""
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 모델 파일이 없습니다: {MODEL_PATH}")
        print("먼저 train_model.py를 실행하세요")
        return False
    
    print(f"📤 모델 업로드 시작: {MODEL_PATH}")
    
    try:
        api = HfApi()
        
        # 환경변수에서 토큰 가져오기
        token = os.getenv('HF_TOKEN')
        if not token:
            print("❌ HF_TOKEN 환경변수가 설정되지 않았습니다")
            print("\n다음 방법으로 토큰 설정:")
            print("  Windows PowerShell:")
            print("    $env:HF_TOKEN = 'your_token_here'")
            print("    python upload_to_hub.py")
            return False
        
        # 모델 업로드
        api.upload_file(
            path_or_fileobj=MODEL_PATH,
            path_in_repo=MODEL_NAME,
            repo_id=REPO_ID,
            token=token
        )
        
        print(f"✅ 모델 업로드 완료!")
        print(f"   저장소: {REPO_ID}")
        print(f"   파일: {MODEL_NAME}")
        return True
        
    except Exception as e:
        print(f"❌ 업로드 실패: {e}")
        return False

if __name__ == "__main__":
    upload_model()
