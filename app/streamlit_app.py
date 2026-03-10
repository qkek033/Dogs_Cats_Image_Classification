import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
import torch

sys.path.insert(0, str(Path(__file__).parent))
from models.inference import predict_image

st.set_page_config(
    page_title="Dogs vs Cats Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Dogs vs Cats Classifier")
st.write("""
강아지와 고양이를 구분하는 AI 모델입니다.
이미지를 업로드하고 Predict 버튼을 클릭하세요!
""")

with st.sidebar:
    st.markdown("### 정보")
    st.markdown("""
    - **모델**: SimpleCNN
    - **GPU**: CUDA 사용 가능시 GPU에서 실행
    """)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.markdown(f"**현재 장치**: {device}")

uploaded_file = st.file_uploader("이미지 파일 선택", type=["jpg", "png", "jpeg", "webp"])

if uploaded_file is not None:
    st.markdown("---")
    
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="업로드된 이미지", width=400)
    except (UnidentifiedImageError, OSError):
        st.error("이미지를 읽을 수 없습니다. 유효한 이미지 파일을 업로드하세요.")
        st.stop()
    
    if st.button("Predict", type="primary"):
        with st.spinner("분석 중..."):
            try:
                image_bytes = uploaded_file.getvalue()
                label, confidence, cam, rejected, reject_reason = predict_image(image_bytes)
                
                label_ko = "강아지" if label == "dog" else "고양이"
                
                st.markdown("---")
                st.markdown("### 결과")
                st.success(f"**{label_ko}**")
                
                progress_bar = st.progress(0)
                for percent_complete in range(int(confidence * 100) + 1):
                    progress_bar.progress(percent_complete / 100)
                
                st.metric("신뢰도", f"{confidence:.1%}")
                
                if rejected:
                    st.warning(f"신뢰도가 낮습니다 ({reject_reason})")
                
                if cam is not None:
                    cam_pil = Image.fromarray((255 * cam).astype('uint8'), mode='L')
                    st.image(cam_pil, caption="모델 주목 영역 (Grad-CAM)", width=400)
                
            except Exception as e:
                st.error(f"예측 실패: {str(e)}")
                st.info("팁: 모델이 처음 실행될 때 HuggingFace Hub에서 다운로드됩니다. 시간이 걸릴 수 있습니다.")
