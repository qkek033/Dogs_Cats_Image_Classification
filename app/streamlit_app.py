import streamlit as st
import requests
from PIL import Image
import numpy as np
import cv2

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Dogs vs Cats", layout="centered")

st.title("🐶🐱 Dogs vs Cats Classifier")
st.write("이미지를 업로드하면 개/고양이를 분류합니다.")

uploaded_file = st.file_uploader("이미지 업로드", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="업로드 이미지", use_column_width=True)

    if st.button("예측하기"):
        with st.spinner("분석 중..."):
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            result = response.json()

            label = result["label"]
            confidence = result["confidence"]

            st.success(f"결과: {label}")
            st.write(f"신뢰도: {confidence:.3f}")

            # Grad-CAM 시각화
            cam = np.array(result["cam"])

            heatmap = cv2.applyColorMap(
                np.uint8(255 * cam),
                cv2.COLORMAP_JET
            )

            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            st.image(heatmap, caption=" Grad-CAM")

        else:
            st.error("API 요청 실패")