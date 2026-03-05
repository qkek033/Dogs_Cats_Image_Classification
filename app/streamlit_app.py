import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image, UnidentifiedImageError

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Dogs vs Cats", layout="centered")

st.title("Dogs vs Cats Classifier")
st.write("Upload an image to classify dog vs cat.")
user_name = st.text_input("User name (for audit log)", value="anonymous")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg", "webp"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_container_width=True)
    except (UnidentifiedImageError, OSError):
        st.error("Cannot read this file as an image. Please upload a valid image file.")
        st.stop()

    if st.button("Predict"):
        try:
            with st.spinner("Analyzing..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                headers = {"X-User-Name": (user_name or "anonymous").strip() or "anonymous"}
                response = requests.post(API_URL, files=files, headers=headers, timeout=30)
        except requests.RequestException:
            st.error("Cannot connect to API server. Check if FastAPI is running.")
            st.stop()

        if response.status_code == 200:
            result = response.json()

            label = result["label"]
            confidence = result["confidence"]
            label_ko = "강아지" if label == "dog" else "고양이" if label == "cat" else "알 수 없음"

            st.success(f"결과: {label_ko}입니다.")
            st.write(f"Confidence: {confidence:.3f}")
            st.caption(f"Request ID: {result.get('request_id', '-')}")

            cam = np.array(result["cam"])
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            st.image(heatmap, caption="Grad-CAM", use_container_width=True)
        else:
            detail = "Request failed."
            try:
                payload = response.json()
                detail = payload.get("detail", detail)
            except ValueError:
                pass
            st.error(f"Error ({response.status_code}): {detail}")
