import os
import streamlit as st
import time
import PIL.Image
import numpy as np
import cv2
from streamlit_image_comparison import image_comparison
import pandas as pd
from zero_dce import Trainer
from metrics.niqe_score import Niqe_score
from metrics.brisque_score import Brisque_score

# CLAHE function
def apply_clahe(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    clahe_img = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)
    return clahe_img

# Retinex function
def apply_retinex(img):
    img = np.array(img, dtype=np.float32) + 1.0
    img_log = np.log(img)
    img_retinex = np.exp(img_log) - 1.0
    img_retinex = np.clip(img_retinex, 0, 255).astype(np.uint8)
    return img_retinex

@st.cache_resource
def load_model():
    checkpoint_path = os.path.join("zero_dce_pretrained_models", "model200_dark_faces.pth")

    trainer = Trainer()
    trainer.build_model(pretrain_weights=checkpoint_path)

    return trainer

st.markdown("### Low Light Image Enhancement (Dashcam Image) | Zero DCE Method")

uploaded_image = st.file_uploader(label="Upload Low Light Image", type=['jpg', 'png'])
if uploaded_image is not None:
    uploaded_image_pil = PIL.Image.open(uploaded_image)
    img_array_uploaded = np.array(uploaded_image_pil)

    # Sidebar for extra processing options
    st.sidebar.header("Enhancement Options")
    enhance_clahe = st.sidebar.checkbox("Apply CLAHE (Histogram Equalization)", value=False)
    enhance_retinex = st.sidebar.checkbox("Apply Retinex (Color Enhancement)", value=False)

    with st.spinner("Processing...", show_time=True):
        start_time = time.time()

        # Apply CLAHE if selected
        if enhance_clahe:
            img_array_uploaded = apply_clahe(img_array_uploaded)
        
        # Apply Retinex if selected
        if enhance_retinex:
            img_array_uploaded = apply_retinex(img_array_uploaded)

        # Load model and enhance
        trainer = load_model()
        enhanced_img = trainer.infer_cpu(uploaded_image, image_resize_factor=1)

        # Evaluate with NIQE and BRISQUE
        niqe = Niqe_score()
        niqe_before = niqe.count(np.array(uploaded_image_pil))
        niqe_after = niqe.count(enhanced_img)

        brisque = Brisque_score()
        brisque_before = brisque.count(np.array(uploaded_image_pil))
        brisque_after = brisque.count(enhanced_img)

        # Display images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_image_pil, caption="Original Image (Low Light)")
        with col2:
            st.image(enhanced_img, caption="Enhanced Image")

        # Execution time
        end_time = time.time()
        execution_time = end_time - start_time
        st.write(f"Inference Time: {execution_time:.7f} seconds")

        # Metrics table
        metrics_data = pd.DataFrame({
            "Metrics": ["NIQE", "BRISQUE"],
            "Before": [niqe_before, brisque_before],
            "After": [niqe_after, brisque_after]
        })
        st.dataframe(metrics_data.set_index(metrics_data.columns[0]))

    # Interactive Comparison Slider
    st.subheader("Interactive Image Comparison")
    image_comparison(
        img1=uploaded_image_pil,
        img2=enhanced_img,
        label1="Original Image",
        label2="Enhanced Image",
    )

    # Provide option to download the enhanced image
    st.download_button(
        label="Download Enhanced Image",
        data=enhanced_img,
        file_name="enhanced_image.jpg",
        mime="image/jpeg"
    )
