import os
import streamlit as st
import time
import PIL.Image
import numpy as np
from streamlit_image_comparison import image_comparison
import pandas as pd
from enlighten_inference import EnlightenOnnxModel
from metrics.niqe_score import Niqe_score
from metrics.brisque_score import Brisque_score

# Function to process the uploaded image with EnlightenOnnxModel
def process_enlighted_gan(uploaded_file):
    img_pil = PIL.Image.open(uploaded_file)
    img_array = np.array(img_pil)

    model = EnlightenOnnxModel(providers=["CPUExecutionProvider"])
    processed = model.predict(img_array)  # No need for transpose or batch dimension

    return processed

# Streamlit UI setup
st.markdown("### Low Light Image Enhancement (Dashcam Image) | EnlightenedGAN Method")

# Upload low light image
uploaded_image = st.file_uploader(label="Upload Low Light Image", type=['jpg', 'png'])
if uploaded_image is not None:
    uploaded_image_pil = PIL.Image.open(uploaded_image)
    img_array_uploaded = np.array(uploaded_image_pil)
    
    with st.spinner("Processing...", show_time=True):
        # Start processing time
        start_time = time.time()
        
        # Enhance image using EnlightenGAN
        enhanced_img = process_enlighted_gan(uploaded_image)
        enhanced_img_pil = PIL.Image.fromarray(enhanced_img)
        
        # Calculate inference time
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Calculate NIQE and BRISQUE scores
        niqe = Niqe_score()
        niqe_before = niqe.count(img_array_uploaded)
        niqe_after = niqe.count(enhanced_img)

        brisque = Brisque_score()
        brisque_before = brisque.count(img_array_uploaded)
        brisque_after = brisque.count(enhanced_img)

        # Display the images
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_image, caption="Original Image (Low Light)", use_column_width=True)
        with col2:
            st.image(enhanced_img_pil, caption="Enhanced Image", use_column_width=True)

        # Display inference time
        st.write(f"Inference Time: {execution_time:.3f} seconds")

        # Display metrics (NIQE and BRISQUE) in a table
        metrics_data = pd.DataFrame({
            "Metrics": ["NIQE", "BRISQUE"],
            "Before": [niqe_before, brisque_before],
            "After": [niqe_after, brisque_after]
        })
        st.dataframe(metrics_data.set_index(metrics_data.columns[0]))

        # Interactive image comparison
        st.subheader("Interactive Image Comparison")
        image_comparison(
            img1=uploaded_image_pil,
            img2=enhanced_img_pil,
            label1="Original Image",
            label2="Enhanced Image"
        )

        # Add a download button for the enhanced image
        st.download_button(
            label="Download Enhanced Image",
            data=enhanced_img_pil,
            file_name="enhanced_image.jpg",
            mime="image/jpeg"
        )
