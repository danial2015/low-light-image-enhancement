import os
import streamlit as st
import time
import numpy as np
import pandas as pd
import PIL.Image
import cv2
from streamlit_image_comparison import image_comparison
from enlighten_inference import EnlightenOnnxModel
from zero_dce import Trainer
from metrics.niqe_score import Niqe_score
from metrics.brisque_score import Brisque_score

@st.cache_resource
def load_zero_dce_model():
    checkpoint_path = os.path.join("zero_dce_pretrained_models", "model200_dark_faces.pth")
    trainer = Trainer()
    trainer.build_model(pretrain_weights=checkpoint_path)
    return trainer

def process_enlighten_gan(img_array):
    model = EnlightenOnnxModel(providers=["CPUExecutionProvider"])
    output = model.predict(img_array)
    return np.clip(output, 0, 255).astype(np.uint8)

def process_zero_dce(img_array):
    trainer = load_zero_dce_model()

    # Safe input conversion
    if img_array.dtype != np.uint8:
        img_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)

    img_pil = PIL.Image.fromarray(img_array).convert("RGB").resize((512, 512))
    from io import BytesIO
    img_bytes = BytesIO()
    img_pil.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    _, enhanced_img = trainer.infer_cpu(img_bytes, image_resize_factor=1)

    # Handle float32 output
    if enhanced_img.dtype != np.uint8:
        enhanced_img = (np.clip(enhanced_img, 0, 1) * 255).astype(np.uint8)

    enhanced_img = cv2.resize(enhanced_img, (img_array.shape[1], img_array.shape[0]))

    return np.clip(enhanced_img, 0, 255).astype(np.uint8)

def process_clahe(img_array):
    img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    img_clahe = cv2.merge((cl, a, b))
    return cv2.cvtColor(img_clahe, cv2.COLOR_LAB2RGB)

def SSR(img, sigma):
    img = np.maximum(img, 1e-6)
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    blur = np.maximum(blur, 1e-6)
    retinex = np.log1p(img) - np.log1p(blur)
    return retinex

def MSR(img, sigma_list):
    result = np.zeros_like(img)
    for c in range(img.shape[2]):
        channel = img[:, :, c]
        msr_channel = sum(SSR(channel, sigma) for sigma in sigma_list) / len(sigma_list)
        result[:, :, c] = msr_channel
    return result

def color_restoration(img, alpha=125, beta=46):
    img_sum = np.sum(img, axis=2, keepdims=True)
    return beta * (np.log1p(alpha * img) - np.log1p(img_sum))

def MSRCR(img, sigma_list, G=5, b=25, alpha=125, beta=46, low_clip=0.01, high_clip=0.99):
    img = img.astype(np.float32) + 1.0
    retinex = MSR(img, sigma_list)
    color_rest = color_restoration(img, alpha, beta)
    msrcr = G * (retinex * color_rest + b)
    for i in range(msrcr.shape[2]):
        channel = msrcr[:, :, i]
        channel = np.clip(channel, np.percentile(channel, low_clip * 100), np.percentile(channel, high_clip * 100))
        channel = (channel - np.min(channel)) / (np.max(channel) - np.min(channel)) * 255
        msrcr[:, :, i] = channel
    return np.clip(msrcr, 0, 255).astype(np.uint8)

def MSRCP(img, sigma_list, low_clip=0.01, high_clip=0.99):
    img = img.astype(np.float32) + 1.0
    intensity = np.mean(img, axis=2)
    retinex = MSR(np.expand_dims(intensity, axis=2), sigma_list)
    retinex = np.squeeze(retinex)
    retinex = np.clip(retinex, np.percentile(retinex, low_clip * 100), np.percentile(retinex, high_clip * 100))
    retinex = (retinex - np.min(retinex)) / (np.max(retinex) - np.min(retinex)) * 255
    img_cp = np.zeros_like(img)
    for i in range(3):
        img_cp[:, :, i] = retinex * (img[:, :, i] / intensity)
    return np.clip(img_cp, 0, 255).astype(np.uint8)

def process_retinex(img_array, mode='MSRCR'):
    sigma_list = [15, 80, 250]
    img = img_array.astype(np.float32) / 255.0 + 1e-6  # avoid log(0)

    if mode == 'SSR':
        result = np.zeros_like(img_array.astype(np.float32))
        img = img_array.astype(np.float32) + 1.0
        for i in range(3):
            result[:, :, i] = SSR(img[:, :, i], sigma=80)
        result = (result - np.min(result)) / (np.max(result) - np.min(result)) * 255
        return np.clip(result, 0, 255).astype(np.uint8)

    elif mode == 'MSR':
        img = img_array.astype(np.float32) + 1.0
        result = MSR(img, sigma_list)
        result = (result - np.min(result)) / (np.max(result) - np.min(result)) * 255
        return np.clip(result, 0, 255).astype(np.uint8)

    elif mode == 'MSRCR':
        return MSRCR(img_array.astype(np.float32) / 255.0 + 1e-6, sigma_list)

    elif mode == 'MSRCP':
        return MSRCP(img_array.astype(np.float32) / 255.0 + 1e-6, sigma_list)

def convert_to_bytes(img):
    img_pil = PIL.Image.fromarray(img)
    from io import BytesIO
    buf = BytesIO()
    img_pil.save(buf, format="PNG")
    return buf.getvalue()

st.set_page_config(page_title="Low Light Enhancement Ultimate", page_icon="📸", layout="wide")

# Custom CSS for elegant styling
st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton > button {
        border-radius: 10px;
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        font-size: 16px;
        margin: 10px 0;
    }
    .stDownloadButton > button {
        border-radius: 10px;
        background-color: #2196F3;
        color: white;
        padding: 8px 16px;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)
st.markdown("## 📸 Low-Light Enhancement Ultimate Dashboard")
st.markdown("Upload citra low-light untuk membandingkan hasil dari semua metode.")

with st.sidebar:
    st.header("🔧 Upload & Process")
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    start_button = st.button("🚀 Start Comparison")

if uploaded_file:
    uploaded_image_pil = PIL.Image.open(uploaded_file).convert("RGB")
    img_array_uploaded = np.array(uploaded_image_pil)
    st.image(uploaded_image_pil, caption="Original Image", use_column_width=True)

    if st.button("Start Comparison"):
        results = {}
        niqe = Niqe_score()
        brisque = Brisque_score()

        results['Original'] = {'image': img_array_uploaded, 'time': 0,
                               'niqe': niqe.count(img_array_uploaded),
                               'brisque': brisque.count(img_array_uploaded)}

        models = {
            'EnlightenGAN': lambda x: process_enlighten_gan(x),
            'Zero-DCE': lambda x: process_zero_dce(x),
            'CLAHE': lambda x: process_clahe(x),
            'SSR': lambda x: process_retinex(x, 'SSR'),
            'MSR': lambda x: process_retinex(x, 'MSR'),
            'MSRCR': lambda x: process_retinex(x, 'MSRCR'),
            'MSRCP': lambda x: process_retinex(x, 'MSRCP')
        }

        for name, func in models.items():
            with st.spinner(f"Running {name}..."):
                start = time.time()
                result = func(img_array_uploaded)
                t = time.time() - start
                results[name] = {'image': result, 'time': t,
                                 'niqe': niqe.count(result),
                                 'brisque': brisque.count(result)}

        st.subheader("🔎 Results")
        cols = st.columns(4)
        methods = list(results.keys())[1:]
        for idx, method in enumerate(methods):
            col = cols[idx % 4]
            img_pil = PIL.Image.fromarray(results[method]['image'])
            with col:
                st.image(img_pil, caption=f"{method}\n{results[method]['time']:.2f}s", use_column_width=True)
                st.download_button(label=f"Download {method}", data=convert_to_bytes(results[method]['image']),
                                   file_name=f"{method}.png", mime="image/png")

        st.markdown("---")
        st.subheader("📊 Quality Metrics")
        metrics_df = pd.DataFrame({
            "NIQE": [results[m]['niqe'] for m in results],
            "BRISQUE": [results[m]['brisque'] for m in results],
            "Time (s)": [results[m]['time'] for m in results]
        }, index=results.keys())
        st.dataframe(metrics_df)

        st.markdown("---")
        st.subheader("🖼️ Image Comparison Slider")
        for method in methods:
            image_comparison(img1=uploaded_image_pil,
                             img2=PIL.Image.fromarray(results[method]['image']),
                             label1="Original", label2=method, show_labels=True)
