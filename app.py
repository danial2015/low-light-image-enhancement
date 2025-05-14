import os
import streamlit as st
import time
import numpy as np
import pandas as pd
import PIL.Image
import cv2
import matplotlib.pyplot as plt
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
    if img_array.dtype != np.uint8:
        img_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
    img_pil = PIL.Image.fromarray(img_array).convert("RGB").resize((512, 512))
    from io import BytesIO
    img_bytes = BytesIO()
    img_pil.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    _, enhanced_img = trainer.infer_cpu(img_bytes, image_resize_factor=1)
    if enhanced_img.dtype != np.uint8:
        enhanced_img = (np.clip(enhanced_img, 0, 1) * 255).astype(np.uint8)
    enhanced_img = cv2.resize(enhanced_img, (img_array.shape[1], img_array.shape[0]))
    return np.clip(enhanced_img, 0, 255).astype(np.uint8)

def process_clahe(img_array):
    img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    img_clahe = cv2.merge((cl, a, b))
    return cv2.cvtColor(img_clahe, cv2.COLOR_LAB2RGB)

def SSR(img, sigma):
    img = np.maximum(img, 1e-6)
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    blur = np.maximum(blur, 1e-6)
    return np.log1p(img) - np.log1p(blur)

def MSR(img, sigma_list):
    result = np.zeros_like(img)
    for c in range(img.shape[2]):
        channel = img[:, :, c]
        result[:, :, c] = sum(SSR(channel, sigma) for sigma in sigma_list) / len(sigma_list)
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

def plot_bar(data, title, ylabel):
    fig, ax = plt.subplots()
    bars = ax.bar(data.index, data.values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(data.index, rotation=45, ha="right")
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    st.pyplot(fig)

# ================= STREAMLIT APP =================
st.set_page_config(page_title="Low Light Enhancement", page_icon="📸", layout="wide")
st.title("📸 Low-Light Enhancement Dashboard")
st.divider()

with st.sidebar:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Upload low-light image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    uploaded_image_pil = PIL.Image.open(uploaded_file).convert("RGB")
    img_array_uploaded = np.array(uploaded_image_pil)

    st.subheader("🎯 Preview Uploaded Image")
    st.image(uploaded_image_pil, caption="Original Image", use_container_width=True)
    st.divider()

    if st.button("🚀 Run Enhancement Comparison"):
        niqe = Niqe_score()
        brisque = Brisque_score()
        results = {}
        results['Original'] = {'image': img_array_uploaded, 'time': 0,
                               'niqe': niqe.count(img_array_uploaded),
                               'brisque': brisque.count(img_array_uploaded)}
        models = {
            'EnlightenGAN': process_enlighten_gan,
            'Zero-DCE': process_zero_dce,
            'CLAHE': process_clahe,
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

        st.subheader("🔎 Enhanced Images")
        cols = st.columns(4)
        methods = list(results.keys())[1:]
        for idx, method in enumerate(methods):
            col = cols[idx % 4]
            img_pil = PIL.Image.fromarray(results[method]['image'])
            with col:
                st.image(img_pil, caption=f"{method}\n{results[method]['time']:.2f}s", use_container_width=True)
                st.download_button(f"Download {method}", data=convert_to_bytes(results[method]['image']),
                                   file_name=f"{method}.png", mime="image/png")

        st.divider()
        metrics_df = pd.DataFrame({
            "NIQE": [results[m]['niqe'] for m in results],
            "BRISQUE": [results[m]['brisque'] for m in results],
            "Time (s)": [results[m]['time'] for m in results]
        }, index=results.keys())
        metrics_df.loc['Original', 'Time (s)'] = np.nan

        st.subheader("📊 Quality Metrics Table")

        def highlight_best_worst(s):
            s_wo_original = s.drop(labels='Original', errors='ignore')
            is_best = s == s_wo_original.min()
            is_worst = s == s_wo_original.max()
            return ['background-color: lightgreen' if b else 
                    'background-color: salmon' if w else '' 
                    for b, w in zip(is_best, is_worst)]

        styled_df = metrics_df.style.apply(highlight_best_worst, subset=['NIQE'])
        styled_df = styled_df.apply(highlight_best_worst, subset=['BRISQUE'])
        styled_df = styled_df.apply(highlight_best_worst, subset=['Time (s)'])
        st.dataframe(styled_df)

        st.subheader("📈 Quality Metrics Bar Charts")
        metrics_plot = metrics_df.drop(index='Original')
        plot_bar(metrics_plot["NIQE"], "NIQE Score per Method", "NIQE")
        plot_bar(metrics_plot["BRISQUE"], "BRISQUE Score per Method", "BRISQUE")
        plot_bar(metrics_plot["Time (s)"], "Processing Time per Method", "Time (s)")

        
        st.divider()
        st.subheader("🖼️ Image Comparison Sliders")
        for method in methods:
            image_comparison(img1=uploaded_image_pil,
                             img2=PIL.Image.fromarray(results[method]['image']),
                             label1="Original", label2=method, show_labels=True)
