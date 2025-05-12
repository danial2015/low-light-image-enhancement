import os
import time
import numpy as np
import cv2
import streamlit as st
import PIL.Image
import pandas as pd
from streamlit_image_comparison import image_comparison
from pathlib import Path
import torch
from skimage import exposure
import matplotlib.pyplot as plt

# Configuration
TEMP_IMAGE_PATH = "temp_input.jpg"
MAX_IMAGE_DIMENSION = 1024
SIGMA_LIST = [15, 80, 250]  # For Retinex

# --- Setup NIQE Metric ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
try:
    import pyiqa
    iqa_metric = pyiqa.create_metric('niqe').to(device)
except ImportError:
    iqa_metric = None
    st.warning("pyiqa not available, NIQE metrics will be simplified")

# --- Theme Configuration ---
def set_theme(light_mode=True):
    """Set light or dark theme for plots"""
    if light_mode:
        plt.style.use('default')
        bg_color = '#ffffff'
        text_color = '#000000'
    else:
        plt.style.use('dark_background')
        bg_color = '#0e1117'
        text_color = '#ffffff'
    plt.rcParams.update({
        'figure.facecolor': bg_color,
        'axes.facecolor': bg_color,
        'axes.edgecolor': text_color,
        'axes.labelcolor': text_color,
        'text.color': text_color,
        'xtick.color': text_color,
        'ytick.color': text_color
    })
    return bg_color, text_color

# --- CLAHE Enhancement ---
def apply_clahe(img, clip_limit=6.0, grid_size=(8,8)):
    """Apply CLAHE to LAB color space"""
    if not isinstance(img, np.ndarray):
        img = np.array(img).astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

# --- Retinex Functions ---
# --- Fungsi SSR ---
def SSR(img, sigma):
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    if img.ndim != blur.ndim:
        blur = np.expand_dims(blur, axis=-1)
    retinex = np.log1p(img) - np.log1p(blur)
    return retinex

# --- Fungsi MSR ---
def MSR(img, sigma_list):
    retinex = np.zeros_like(img)
    for sigma in sigma_list:
        retinex += SSR(img, sigma)
    retinex = retinex / len(sigma_list)
    return retinex

# --- Fungsi Color Restoration ---
def color_restoration(img, alpha=125, beta=46):
    img_sum = np.sum(img, axis=2, keepdims=True)
    color_rest = beta * (np.log1p(alpha * img) - np.log1p(img_sum))
    return color_rest

# --- Fungsi MSRCR ---
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
    msrcr = np.clip(msrcr, 0, 255).astype(np.uint8)
    return msrcr

# --- Fungsi MSRCP ---
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
    img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
    return img_cp

# --- Zero-DCE Implementation ---
@st.cache_resource
def load_zero_dce_model():
    """Load and cache Zero-DCE model with error handling"""
    try:
        from zero_dce import Trainer
        checkpoint_path = os.path.join("zero_dce_pretrained_models", "model200_dark_faces.pth")
        if not os.path.exists(checkpoint_path):
            st.error(f"Model file not found at: {checkpoint_path}")
            return None
        trainer = Trainer()
        trainer.build_model(pretrain_weights=checkpoint_path)
        return trainer
    except Exception as e:
        st.error(f"Failed to load Zero-DCE model: {str(e)}")
        return None

def process_zero_dce(img_array):
    """Process image with Zero-DCE with robust error handling"""
    try:
        if not isinstance(img_array, np.ndarray):
            img_array = np.array(img_array)
        # Validate input
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            st.error("Zero-DCE requires RGB color image")
            return img_array
        # Convert to BGR and ensure uint8 type
        img_bgr = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2BGR)
        # Resize if needed
        h, w = img_bgr.shape[:2]
        if max(h, w) > MAX_IMAGE_DIMENSION:
            scale = MAX_IMAGE_DIMENSION / max(h, w)
            img_bgr = cv2.resize(img_bgr, (int(w*scale), int(h*scale)))
        # Save temp file
        temp_path = Path(TEMP_IMAGE_PATH)
        cv2.imwrite(str(temp_path), img_bgr)
        # Process with Zero-DCE
        trainer = load_zero_dce_model()
        if trainer is None:
            return img_array
        result = trainer.infer_cpu(str(temp_path), image_resize_factor=1.0)
        # Handle different return types including PIL Image
        if result is None:
            st.warning("Zero-DCE returned None")
            return img_array
        if isinstance(result, PIL.Image.Image):
            # Convert PIL Image to numpy array
            enhanced = np.array(result)
        elif isinstance(result, (list, tuple)):
            enhanced = result[0] if len(result) > 0 else img_array
            if isinstance(enhanced, PIL.Image.Image):
                enhanced = np.array(enhanced)
        else:
            enhanced = result
        # Validate output format
        if not isinstance(enhanced, np.ndarray):
            st.warning(f"Zero-DCE returned unsupported type: {type(enhanced)}")
            return img_array
        # Ensure proper color format
        if len(enhanced.shape) == 2:  # Grayscale
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        elif enhanced.shape[2] == 4:  # RGBA
            enhanced = enhanced[:, :, :3]
        # Convert from BGR to RGB if needed
        if enhanced.shape[2] == 3:
            try:
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            except:
                pass  # Assume it's already RGB
        return enhanced.astype(np.uint8)
    except Exception as e:
        st.error(f"Zero-DCE processing error: {str(e)}")
        return img_array
    finally:
        if Path(TEMP_IMAGE_PATH).exists():
            Path(TEMP_IMAGE_PATH).unlink()

# --- EnlightenGAN Implementation ---
def load_enlighten_model():
    """Load EnlightenGAN model with error handling"""
    try:
        from enlighten_inference import EnlightenOnnxModel
        return EnlightenOnnxModel(providers=["CPUExecutionProvider"])
    except Exception as e:
        st.error(f"Failed to load EnlightenGAN model: {str(e)}")
        return None

def process_enlighten_gan(img_array):
    """Process image with EnlightenGAN with robust error handling"""
    try:
        if not isinstance(img_array, np.ndarray):
            img_array = np.array(img_array)
        # Validate input
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            st.error("EnlightenGAN requires RGB color image")
            return img_array
        model = load_enlighten_model()
        if model is None:
            return img_array
        result = model.predict(img_array.astype(np.uint8))
        # Handle different return formats
        if isinstance(result, (list, tuple)):
            return result[0] if len(result) > 0 else img_array
        return result
    except Exception as e:
        st.error(f"EnlightenGAN processing error: {str(e)}")
        return img_array

# --- Quality Metrics ---
def calculate_metrics(img_array):
    """Calculate NIQE and BRISQUE scores with error handling"""
    metrics = {'NIQE': 0, 'BRISQUE': 0}
    if not isinstance(img_array, np.ndarray):
        return metrics
    # BRISQUE calculation
    try:
        from metrics.brisque_score import Brisque_score
        if len(img_array.shape) == 3:
            gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray_img = img_array
        metrics['BRISQUE'] = Brisque_score().count(gray_img)
    except Exception as e:
        st.warning(f"BRISQUE calculation failed: {str(e)}")
    # NIQE calculation
    try:
        if iqa_metric and len(img_array.shape) == 3:
            img_tensor = torch.from_numpy(img_array).permute(2,0,1).unsqueeze(0).float()/255.0
            metrics['NIQE'] = iqa_metric(img_tensor.to(device)).item()
    except Exception as e:
        st.warning(f"NIQE calculation failed: {str(e)}")
    return metrics

# --- Display Results ---
def display_results(original, enhanced, original_metrics, enhanced_metrics, exec_time, light_mode=True):
    """Display comparison results for individual methods"""
    bg_color, text_color = set_theme(light_mode)
    col1, col2 = st.columns(2)
    with col1:
        st.image(original, caption="Original Image", use_container_width=True)
        st.metric("NIQE (Lower better)", 
                 f"{original_metrics['NIQE']:.2f}", 
                 f"{enhanced_metrics['NIQE']-original_metrics['NIQE']:.2f}")
        st.metric("BRISQUE (Lower better)", 
                 f"{original_metrics['BRISQUE']:.2f}", 
                 f"{enhanced_metrics['BRISQUE']-original_metrics['BRISQUE']:.2f}")
    with col2:
        st.image(enhanced, caption="Enhanced Image", use_container_width=True)
        st.metric("NIQE", f"{enhanced_metrics['NIQE']:.2f}")
        st.metric("BRISQUE", f"{enhanced_metrics['BRISQUE']:.2f}")
    st.caption(f"Processing time: {exec_time:.2f} seconds")
    st.subheader("Side-by-Side Comparison")
    image_comparison(
        img1=original,
        img2=enhanced,
        label1="Original",
        label2="Enhanced",
        width=700
    )

# --- Image Comparison Tab ---
def create_image_comparison_tab(results, light_mode=True):
    """Create tab with all enhanced images for comparison in 3x3 grid"""
    bg_color, text_color = set_theme(light_mode)
    st.header("All Results Comparison")
    
    # Add background toggle
    bg_toggle = st.toggle("Light Background", value=light_mode, key="bg_toggle_all_results")
    if bg_toggle != light_mode:
        light_mode = bg_toggle
        bg_color, text_color = set_theme(light_mode)

    if len(results) <= 1:
        st.warning("Process at least one enhancement method to see comparisons")
        return

    # Prepare the methods list (excluding original image)
    methods = [m for m in results.keys() if m != "Original"]
    num_methods = len(methods)
    cols_per_row = 3  # Display 3 columns per row

    # Group methods into rows of 3
    grouped_methods = [methods[i:i + cols_per_row] for i in range(0, num_methods, cols_per_row)]

    # Create a grid layout using Streamlit columns
    for row in grouped_methods:
        cols = st.columns(cols_per_row)
        for idx, col in enumerate(cols):
            if idx < len(row):
                method = row[idx]
                data = results[method]
                with col:
                    st.markdown(f"**{method}**")
                    st.image(data["image"], 
                            caption=f"NIQE: {data['metrics']['NIQE']:.2f}, BRISQUE: {data['metrics']['BRISQUE']:.2f}, Time: {data['time']:.2f}s",
                            use_container_width=True)
                    st.markdown("---")

    # Optional: Show metrics summary again or add additional controls
    st.subheader("Summary")
    st.write("You can scroll through the results and compare different enhancement methods visually.")

    # Add CSS for background styling
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-color: {bg_color};
                color: {text_color};
            }}
            .stImage > img {{
                border: 2px solid {text_color};
                border-radius: 10px;
            }}
            .css-1aumxhk {{
                background-color: {bg_color};
                color: {text_color};
            }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
# --- Metrics Comparison ---
def display_metrics_comparison(results, light_mode=True):
    """Display comprehensive metrics comparison across all methods"""
    bg_color, text_color = set_theme(light_mode)
    st.header("Metrics Comparison Across Models")
    # Create metrics dataframe
    metrics_data = []
    for method, data in results.items():
        if method == "Original":
            continue
        metrics_data.append({
            "Method": method,
            "NIQE": data["metrics"]["NIQE"],
            "BRISQUE": data["metrics"]["BRISQUE"],
            "Processing Time (s)": data["time"]
        })
    df = pd.DataFrame(metrics_data)
    # Display metrics table
    st.subheader("Numerical Metrics Comparison")
    st.dataframe(df.set_index("Method").style.format({
        "NIQE": "{:.2f}",
        "BRISQUE": "{:.2f}",
        "Processing Time (s)": "{:.3f}"
    }).highlight_min(axis=0, color='lightgreen'))
    # Visual comparison
    st.subheader("Visual Metrics Comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**NIQE Scores (Lower is better)**")
        fig1, ax1 = plt.subplots()
        ax1.bar(df["Method"], df["NIQE"], color='skyblue')
        ax1.set_xticklabels(df["Method"], rotation=45)
        ax1.set_ylabel("NIQE Score")
        st.pyplot(fig1)
    with col2:
        st.markdown("**BRISQUE Scores (Lower is better)**")
        fig2, ax2 = plt.subplots()
        ax2.bar(df["Method"], df["BRISQUE"], color='lightgreen')
        ax2.set_xticklabels(df["Method"], rotation=45)
        ax2.set_ylabel("BRISQUE Score")
        st.pyplot(fig2)
    st.markdown("**Processing Times**")
    fig3, ax3 = plt.subplots()
    ax3.bar(df["Method"], df["Processing Time (s)"], color='salmon')
    ax3.set_xticklabels(df["Method"], rotation=45)
    ax3.set_ylabel("Time (seconds)")
    st.pyplot(fig3)
    # Show best methods
    st.subheader("Best Performing Methods")
    best_niqe = df.loc[df["NIQE"].idxmin()]["Method"]
    best_brisque = df.loc[df["BRISQUE"].idxmin()]["Method"]
    fastest = df.loc[df["Processing Time (s)"].idxmin()]["Method"]
    st.markdown(f"""
    - **Best NIQE Score**: {best_niqe} ({df[df['Method'] == best_niqe]['NIQE'].values[0]:.2f})
    - **Best BRISQUE Score**: {best_brisque} ({df[df['Method'] == best_brisque]['BRISQUE'].values[0]:.2f})
    - **Fastest Processing**: {fastest} ({df[df['Method'] == fastest]['Processing Time (s)'].values[0]:.3f}s)
    """)

# --- Main Application ---
def main():
    # Set page config
    st.set_page_config(page_title="Low Light Image Enhancement", layout="wide")
    # Initialize session state for theme
    if 'light_mode' not in st.session_state:
        st.session_state.light_mode = True
    # Remove Streamlit menu and footer
    st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
    # Interactive theme toggle
    col1, col2 = st.columns([10, 1])
    with col2:
        st.session_state.light_mode = st.toggle("☀️", value=st.session_state.light_mode)
    # Apply theme
    bg_color, text_color = set_theme(st.session_state.light_mode)
    # Inject CSS dynamically based on theme
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-color: {bg_color};
                color: {text_color};
            }}
            h1, h2, h3, h4, h5, h6 {{
                color: {text_color};
            }}
            .stButton>button {{
                border-color: {text_color};
                color: {text_color};
                background-color: {'#f0f0f0' if st.session_state.light_mode else '#262730'};
            }}
            .stTextInput input {{
                color: {text_color};
                background-color: {'#f9f9f9' if st.session_state.light_mode else '#262730'};
                border-color: {text_color};
            }}
            .stSelectbox label {{
                color: {text_color};
            }}
            .stImage > img {{
                border: 2px solid {text_color};
                border-radius: 10px;
            }}
            /* Responsive design */
            @media (max-width: 768px) {{
                .css-1aumxhk {{
                    padding: 10px;
                }}
                .stMarkdown {{
                    font-size: 14px;
                }}
            }}
        </style>
        """,
        unsafe_allow_html=True
    )
    # App title
    st.title("📸 Low Light Image Enhancement")
    st.markdown("Compare CLAHE, Retinex, Zero-DCE, and EnlightenGAN methods")
    # File uploader
    uploaded_file = st.file_uploader("Upload low light image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            # Load and validate image
            with st.spinner("Loading image..."):
                img = PIL.Image.open(uploaded_file)
                img_array = np.array(img.convert('RGB')).astype(np.uint8)
                if len(img_array.shape) != 3 or img_array.shape[2] != 3:
                    st.error("Please upload a valid RGB color image")
                    return
                original_metrics = calculate_metrics(img_array)
            # Initialize results storage
            results = {
                "Original": {
                    "image": img_array,
                    "metrics": original_metrics,
                    "time": 0
                }
            }
            # Create tabs for each method plus comparison
            tab_names = ["CLAHE", "Retinex", "Zero-DCE", "EnlightenGAN", "All Results", "Metrics Comparison"]
            tabs = st.tabs(tab_names)
            # Process CLAHE
            with tabs[0]:
                st.header("CLAHE Enhancement")
                with st.spinner("Processing with CLAHE..."):
                    start = time.time()
                    enhanced = apply_clahe(img_array)
                    metrics = calculate_metrics(enhanced)
                    exec_time = time.time() - start
                    results["CLAHE"] = {
                        "image": enhanced,
                        "metrics": metrics,
                        "time": exec_time
                    }
                    display_results(img_array, enhanced, original_metrics, metrics, exec_time, st.session_state.light_mode)
            # Process Retinex
            with tabs[1]:
                st.header("Retinex Enhancement")
                # Process all Retinex variants
                with st.spinner("Processing Retinex variants..."):
                    # Siapkan sigma list
                    sigma_list = [15, 80, 250]
                    img_input = img_array.astype(np.float32) + 1.0  # Pastikan img float32 + 1 seperti di definisi fungsi

                    # Process SSR
                    start_ssr = time.time()
                    enhanced_ssr = SSR(img_input, sigma=80)  # Pakai sigma 80 default untuk SSR
                    enhanced_ssr = exposure.rescale_intensity(enhanced_ssr, out_range=(0, 255)).astype(np.uint8)
                    metrics_ssr = calculate_metrics(enhanced_ssr)
                    time_ssr = time.time() - start_ssr
                    results["Retinex (SSR)"] = {
                        "image": enhanced_ssr,
                        "metrics": metrics_ssr,
                        "time": time_ssr
                    }

                    # Process MSR
                    start_msr = time.time()
                    enhanced_msr = MSR(img_input, sigma_list)
                    enhanced_msr = exposure.rescale_intensity(enhanced_msr, out_range=(0, 255)).astype(np.uint8)
                    metrics_msr = calculate_metrics(enhanced_msr)
                    time_msr = time.time() - start_msr
                    results["Retinex (MSR)"] = {
                        "image": enhanced_msr,
                        "metrics": metrics_msr,
                        "time": time_msr
                    }

                    # Process MSRCR
                    start_msrcr = time.time()
                    enhanced_msrcr = MSRCR(img_array, sigma_list)  # MSRCR sudah auto normalisasi dalam fungsinya
                    metrics_msrcr = calculate_metrics(enhanced_msrcr)
                    time_msrcr = time.time() - start_msrcr
                    results["Retinex (MSRCR)"] = {
                        "image": enhanced_msrcr,
                        "metrics": metrics_msrcr,
                        "time": time_msrcr
                    }

                    # Process MSRCP
                    start_msrcp = time.time()
                    enhanced_msrcp = MSRCP(img_array, sigma_list)  # MSRCP juga auto normalisasi dalam fungsinya
                    metrics_msrcp = calculate_metrics(enhanced_msrcp)
                    time_msrcp = time.time() - start_msrcp
                    results["Retinex (MSRCP)"] = {
                        "image": enhanced_msrcp,
                        "metrics": metrics_msrcp,
                        "time": time_msrcp
                    }

                    # Display results for the selected method
                    method = st.radio(
                        "Retinex Method", 
                        ["SSR", "MSR", "MSRCR", "MSRCP"], 
                        horizontal=True, 
                        key="retinex_method"
                    )

                    display_results(
                        img_array,
                        results[f"Retinex ({method})"]["image"],
                        original_metrics,
                        results[f"Retinex ({method})"]["metrics"],
                        results[f"Retinex ({method})"]["time"],
                        st.session_state.light_mode
                    )
            # Process Zero-DCE
            with tabs[2]:
                st.header("Zero-DCE Enhancement")
                if load_zero_dce_model() is not None:
                    with st.spinner("Processing with Zero-DCE..."):
                        start = time.time()
                        enhanced = process_zero_dce(img_array)
                        if enhanced is not None:
                            metrics = calculate_metrics(enhanced)
                            exec_time = time.time() - start
                            results["Zero-DCE"] = {
                                "image": enhanced,
                                "metrics": metrics,
                                "time": exec_time
                            }
                            display_results(img_array, enhanced, original_metrics, metrics, exec_time, st.session_state.light_mode)
                        else:
                            st.error("Zero-DCE processing failed")
                else:
                    st.warning("Zero-DCE is not available")
            # Process EnlightenGAN
            with tabs[3]:
                st.header("EnlightenGAN Enhancement")
                if load_enlighten_model() is not None:
                    with st.spinner("Processing with EnlightenGAN..."):
                        start = time.time()
                        enhanced = process_enlighten_gan(img_array)
                        if enhanced is not None:
                            metrics = calculate_metrics(enhanced)
                            exec_time = time.time() - start
                            results["EnlightenGAN"] = {
                                "image": enhanced,
                                "metrics": metrics,
                                "time": exec_time
                            }
                            display_results(img_array, enhanced, original_metrics, metrics, exec_time, st.session_state.light_mode)
                        else:
                            st.error("EnlightenGAN processing failed")
                else:
                    st.warning("EnlightenGAN is not available")
            # All Results Comparison Tab
            with tabs[4]:
                create_image_comparison_tab(results, st.session_state.light_mode)
            # Metrics Comparison Tab
            with tabs[5]:
                if len(results) > 1:
                    display_metrics_comparison(results, st.session_state.light_mode)
                else:
                    st.warning("Process at least one enhancement method to see comparisons")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
if __name__ == "__main__":
    main()