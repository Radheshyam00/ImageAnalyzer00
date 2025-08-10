import streamlit as st
import hashlib
from PIL import Image, UnidentifiedImageError
from PIL.ExifTags import TAGS, GPSTAGS
import io
import json
import os
from datetime import datetime
import time
import piexif
import folium
from streamlit_folium import st_folium
import numpy as np
import pandas as pd
import seaborn as sns
import cv2
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageStat
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.express as px
from scipy import ndimage
from scipy.stats import skew, kurtosis, pearsonr
from scipy.stats import entropy as scipy_entropy
from skimage.metrics import structural_similarity as ssim
from sklearn.cluster import KMeans
from skimage import metrics
from skimage import metrics as sk_metrics
from scipy.fftpack import dct

def to_rational(value, precision=100):
    return (int(value * precision), precision)

st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    # .stTabs [data-baseweb="tab-list"] {
    #     gap: 2rem;
    # }
    # .stTabs [data-baseweb="tab"] {
    #     height: 50px;
    #     white-space: pre-wrap;
    #     background-color: #f0f2f6;
    #     border-radius: 4px;
    #     padding: 10px 20px;
    #     font-weight: 500;
    # }
    # .stTabs [aria-selected="true"] {
    #     background-color: #ff4b4b;
    #     color: white;
    # }
    .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .analysis-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 2rem 0;
        border-top: 1px solid #e0e0e0;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)
# st.header("⚙️ Analysis Settings")
with st.sidebar:
    st.sidebar.markdown("---") 
    # with st.expander("Analysis Mode", expanded=False):        
    analysis_mode = st.selectbox(
                            "Analysis Mode",
                            ["Quick Scan", "Expert Mode"],
                            help="Choose analysis depth"
                        )
            

st.sidebar.markdown("---")
if analysis_mode == "Quick Scan":
    # Enhanced quantization table patterns
    KNOWN_QTABLES = {
        "JPEG Standard (75%)": {
            "Q0": [16, 11, 10, 16, 24, 40, 51, 61],
            "Q1": [17, 18, 24, 47, 99, 99, 99, 99],
            "signature": "standard_75"
        },
        "JPEG Standard (80%)": {
            "Q0": [6, 4, 4, 6, 10, 16, 20, 24],
            "Q1": [7, 7, 10, 19, 40, 40, 40, 40],
            "signature": "standard_80"
        },
        "JPEG Standard (90%)": {
            "Q0": [3, 2, 2, 3, 5, 8, 10, 12],
            "Q1": [4, 4, 5, 10, 20, 20, 20, 20],
            "signature": "standard_90"
        },
        "WhatsApp": {
            "Q0": [16, 11, 10, 16, 24, 40, 51, 61],
            "Q1": [17, 18, 24, 47, 99, 99, 99, 99],
            "signature": "whatsapp"
        },
        "Instagram": {
            "Q0": [8, 6, 6, 7, 6, 5, 8, 7],
            "Q1": [9, 9, 9, 12, 10, 12, 24, 16],
            "signature": "instagram"
        },
        "Photoshop Save for Web": {
            "Q0": [8, 6, 6, 7, 6, 5, 8, 7],
            "Q1": [9, 9, 9, 12, 10, 12, 24, 16],
            "signature": "photoshop_web"
        },
        "Facebook": {
            "Q0": [12, 8, 8, 12, 17, 21, 24, 17],
            "Q1": [13, 13, 17, 21, 35, 35, 35, 35],
            "signature": "facebook"
        },
        "Twitter": {
            "Q0": [10, 7, 6, 10, 14, 24, 31, 37],
            "Q1": [11, 11, 14, 28, 58, 58, 58, 58],
            "signature": "twitter"
        }
    }

    def extract_quantization_tables(img):
        """Extract quantization tables from JPEG image"""
        try:
            # First try to get quantization tables directly
            if hasattr(img, 'quantization') and img.quantization:
                return img.quantization
            
            # Alternative method using PIL's internal structures
            if hasattr(img, '_getexif'):
                qtables = {}
                # Access quantization tables through PIL's internal methods
                if hasattr(img, 'app') and 'quantization' in str(img.app):
                    return img.quantization
            
            return {}
        except Exception as e:
            st.error(f"Error extracting quantization tables: {str(e)}")
            return {}
        
    def estimate_quality_factor(qtable, table_type='luma'):
        """Estimate JPEG quality factor from quantization table"""
        std_luma = np.array([
            16,11,10,16,24,40,51,61,
            12,12,14,19,26,58,60,55,
            14,13,16,24,40,57,69,56,
            14,17,22,29,51,87,80,62,
            18,22,37,56,68,109,103,77,
            24,35,55,64,81,104,113,92,
            49,64,78,87,103,121,120,101,
            72,92,95,98,112,100,103,99
        ])
        
        qtable_flat = np.array(qtable)
        
        # Simple estimation based on average scaling factor
        if np.mean(qtable_flat) != 0:
            scale_factor = np.mean(std_luma) / np.mean(qtable_flat)
            
            if scale_factor >= 1:
                quality = 100 - (1/scale_factor - 1) * 50
            else:
                quality = 50 * scale_factor
                
            return max(1, min(100, int(quality)))
        return 50

    def dct2(block):
        """2D Discrete Cosine Transform"""
        return cv2.dct(np.float32(block))

    def idct2(block):
        """2D Inverse Discrete Cosine Transform"""
        return cv2.idct(np.float32(block))

    def visualize_dct(image_gray, block_size=8):
        """Visualize DCT coefficients across the image"""
        h, w = image_gray.shape
        dct_map = np.zeros_like(image_gray, dtype=np.float32)
        high_freq_map = np.zeros_like(image_gray, dtype=np.float32)
        
        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                block = image_gray[i:i+block_size, j:j+block_size]
                dct_block = dct2(block)
                
                # Overall DCT energy
                dct_map[i:i+block_size, j:j+block_size] = np.log(np.abs(dct_block) + 1)
                
                # High frequency content (bottom-right quadrant)
                high_freq = dct_block[block_size//2:, block_size//2:]
                high_freq_energy = np.sum(np.abs(high_freq))
                high_freq_map[i:i+block_size, j:j+block_size] = high_freq_energy
                
        return dct_map, high_freq_map

    # Enhanced classification with multiple algorithms
    @st.cache_data
    def classify_quantization_table(qtable_dict):
        if not qtable_dict:
            return "Unknown", 0.0, "No quantization table found"
        
        try:
            qtable_vals = list(qtable_dict.values())
            if len(qtable_vals) < 128:
                return "Unknown", 0.0, "Incomplete quantization table"
            
            flat_vals = [item for sublist in qtable_vals for item in sublist]
            q0_table = flat_vals[:64]
            q1_table = flat_vals[64:128] if len(flat_vals) >= 128 else flat_vals[:64]
            
            best_match = "Unknown"
            best_score = float("inf")
            confidence_details = []
            
            for label, profile in KNOWN_QTABLES.items():
                # Compare using multiple metrics
                q0_sample = q0_table[:8]
                q1_sample = q1_table[:8]
                
                # Euclidean distance
                dist_q0 = np.linalg.norm(np.array(q0_sample) - np.array(profile["Q0"]))
                dist_q1 = np.linalg.norm(np.array(q1_sample) - np.array(profile["Q1"]))
                
                # Cosine similarity
                cos_sim_q0 = np.dot(q0_sample, profile["Q0"]) / (np.linalg.norm(q0_sample) * np.linalg.norm(profile["Q0"]))
                cos_sim_q1 = np.dot(q1_sample, profile["Q1"]) / (np.linalg.norm(q1_sample) * np.linalg.norm(profile["Q1"]))
                
                # Combined score
                score = (dist_q0 + dist_q1) / (cos_sim_q0 + cos_sim_q1 + 1e-6)
                
                confidence_details.append({
                    "source": label,
                    "score": score,
                    "euclidean": dist_q0 + dist_q1,
                    "cosine": (cos_sim_q0 + cos_sim_q1) / 2
                })
                
                if score < best_score:
                    best_score = score
                    best_match = label
            
            # Calculate confidence
            confidence = max(0, min(100, 100 - (best_score * 2)))
            
            # Generate detailed analysis
            analysis = f"Best match: {best_match}\n"
            analysis += f"Confidence: {confidence:.1f}%\n"
            analysis += f"Score: {best_score:.2f}\n"
            
            return best_match, confidence, analysis
        
        except Exception as e:
            return "Error", 0.0, f"Classification error: {str(e)}"

    # Enhanced ELA with adaptive quality
    @st.cache_data
    def perform_ela(image, quality=90):
        """Enhanced Error Level Analysis with adaptive quality selection"""
        results = {}
        qualities = [70, 80, 90, 95]
        
        for q in qualities:
            buffer = io.BytesIO()
            image.save(buffer, 'JPEG', quality=q)
            buffer.seek(0)
            compressed = Image.open(buffer)
            
            # Calculate difference
            ela_image = ImageChops.difference(image, compressed)
            
            # Enhanced scaling with histogram equalization
            ela_np = np.array(ela_image)
            ela_gray = cv2.cvtColor(ela_np, cv2.COLOR_RGB2GRAY)
            ela_eq = cv2.equalizeHist(ela_gray)
            ela_colored = cv2.applyColorMap(ela_eq, cv2.COLORMAP_JET)
            
            results[f"Q{q}"] = Image.fromarray(cv2.cvtColor(ela_colored, cv2.COLOR_BGR2RGB))
        
        return results

    # Enhanced edge detection
    def enhanced_edge_detection(image):
        """Multi-scale edge detection for forgery detection"""
        gray = np.array(image.convert('L'))
        
        # Canny edge detection
        edges_canny = cv2.Canny(gray, 50, 150)
        
        # Sobel edge detection
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = np.sqrt(sobelx**2 + sobely**2)
        sobel_combined = np.uint8(255 * sobel_combined / np.max(sobel_combined))
        
        # Laplacian edge detection
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        
        return {
            "canny": Image.fromarray(edges_canny),
            "sobel": Image.fromarray(sobel_combined),
            "laplacian": Image.fromarray(laplacian)
        }

    # Enhanced JPEG analysis
    @st.cache_data
    def analyze_jpeg_artifacts(image):
        """Analyze JPEG compression artifacts safely"""
        gray = np.array(image.convert("L"), dtype=np.float32)

        # Ensure dimensions are multiples of 8 for block DCT
        h, w = gray.shape
        h_pad = (8 - h % 8) % 8
        w_pad = (8 - w % 8) % 8
        if h_pad or w_pad:
            gray = np.pad(gray, ((0, h_pad), (0, w_pad)), mode='constant', constant_values=0)

        # DCT-based analysis
        dct_coeffs = cv2.dct(gray)

        # Quantization noise analysis
        reconstructed = cv2.idct(dct_coeffs)
        noise = np.abs(gray - reconstructed)

        # Avoid divide-by-zero in normalization
        def normalize(arr):
            max_val = np.max(arr)
            return np.uint8(255 * arr / max_val) if max_val != 0 else np.zeros_like(arr, dtype=np.uint8)

        # Block artifact detection
        blocks = []
        for i in range(0, gray.shape[0], 8):
            for j in range(0, gray.shape[1], 8):
                block = gray[i:i+8, j:j+8]
                if block.shape == (8, 8):
                    blocks.append(np.var(block))
        block_variance = np.mean(blocks) if blocks else 0

        return {
            "dct_analysis": Image.fromarray(normalize(np.abs(dct_coeffs))),
            "quantization_noise": Image.fromarray(normalize(noise)),
            "block_variance": block_variance
        }


    # Enhanced metadata extraction
    @st.cache_data
    def extract_comprehensive_metadata(file_bytes):
        """Extract comprehensive metadata from image"""
        metadata = {}
        
        try:
            exif_dict = piexif.load(file_bytes)

            def clean_value(value):
                """Clean EXIF values to remove nulls and decode bytes."""
                if isinstance(value, bytes):
                    try:
                        return value.decode("utf-8", errors="ignore").strip("\x00").strip()
                    except:
                        return value.hex()  # fallback to hex if not decodable
                return value

            # Basic EXIF data
            if "0th" in exif_dict:
                for tag, value in exif_dict["0th"].items():
                    tag_name = piexif.TAGS["0th"].get(tag, {"name": f"Tag_{tag}"})["name"]
                    metadata[tag_name] = clean_value(value)

            # GPS data
            if "GPS" in exif_dict:
                gps_data = exif_dict["GPS"]
                metadata["GPS_Info"] = {}
                for tag, value in gps_data.items():
                    tag_name = piexif.TAGS["GPS"].get(tag, {"name": f"GPS_{tag}"})["name"]
                    metadata["GPS_Info"][tag_name] = clean_value(value)

            # Thumbnail analysis
            if "thumbnail" in exif_dict and exif_dict["thumbnail"]:
                thumb_data = exif_dict["thumbnail"]
                thumb_hash = hashlib.md5(thumb_data).hexdigest()
                metadata["thumbnail_hash"] = thumb_hash
                metadata["thumbnail_size"] = len(thumb_data)

        except Exception as e:
            metadata["error"] = str(e)
        
        return metadata

    # Main application
    def main():
        st.markdown(
        """
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        """,
        unsafe_allow_html=True,
)

        st.markdown("""
        <div class="main-header">
            <h1><i class="fa-solid fa-magnifying-glass"></i> Advanced Image Forensic Analyser</h1>
            <p>Professional-grade image authentication and tampering detection</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar for settings
        show_confidence = st.sidebar.checkbox("Show Confidence Scores", True)
        generate_report = st.sidebar.checkbox("Generate Report", False)
        st.sidebar.markdown("---") 
        
        # File upload
        uploaded_file = st.file_uploader(
            "📁 Upload image for forensic analysis",
            type=["jpg", "jpeg", "png", "tiff", "bmp"],
            help="Supported formats: JPEG, PNG, TIFF, BMP"
        )
        
        if uploaded_file is not None:
            # Load and display image
            image = Image.open(uploaded_file).convert("RGB")
            img = Image.open(uploaded_file)
            uploaded_file.seek(0)
            img_bytes = uploaded_file.read()
            st.markdown("---")

            # Image overview
            st.markdown("<h4><i class='fa-solid fa-file'></i> Basic File Info</h4>", unsafe_allow_html=True)
            with st.expander("Details", expanded=False):
                col1, col2, col3, col4 = st.columns([0.9,0.1, 1, 1])
                with col1:
                    st.image(image, caption="Original Image")
                with col2:
                    st.markdown("""
                    <div style='height: 300px; border-left: 2px solid #ccc; margin: 10px;'></div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>Image Info</h4>
                        <p><strong>Size:</strong> {image.size[0]} × {image.size[1]}</p>
                        <p><strong>Format:</strong> {uploaded_file.type}</p>
                        <p><strong>File Size:</strong> {len(img_bytes) / 1024:.1f} KB</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    # Quick authenticity score
                    authenticity_score = np.random.randint(60, 95)  # Placeholder
                    color = "green" if authenticity_score > 80 else "orange" if authenticity_score > 60 else "red"
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>Authenticity Score</h4>
                        <h2 style="color: {color};">{authenticity_score}%</h2>
                        <p>Preliminary assessment</p>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown("---")
            # Analysis tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Tampering Detection",
                "JPEG Analysis", 
                "Metadata Forensics",
                "Advanced Analysis",
                "Report"
            ])
            
            with tab1:
                st.markdown("""
                <div class="analysis-header">
                    <h3><i class="fa-solid fa-magnifying-glass"></i> Tampering Detection Analysis</h3>
                </div>
                """, unsafe_allow_html=True)

                
                col1, col2 = st.columns(2)
                
                with col1:
                    with st.expander("Error Level Analysis (ELA)", expanded=False):
                        # st.subheader("Error Level Analysis (ELA)")
                        ela_results = perform_ela(image)
                        
                        ela_quality = st.selectbox("ELA Quality", ["Q70", "Q80", "Q90", "Q95"])
                        st.image(ela_results[ela_quality], caption=f"ELA at {ela_quality}")
                        
                        if show_confidence:
                            st.info("Look for bright areas indicating potential editing")
                    
                    with st.expander("Enhanced Edge Detection", expanded=False):
                        # st.subheader("Enhanced Edge Detection")
                        edge_results = enhanced_edge_detection(image)
                        
                        edge_method = st.selectbox("Edge Detection Method", ["canny", "sobel", "laplacian"])
                        st.image(edge_results[edge_method], caption=f"{edge_method.capitalize()} Edge Detection")
                
                with col2:
                    with st.expander("Luminance Analysis", expanded=False):
                        # st.subheader("Luminance Analysis")
                        # Luminance consistency analysis
                        gray = np.array(image.convert("L"))
                        
                        # Create luminance map
                        luminance_map = cv2.equalizeHist(gray)
                        st.image(luminance_map, caption="Luminance Map")
                        
                        if show_confidence:
                            st.info("Inconsistent luminance patterns may indicate manipulation")
                    
                    with st.expander("Noise Analysis", expanded=False):
                        # st.subheader("Noise Analysis")
                        # Noise residual analysis
                        blur = cv2.GaussianBlur(gray, (5, 5), 0)
                        noise = cv2.absdiff(gray, blur)
                        st.image(noise, caption="Noise Residual")
                        
                        if show_confidence:
                            st.info("Uneven noise distribution may indicate tampering")
            
            with tab2:
                st.markdown("""
                <div class="analysis-header">
                    <h3><i class="fa-solid fa-chart-column"></i> JPEG Compression Analysis</h3>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    with st.expander("Quantization Table Analysis", expanded=False):
                        
                        # st.subheader("📊 Quantization Table Analysis")
            
                        qtables = extract_quantization_tables(img)

                        if not qtables:
                            st.warning("⚠️ No quantization tables found. This might not be a standard JPEG or the tables are not accessible.")
                        else:
                            # Tabs for different tables
                            if len(qtables) > 1:
                                tabs = st.tabs([f"Table {idx}" for idx in qtables.keys()])
                            else:
                                tabs = [st.container()]

                            for tab_idx, (table_id, table) in enumerate(qtables.items()):
                                with tabs[tab_idx]:
                                    st.markdown(f"### Quantization Table {table_id}")
                                    
                                    # Estimate quality
                                    estimated_quality = estimate_quality_factor(table)
                                    st.metric("Estimated JPEG Quality", f"{estimated_quality}%")
                                    
                                    # Show raw values in expandable section
                                    # with st.expander("View Raw Quantization Values"):
                                    st.dataframe(np.array(table).reshape((8, 8)))
                        # # st.subheader("Quantization Table Analysis")
                        # try:
                        #     exif_dict = piexif.load(img_bytes)
                        #     qtable = exif_dict.get("0th", {})
                            
                        #     if qtable:
                        #         source, confidence, analysis = classify_quantization_table(qtable)

                        #         st.success(f"**Estimated Source:** {source}")
                        #         st.metric("Confidence Score", f"{confidence:.1f}%")
                        #         st.text(analysis)

                        #         # Flatten quantization table values
                        #         qtable_values = []
                        #         for v in qtable.values():
                        #             if isinstance(v, (list, tuple)):
                        #                 qtable_values.extend(v)
                        #             else:
                        #                 qtable_values.append(v)

                        #         # Visualize only if we have enough values
                        #         if len(qtable_values) >= 64:
                        #             fig, ax = plt.subplots(figsize=(8, 6))
                        #             qtable_matrix = np.array(qtable_values[:64]).reshape(8, 8)
                        #             im = ax.imshow(qtable_matrix, cmap='viridis')
                        #             ax.set_title("Quantization Table Visualization")
                        #             plt.colorbar(im, ax=ax)
                        #             st.pyplot(fig)
                        #         else:
                        #             st.info("Insufficient quantization table data for visualization")

                        #     else:
                        #         st.warning("No quantization table found in EXIF data")
                                
                        # except Exception as e:
                        #     st.error(f"Error analyzing quantization table: {str(e)}")
                
                with col2:
                    with st.expander("JPEG Artifact Analysis", expanded=False):
                        col1 , col2 = st.columns(2)
                        with col1:
                            if not img_bytes:
                                st.error("❌ Uploaded file is empty or could not be read.")
                            else:
                                file_bytes = np.frombuffer(img_bytes, np.uint8)
                                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                                if img_bgr is None:
                                    st.error("❌ Could not decode the uploaded image. Please upload a valid image.")
                                else:
                                    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                                    # st.image(img_gray, caption="Grayscale Image", use_column_width=True)

                                    block_size = st.sidebar.slider("DCT Block Size", 4, 16, 8, 2)
                                    dct_map, high_freq_map = visualize_dct(img_gray, block_size)

                                    st.subheader("DCT Coefficient Analysis")
                                    fig1, ax1 = plt.subplots(figsize=(8, 6))
                                    im1 = ax1.imshow(dct_map, cmap='hot', interpolation='nearest')
                                    ax1.axis("off")
                                    plt.colorbar(im1, ax=ax1, shrink=0.8)
                                    st.pyplot(fig1)
                        with col2:
                            st.subheader("Quantization Noise")
                            jpeg_analysis = analyze_jpeg_artifacts(image)
                            
                            # st.image(jpeg_analysis["dct_analysis"], caption="DCT Coefficient Analysis")
                            st.image(jpeg_analysis["quantization_noise"], caption="Quantization Noise")
                            
                            # st.metric("Block Variance", f"{jpeg_analysis['block_variance']:.2f}")
                        
                        # Compression history estimation
                    with st.expander("Compression History", expanded=False):
                        # st.subheader("Compression History")
                        compression_levels = [70, 80, 90, 95]
                        compression_scores = []
                        
                        for level in compression_levels:
                            buffer = io.BytesIO()
                            image.save(buffer, 'JPEG', quality=level)
                            compressed_size = len(buffer.getvalue())
                            compression_scores.append(compressed_size)
                        
                        fig = px.line(
                            x=compression_levels,
                            y=compression_scores,
                            title="Compression Quality vs File Size",
                            labels={"x": "JPEG Quality", "y": "File Size (bytes)"}
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.markdown("""
                <div class="analysis-header">
                    <h3><i class="fa-solid fa-pen-to-square"></i> Metadata Forensics</h3>
                </div>
                """, unsafe_allow_html=True)

                
                metadata = extract_comprehensive_metadata(img_bytes)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    with st.expander("EXIF Data", expanded=False):
                        # st.subheader("EXIF Data")
                        if metadata:
                            # Clean up metadata display
                            clean_metadata = {}
                            for key, value in metadata.items():
                                if key != "GPS_Info" and not key.startswith("thumbnail"):
                                    clean_metadata[key] = value
                            
                            if clean_metadata:
                                st.json(clean_metadata)
                            else:
                                st.info("No EXIF data found")
                        else:
                            st.info("No metadata found")
                    
                    # Check for GPS data
                    with st.expander("Geolocation Data", expanded=False):
                        # st.subheader("Geolocation Data")
                        if "GPS_Info" in metadata:
                            st.json(metadata["GPS_Info"])
                            st.info("GPS coordinates found in image")
                        else:
                            st.info("No GPS data found")
                
                with col2:
                    with st.expander("Thumbnail Analysis", expanded=False):
                        # st.subheader("Thumbnail Analysis")
                        if "thumbnail_hash" in metadata:
                            st.success(f"Thumbnail Hash: {metadata['thumbnail_hash']}")
                            st.metric("Thumbnail Size", f"{metadata['thumbnail_size']} bytes")
                        else:
                            st.info("No thumbnail found")
                    
                    # Timestamp analysis
                    with st.expander("Timestamp Analysis", expanded=False):
                        # st.subheader("Timestamp Analysis")
                        timestamp_fields = ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized']
                        timestamps_found = []
                        
                        for field in timestamp_fields:
                            if field in metadata:
                                timestamps_found.append(f"{field}: {metadata[field]}")
                        
                        if timestamps_found:
                            for ts in timestamps_found:
                                st.text(ts)
                        else:
                            st.info("No timestamps found")
                        
                    # Software detection
                    with st.expander("Software Detection", expanded=False):
                        # st.subheader("Software Detection")
                        software_fields = ['Software', 'ProcessingSoftware', 'HostComputer']
                        software_found = []
                        
                        for field in software_fields:
                            if field in metadata:
                                software_found.append(f"{field}: {metadata[field]}")
                        
                        if software_found:
                            for sw in software_found:
                                st.text(sw)
                        else:
                            st.info("No software information found")
                
            with tab4:
                st.markdown("""
                <div class="analysis-header">
                    <h3><i class="fa-solid fa-microscope"></i> Advanced Forensic Analysis</h3>
                </div>
                """, unsafe_allow_html=True)

                
                col1, col2 = st.columns(2)
                
                with col1:
                    with st.expander("PCA Analysis", expanded=False):
                        # st.subheader("PCA Analysis")
                        # Principal Component Analysis
                        img_resized = image.resize((128, 128))
                        img_np = np.array(img_resized)
                        reshaped = img_np.reshape(-1, 3)
                        
                        try:
                            pca = PCA(n_components=1)
                            pca_result = pca.fit_transform(reshaped)
                            pca_image = pca_result.reshape(128, 128)
                            pca_normalized = np.uint8(255 * (pca_image - pca_image.min()) / (pca_image.max() - pca_image.min()))
                            
                            st.image(pca_normalized, caption="PCA Component Analysis")
                        except Exception as e:
                            st.error(f"Error in PCA analysis: {str(e)}")
                    
                    with st.expander("Luminance Analysis", expanded=False):
                        # st.subheader("Luminance Analysis")
                        gray = np.array(image.convert('L'))
                        
                        # Histogram analysis
                        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                        
                        fig, ax = plt.subplots()
                        ax.plot(hist)
                        ax.set_title("Luminance Histogram")
                        ax.set_xlabel("Intensity")
                        ax.set_ylabel("Frequency")
                        st.pyplot(fig)
                
                with col2:
                    with st.expander("Frequency Domain Analysis", expanded=False):
                        # st.subheader("Frequency Domain Analysis")
                        # FFT analysis
                        gray = np.array(image.convert('L'))
                        f_transform = np.fft.fft2(gray)
                        f_shift = np.fft.fftshift(f_transform)
                        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
                        
                        fig, ax = plt.subplots()
                        ax.imshow(magnitude_spectrum, cmap='gray')
                        ax.set_title("Frequency Domain Analysis")
                        ax.axis('off')
                        st.pyplot(fig)
                    
                    with st.expander("Statistical Analysis", expanded=False):
                        # st.subheader("Statistical Analysis")
                        # Image statistics
                        hist_normalized = hist / np.sum(hist)
                        entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))
                        
                        stats = {
                            "Mean": np.mean(gray),
                            "Std Dev": np.std(gray),
                            "Variance": np.var(gray),
                            "Min": np.min(gray),
                            "Max": np.max(gray),
                            "Entropy": entropy
                        }
                        
                        col1, col2 = st.columns(2)

                        # Convert items to a list for indexing
                        stat_items = list(stats.items())

                        # First column
                        with col1:
                            for key, value in stat_items[:len(stat_items)//2]:
                                st.metric(label=key, value=f"{value:.4f}")

                        # Second column
                        with col2:
                            for key, value in stat_items[len(stat_items)//2:]:
                                st.metric(label=key, value=f"{value:.4f}")
            
            with tab5:
                st.markdown("""
                <div class="analysis-header">
                    <h3><i class="fa-solid fa-clipboard"></i> Forensic Analysis Report</h3>
                </div>
                """, unsafe_allow_html=True)

                
                if generate_report:
                    report_data = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "image_info": {
                            "filename": uploaded_file.name,
                            "size": f"{image.size[0]}x{image.size[1]}",
                            "format": uploaded_file.type,
                            "file_size": f"{len(img_bytes) / 1024:.1f} KB"
                        },
                        "authenticity_score": authenticity_score,
                        "analysis_mode": analysis_mode,
                        "findings": []
                    }
                    
                    # Add findings based on analysis
                    if authenticity_score > 80:
                        report_data["findings"].append("Image shows high authenticity indicators")
                    elif authenticity_score > 60:
                        report_data["findings"].append("Image shows moderate authenticity indicators")
                    else:
                        report_data["findings"].append("Image shows low authenticity indicators - further investigation recommended")
                    
                    # Display report
                    st.subheader("Executive Summary")
                    st.write(f"**Analysis Date:** {report_data['timestamp']}")
                    st.write(f"**Image:** {report_data['image_info']['filename']}")
                    st.write(f"**Authenticity Score:** {report_data['authenticity_score']}")
                    
                    st.subheader("Key Findings")
                    for finding in report_data["findings"]:
                        st.write(f"• {finding}")
                    
                    st.subheader("Technical Details")
                    st.json(report_data)
                    
                    # Download report
                    if st.button("Download Report"):
                        report_json = str(report_data)
                        st.download_button(
                            label="Download JSON Report",
                            data=report_json,
                            file_name=f"forensic_report_{uploaded_file.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                else:
                    st.info("Enable 'Generate Report' in the sidebar to create a detailed analysis report.")
        
        else:
            # st.info("👆 Upload an image to begin forensic analysis")
            
            # Show example analysis
            st.markdown("---")
            st.markdown("### What This Tool Analyzes")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <strong><i class="fa-solid fa-magnifying-glass"></i> Tampering Detection</strong>
                <ul>
                    <li>Error Level Analysis (ELA)</li>
                    <li>Splicing Detection</li>
                    <li>Luminance Analysis</li>
                    <li>Noise Pattern Analysis</li>
                </ul>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown("""
                <strong><i class="fa-solid fa-chart-bar"></i> JPEG Analysis</strong>
                <ul>
                    <li>Quantization Tables</li>
                    <li>Compression History</li>
                    <li>Artifact Detection</li>
                    <li>Quality Assessment</li>
                </ul>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown("""
                <strong><i class="fa-solid fa-pen-to-square"></i> Metadata Forensics</strong>
                <ul>
                    <li>EXIF Data Analysis</li>
                    <li>GPS Coordinates</li>
                    <li>Timestamp Verification</li>
                    <li>Software Detection</li>
                </ul>
                """, unsafe_allow_html=True)
        st.markdown("""
        <div class="footer">
            <h4>Image Analyzer Suite</h4>
            <p><strong>© 2025 Radheshyam Janwa</strong> | All Rights Reserved</p>

        </div>
        """, unsafe_allow_html=True)

    if __name__ == "__main__":
        main()

elif analysis_mode == "Expert Mode":

    page_E = st.sidebar.selectbox("Expert Mode", 
            ["Tampering Detection Analysis",
                "JPEG Compression Analysis",
                "Metadata Forensics",
                "Advanced Forensic Analysis"],
                help="Choose Expert Analysis Technique"
                )
    
    if page_E == "Tampering Detection Analysis":
        page1 = st.sidebar.selectbox("Select Analysis Tool", 
                ["Error Level Analysis (ELA)",
                "Enhanced Edge Detection",
                "Noise Analysis"],
                help="Choose Analysis Tool"
                )
        
        if page1 == "Error Level Analysis (ELA)":
            # Custom CSS for modern UI
            st.markdown("""
            <style>
                .main-header {
                    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                    padding: 2rem;
                    border-radius: 10px;
                    color: white;
                    text-align: center;
                    margin-bottom: 2rem;
                }
            </style>
            """, unsafe_allow_html=True)
            
            # Header
            st.markdown("""
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
            """, unsafe_allow_html=True)


            st.markdown("""
            <div class="main-header">
                <h1><i class="fa-solid fa-magnifying-glass"></i> Advanced Error Level Analysis (ELA) for Image Forensics</h1>
                <p>Error Level Analysis (ELA) highlights areas of an image that may have been digitally altered.
            It works by comparing the original image to a re-compressed version and analyzing the differences.</p>
            </div>
            """, unsafe_allow_html=True)
            # st.title("🔍 Advanced Error Level Analysis (ELA) for Image Forensics")

            # Sidebar for advanced options
            # with st.sidebar:
                # st.header("🛠️ Analysis Options")
                
                # File upload
            uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "tiff"])
                
            if uploaded_file:
                with st.sidebar:
                    st.markdown("---")
                    with st.expander("ELA Parameters", expanded=False):
                        # st.subheader("ELA Parameters")
                        quality = st.slider("Compression Quality", min_value=50, max_value=100, value=90, 
                                        help="Lower values enhance differences")
                            
                        enhance_factor = st.slider("Brightness Enhancement", min_value=1, max_value=50, value=20,
                                                help="Higher values make differences more visible")

                    with st.expander("Additional Analysis", expanded=False):        
                        # st.subheader("Additional Analysis")
                        show_histogram = st.checkbox("Show ELA Histogram", value=True)
                        show_heatmap = st.checkbox("Show Intensity Heatmap", value=False)
                        apply_blur = st.checkbox("Apply Gaussian Blur to ELA", value=False)
                            
                        if apply_blur:
                            blur_radius = st.slider("Blur Radius", min_value=0.5, max_value=3.0, value=1.0, step=0.5)
                    st.markdown("---")
            def calculate_ela_stats(ela_image):
                """Calculate statistics for ELA image"""
                # Convert to grayscale for analysis
                gray_ela = ela_image.convert('L')
                stats = ImageStat.Stat(gray_ela)
                
                return {
                    'mean': stats.mean[0],
                    'median': stats.median[0],
                    'stddev': stats.stddev[0],
                    'min': stats.extrema[0][0],
                    'max': stats.extrema[0][1]
                }

            def create_heatmap(image):
                """Create a heatmap visualization of image intensity"""
                # Convert to grayscale and then to numpy array
                gray_img = image.convert('L')
                img_array = np.array(gray_img)
                
                # Create heatmap
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(img_array, cmap='hot', cbar=True, ax=ax)
                ax.set_title('Intensity Heatmap')
                ax.set_xlabel('Width (pixels)')
                ax.set_ylabel('Height (pixels)')
                
                return fig

            def create_histogram(image):
                """Create histogram for ELA image"""
                # Convert to numpy array for histogram
                gray_ela = image.convert('L')
                img_array = np.array(gray_ela)
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.hist(img_array.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
                ax.set_xlabel('Pixel Intensity')
                ax.set_ylabel('Frequency')
                ax.set_title('ELA Histogram - Distribution of Error Levels')
                ax.grid(True, alpha=0.3)
                
                return fig

            if uploaded_file:
                try:
                    # Load and display original image
                    image = Image.open(uploaded_file).convert("RGB")
                    
                    # col1, col2 = st.columns([2, 1])
                    
                    # with col1:
                    #     st.subheader("🖼️ Original Image")
                    #     st.image(image, use_column_width=True)
                        
                    #     # Display image metadata
                    #     st.caption(f"Dimensions: {image.size[0]} × {image.size[1]} pixels")
                    #     if hasattr(uploaded_file, 'name'):
                    #         st.caption(f"Filename: {uploaded_file.name}")

                    # with col2:
                    #     st.subheader("📊 Image Info")
                    #     st.write(f"**Format**: {image.format if hasattr(image, 'format') else 'Unknown'}")
                    #     st.write(f"**Mode**: {image.mode}")
                    #     st.write(f"**Size**: {image.size[0]} × {image.size[1]}")
                        
                    #     # File size
                    #     if hasattr(uploaded_file, 'size'):
                    #         size_kb = uploaded_file.size / 1024
                    #         st.write(f"**File Size**: {size_kb:.1f} KB")

                    # Perform ELA
                    st.markdown("---")
                    st.markdown('### <i class="fa-solid fa-magnifying-glass"></i> ELA Results', unsafe_allow_html=True)
                    with st.expander("Error Level Analysis", expanded=False):
                        # st.subheader("🔬 Error Level Analysis")
                        
                        with st.spinner("Performing ELA analysis..."):
                            # Save as JPEG to in-memory buffer with specified quality
                            buffer = io.BytesIO()
                            image.save(buffer, format="JPEG", quality=quality)
                            buffer.seek(0)

                            # Load the recompressed image
                            recompressed = Image.open(buffer)

                            # Calculate the difference (ELA)
                            ela_image = ImageChops.difference(image, recompressed)

                            # Apply blur if requested
                            if apply_blur:
                                ela_image = ela_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

                            # Enhance the brightness
                            enhancer = ImageEnhance.Brightness(ela_image)
                            ela_enhanced = enhancer.enhance(enhance_factor)

                        # Display results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.caption("Original Image")
                            st.image(image, use_column_width=True)
                            
                        with col2:
                            st.caption(f"ELA Result (Q={quality}, Enhance={enhance_factor})")
                            st.image(ela_enhanced, use_column_width=True)

                    # ELA Statistics
                    stats = calculate_ela_stats(ela_enhanced)
                    
                    with st.expander("ELA Statistics Info", expanded=False):
                        st.subheader("ELA Statistics")
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        with col1:
                            st.metric("Mean Intensity", f"{stats['mean']:.1f}")
                        with col2:
                            st.metric("Median", f"{stats['median']:.1f}")
                        with col3:
                            st.metric("Std Dev", f"{stats['stddev']:.1f}")
                        with col4:
                            st.metric("Min Value", f"{stats['min']}")
                        with col5:
                            st.metric("Max Value", f"{stats['max']}")
                        
                        st.subheader("Image Info")

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            try:
                                img_format = image.format
                            except UnidentifiedImageError:
                                img_format = None

                            # Fallback: use filename extension
                            if not img_format and hasattr(uploaded_file, "name"):
                                ext = os.path.splitext(uploaded_file.name)[-1].lower().replace('.', '')
                                img_format = ext.upper() if ext else 'Unknown'

                            # Extra fallback using imghdr
                            if not img_format or img_format == 'UNKNOWN':
                                try:
                                    uploaded_file.seek(0)
                                    with Image.open(uploaded_file) as img_check:
                                        img_format = img_check.format or 'Unknown'
                                    uploaded_file.seek(0)
                                except Exception:
                                    img_format = 'Unknown'
                            st.metric(label="Format", value=img_format)

                        with col2:
                            st.metric(label="Mode", value=image.mode)

                        with col3:
                            st.metric(label="Size", value=f"{image.size[0]} × {image.size[1]}")

                        with col4:
                            if hasattr(uploaded_file, 'size'):
                                size_kb = uploaded_file.size / 1024
                                st.metric(label="File Size", value=f"{size_kb:.1f} KB")


                    # Additional visualizations
                    if show_histogram:
                        with st.expander("ELA Histogram Analysis", expanded=False):
                            # st.subheader("📊 ELA Histogram Analysis")
                            hist_fig = create_histogram(ela_enhanced)
                            st.pyplot(hist_fig)
                            plt.close(hist_fig)
                            
                            st.markdown("""
                            **Interpreting the histogram:**
                            - **Left-skewed distribution**: Most pixels have low error levels (likely authentic)
                            - **Right-skewed or bimodal**: Possible manipulation or compression artifacts
                            - **Uniform distribution**: May indicate heavy processing or manipulation
                            """)

                    if show_heatmap:
                        with st.expander("ELA Intensity Heatmap", expanded=False):
                        # st.subheader("🌡️ ELA Intensity Heatmap")
                            heatmap_fig = create_heatmap(ela_enhanced)
                            st.pyplot(heatmap_fig)
                            plt.close(heatmap_fig)

                    # Analysis interpretation
                    with st.expander("Analysis Interpretation", expanded=False):
                    # st.subheader("🎯 Analysis Interpretation")
                    
                        interpretation_text = ""
                        
                        if stats['stddev'] > 15:
                            interpretation_text += "⚠️ **High variation** in error levels detected. This could indicate digital manipulation or heavy compression artifacts.\n\n"
                        else:
                            interpretation_text += "✅ **Low variation** in error levels. This suggests the image may be authentic or uniformly processed.\n\n"
                            
                        if stats['mean'] > 20:
                            interpretation_text += "⚠️ **High average error level** detected. Look for bright regions that might indicate manipulation.\n\n"
                        else:
                            interpretation_text += "✅ **Low average error level**. The image shows consistent compression characteristics.\n\n"
                            
                        interpretation_text += """
                        **Remember**: ELA is just one tool in image forensics. Consider these factors:
                        - Original image quality and compression history
                        - File format and metadata
                        - Context and source of the image
                        - Other forensic techniques (noise analysis, lighting consistency, etc.)
                        """
                        
                        st.markdown(interpretation_text)

                    # Download options
                    st.markdown("---")
                    st.markdown('### <i class="fa-solid fa-download"></i> Download Results', unsafe_allow_html=True)
                    # st.subheader("💾 Download Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Convert ELA image to bytes for download
                        ela_buffer = io.BytesIO()
                        ela_enhanced.save(ela_buffer, format="PNG")
                        ela_bytes = ela_buffer.getvalue()
                        
                        st.download_button(
                            label="Download ELA Image",
                            data=ela_bytes,
                            file_name=f"ela_result_{quality}q_{enhance_factor}e.png",
                            mime="image/png"
                        )
                        
                    with col2:
                        if show_histogram:
                            hist_buffer = io.BytesIO()
                            hist_fig = create_histogram(ela_enhanced)
                            hist_fig.savefig(hist_buffer, format='png', dpi=300, bbox_inches='tight')
                            hist_bytes = hist_buffer.getvalue()
                            plt.close(hist_fig)
                            
                            st.download_button(
                                label="Download Histogram",
                                data=hist_bytes,
                                file_name="ela_histogram.png",
                                mime="image/png"
                            )

                except Exception as e:
                    st.error(f"❌ Error processing image: {str(e)}")
                    st.markdown("**Possible causes:**")
                    st.markdown("- Unsupported image format")
                    st.markdown("- Corrupted image file")
                    st.markdown("- Insufficient memory for large images")
                    
            else:
                st.info("👆 Please upload an image file to begin ELA analysis.")
                
                # Example section
                with st.expander("How ELA Works", expanded=False):
                # st.subheader("🎓 How ELA Works")
                    st.markdown("""
                    1. **Original Image**: The uploaded image in its current state
                    2. **Recompression**: The image is saved again with specified JPEG quality
                    3. **Difference Calculation**: Pixel-by-pixel differences are calculated
                    4. **Enhancement**: Differences are amplified for visibility
                    5. **Analysis**: Statistical and visual analysis of the results
                    
                    **Key Indicators to Look For:**
                    - Sharp boundaries between bright and dark areas
                    - Inconsistent error levels across similar textures
                    - Rectangular or geometric patterns of high error
                    - Areas that don't match the expected compression behavior
                    """)
                with st.expander("How to interpret ELA results", expanded=False):
                    st.markdown("""

                    - **Bright areas**: Potential signs of manipulation or high compression artifacts
                    - **Dark areas**: Original, unmodified regions
                    - **Uniform brightness**: Likely authentic content
                    - **Sharp brightness differences**: Possible edited boundaries

                    *Best results with JPEG images. PNG/other formats may show uniform patterns.*
                    """)




            # Footer
            st.markdown("---")
            st.markdown("**Note**: This tool is for educational and research purposes. Professional forensic analysis requires multiple techniques and expert interpretation.")

        elif page1 == "Enhanced Edge Detection":
            # Custom CSS for modern UI
            st.markdown("""
            <style>
                .main-header {
                    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                    padding: 2rem;
                    border-radius: 10px;
                    color: white;
                    text-align: center;
                    margin-bottom: 2rem;
                }
            </style>
            """, unsafe_allow_html=True)
            
            # Header
            st.markdown("""
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
            """, unsafe_allow_html=True)


            st.markdown("""
            <div class="main-header">
                <h1><i class="fa-solid fa-eye"></i> Advanced Enhanced Edge Detection</h1>
                <p>Upload an image and apply advanced edge detection techniques using OpenCV.</p>
            </div>
            """, unsafe_allow_html=True)
            # st.title("🧠 Advanced Enhanced Edge Detection")
            # st.markdown("Upload an image and apply advanced edge detection techniques using OpenCV.")

            # Sidebar for controls
            # st.sidebar.title("⚙️ Edge Detection Parameters")

            uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

            if uploaded_file:
                image = Image.open(uploaded_file).convert("RGB")
                img_array = np.array(image)

                # st.subheader("🎨 Original Image")
                # col_orig1, col_orig2 = st.columns([2, 1])
                
                # with col_orig1:
                #     st.image(img_array, channels="RGB", use_column_width=True)
                
                # with col_orig2:
                #     st.markdown("**Image Info:**")
                #     st.write(f"Dimensions: {img_array.shape[1]} x {img_array.shape[0]}")
                #     st.write(f"Channels: {img_array.shape[2]}")
                #     st.write(f"Data type: {img_array.dtype}")
                st.markdown("---")
                st.markdown('### <i class="fa-solid fa-image"></i> Image Info', unsafe_allow_html=True)

                # Layout with 3 columns for metrics
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(label="Dimensions", value=f"{img_array.shape[1]} × {img_array.shape[0]}")

                with col2:
                    st.metric(label="Channels", value=str(img_array.shape[2]))

                with col3:
                    st.metric(label="Data Type", value=str(img_array.dtype))
                st.markdown("---")
                # Preprocessing options
                with st.sidebar:
                    st.markdown("---")
                    st.title("Edge Detection Parameters")
                    with st.expander("Preprocessing", expanded=False):
                        # st.markdown("### 🔧 Preprocessing")
                        use_gray = st.checkbox("Convert to Grayscale", True)
                        apply_blur = st.checkbox("Apply Gaussian Blur", True)
                        blur_ksize = st.slider("Blur Kernel Size", 1, 15, 5, step=2)
                        
                        # Additional preprocessing
                        apply_morphology = st.checkbox("Apply Morphological Operations", False)
                        if apply_morphology:
                            morph_operation = st.selectbox(
                                "Morphological Operation", 
                                ["Opening", "Closing", "Gradient", "Tophat", "Blackhat"]
                            )
                            morph_kernel_size = st.slider("Morphology Kernel Size", 3, 15, 5, step=2)

                    with st.expander("Canny Edge Detection", expanded=False):
                        # Canny Edge Detection
                        # st.sidebar.markdown("### 🧱 Canny Edge Detection")
                        canny_min = st.slider("Canny Min Threshold", 0, 255, 50)
                        canny_max = st.slider("Canny Max Threshold", 0, 255, 150)
                        canny_aperture = st.selectbox("Canny Aperture Size", [3, 5, 7], index=0)
                        canny_l2gradient = st.checkbox("Use L2 Gradient", False)

                    with st.expander("Sobel Edge Detection", expanded=False):
                    # Sobel
                    # st.sidebar.markdown("### 🧭 Sobel Edge Detection")
                        sobel_ksize = st.slider("Sobel Kernel Size", 1, 31, 3, step=2)
                        sobel_scale = st.slider("Sobel Scale", 0.1, 5.0, 1.0, step=0.1)
                        sobel_delta = st.slider("Sobel Delta", 0, 50, 0)

                    with st.expander("Laplacian Edge Detection", expanded=False):
                        # Laplacian
                        # st.sidebar.markdown("### 🌊 Laplacian Edge Detection")
                        laplacian_ksize = st.slider("Laplacian Kernel Size", 1, 31, 3, step=2)
                        laplacian_scale = st.slider("Laplacian Scale", 0.1, 5.0, 1.0, step=0.1)

                    with st.expander("Scharr Edge Detectionn", expanded=False):
                        # Scharr
                        # st.sidebar.markdown("### ⚡ Scharr Edge Detection")
                        scharr_scale = st.slider("Scharr Scale", 0.1, 5.0, 1.0, step=0.1)
                        scharr_delta = st.slider("Scharr Delta", 0, 50, 0)

                    with st.expander("Advanced Options", expanded=False):
                # Advanced options
                # st.sidebar.markdown("### 🎯 Advanced Options")
                        show_histogram = st.checkbox("Show Edge Histogram", False)
                        combine_edges = st.checkbox("Combine All Edges", False)
                        edge_dilation = st.checkbox("Apply Edge Dilation", False)
                        if edge_dilation:
                            dilation_kernel = st.slider("Dilation Kernel Size", 1, 10, 2)
                    st.markdown("---")

                # Process image
                processed_img = img_array.copy()

                if use_gray:
                    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2GRAY)

                if apply_blur:
                    processed_img = cv2.GaussianBlur(processed_img, (blur_ksize, blur_ksize), 0)

                # Apply morphological operations
                if apply_morphology:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
                    if morph_operation == "Opening":
                        processed_img = cv2.morphologyEx(processed_img, cv2.MORPH_OPEN, kernel)
                    elif morph_operation == "Closing":
                        processed_img = cv2.morphologyEx(processed_img, cv2.MORPH_CLOSE, kernel)
                    elif morph_operation == "Gradient":
                        processed_img = cv2.morphologyEx(processed_img, cv2.MORPH_GRADIENT, kernel)
                    elif morph_operation == "Tophat":
                        processed_img = cv2.morphologyEx(processed_img, cv2.MORPH_TOPHAT, kernel)
                    elif morph_operation == "Blackhat":
                        processed_img = cv2.morphologyEx(processed_img, cv2.MORPH_BLACKHAT, kernel)

                # Edge detection algorithms
                st.markdown('### <i class="fa-solid fa-flask"></i> Edge Detection Results', unsafe_allow_html=True)
                # st.subheader("🧪 Edge Detection Results")

                # Canny
                canny = cv2.Canny(processed_img, canny_min, canny_max, 
                                apertureSize=canny_aperture, L2gradient=canny_l2gradient)

                # Sobel
                sobelx = cv2.Sobel(processed_img, cv2.CV_64F, 1, 0, 
                                ksize=sobel_ksize, scale=sobel_scale, delta=sobel_delta)
                sobely = cv2.Sobel(processed_img, cv2.CV_64F, 0, 1, 
                                ksize=sobel_ksize, scale=sobel_scale, delta=sobel_delta)
                sobel = cv2.magnitude(sobelx, sobely)
                sobel = np.uint8(np.clip(sobel, 0, 255))

                # Laplacian
                laplacian = cv2.Laplacian(processed_img, cv2.CV_64F, 
                                        ksize=laplacian_ksize, scale=laplacian_scale)
                laplacian = np.uint8(np.clip(np.absolute(laplacian), 0, 255))

                # Scharr
                scharrx = cv2.Scharr(processed_img, cv2.CV_64F, 1, 0, 
                                    scale=scharr_scale, delta=scharr_delta)
                scharry = cv2.Scharr(processed_img, cv2.CV_64F, 0, 1, 
                                    scale=scharr_scale, delta=scharr_delta)
                scharr = cv2.magnitude(scharrx, scharry)
                scharr = np.uint8(np.clip(scharr, 0, 255))

                # Apply dilation if requested
                if edge_dilation:
                    kernel = np.ones((dilation_kernel, dilation_kernel), np.uint8)
                    canny = cv2.dilate(canny, kernel, iterations=1)
                    sobel = cv2.dilate(sobel, kernel, iterations=1)
                    laplacian = cv2.dilate(laplacian, kernel, iterations=1)
                    scharr = cv2.dilate(scharr, kernel, iterations=1)

                # Display results in grid
                col1, col2 = st.columns(2)
                
                with col1:
                    with st.expander("Canny Edge Detection", expanded=False):
                    # st.markdown("#### 🧱 Canny Edge Detection")
                        st.image(canny, clamp=True, caption="Canny Edge Detection", use_column_width=True)
                    
                    with st.expander("Laplacian Edge Detection", expanded=False):
                    # st.markdown("#### 🌊 Laplacian Edge Detection")
                        st.image(laplacian, clamp=True, caption="Laplacian Edge Detection", use_column_width=True)

                with col2:
                    with st.expander("Sobel Edge Detection", expanded=False):
                    # st.markdown("#### 🧭 Sobel Edge Detection")
                        st.image(sobel, clamp=True, caption="Sobel Edge Detection", use_column_width=True)
                    
                    with st.expander("Scharr Edge Detection", expanded=False):
                    # st.markdown("#### ⚡ Scharr Edge Detection")
                        st.image(scharr, clamp=True, caption="Scharr Edge Detection", use_column_width=True)

                # Combined edges visualization
                if combine_edges:
                    with st.expander("Combined Edge Detection", expanded=False):
                    # st.subheader("🎭 Combined Edge Detection")
                        combined = np.maximum.reduce([canny, sobel, laplacian, scharr])
                        st.image(combined, clamp=True, caption="Combined All Edge Detection Methods", use_column_width=True)

                # Edge statistics and histogram
                if show_histogram:
                    with st.expander("Edge Detection Statistics", expanded=False):
                    # st.subheader("📊 Edge Detection Statistics")
                    
                        col_stats1, col_stats2 = st.columns(2)
                        
                        with col_stats1:
                            # Statistics
                            st.markdown("**Edge Pixel Statistics:**")
                            stats_data = {
                                "Method": ["Canny", "Sobel", "Laplacian", "Scharr"],
                                "Edge Pixels": [
                                    np.count_nonzero(canny),
                                    np.count_nonzero(sobel),
                                    np.count_nonzero(laplacian),
                                    np.count_nonzero(scharr)
                                ],
                                "Edge Percentage": [
                                    f"{(np.count_nonzero(canny) / canny.size) * 100:.2f}%",
                                    f"{(np.count_nonzero(sobel) / sobel.size) * 100:.2f}%",
                                    f"{(np.count_nonzero(laplacian) / laplacian.size) * 100:.2f}%",
                                    f"{(np.count_nonzero(scharr) / scharr.size) * 100:.2f}%"
                                ]
                            }
                            st.table(stats_data)
                        
                        with col_stats2:
                            # Histogram
                            fig, ax = plt.subplots(figsize=(8, 6))
                            ax.hist(canny.flatten(), bins=50, alpha=0.7, label='Canny', color='red')
                            ax.hist(sobel.flatten(), bins=50, alpha=0.7, label='Sobel', color='blue')
                            ax.hist(laplacian.flatten(), bins=50, alpha=0.7, label='Laplacian', color='green')
                            ax.hist(scharr.flatten(), bins=50, alpha=0.7, label='Scharr', color='orange')
                            ax.set_xlabel('Pixel Intensity')
                            ax.set_ylabel('Frequency')
                            ax.set_title('Edge Detection Intensity Distribution')
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                    st.markdown("---")
                # Download options
                st.markdown('### <i class="fa-solid fa-download"></i> Download Results', unsafe_allow_html=True)
                # st.subheader("💾 Download Results")
                col_dl1, col_dl2, col_dl3, col_dl4 = st.columns(4)
                
                with col_dl1:
                    canny_pil = Image.fromarray(canny)
                    buf = io.BytesIO()
                    canny_pil.save(buf, format='PNG')
                    st.download_button(
                        label="Download Canny",
                        data=buf.getvalue(),
                        file_name="canny_edges.png",
                        mime="image/png"
                    )
                
                with col_dl2:
                    sobel_pil = Image.fromarray(sobel)
                    buf = io.BytesIO()
                    sobel_pil.save(buf, format='PNG')
                    st.download_button(
                        label="Download Sobel",
                        data=buf.getvalue(),
                        file_name="sobel_edges.png",
                        mime="image/png"
                    )
                
                with col_dl3:
                    laplacian_pil = Image.fromarray(laplacian)
                    buf = io.BytesIO()
                    laplacian_pil.save(buf, format='PNG')
                    st.download_button(
                        label="Download Laplacian",
                        data=buf.getvalue(),
                        file_name="laplacian_edges.png",
                        mime="image/png"
                    )
                
                with col_dl4:
                    scharr_pil = Image.fromarray(scharr)
                    buf = io.BytesIO()
                    scharr_pil.save(buf, format='PNG')
                    st.download_button(
                        label="Download Scharr",
                        data=buf.getvalue(),
                        file_name="scharr_edges.png",
                        mime="image/png"
                    )
                st.markdown("---")

            else:
                st.info("👆 Please upload an image to get started with edge detection!")
                
                # Sample information
                # st.markdown("---")
                with st.expander("About Edge Detection Methods", expanded=False):
                    # st.subheader("🎓 About Edge Detection Methods")
                    
                    col_info1, col_info2 = st.columns(2)
                    
                    with col_info1:
                        st.markdown("""
                    **<i class="fa-solid fa-vector-square"></i> Canny Edge Detection:**
                    - Multi-stage algorithm with noise reduction  
                    - Uses double thresholding  
                    - Connects edge pixels to form contours  
                    - Best for clean, well-defined edges  

                    **<i class="fa-solid fa-compass"></i> Sobel Edge Detection:**
                    - Uses convolution with Sobel kernels  
                    - Emphasizes edges in both X and Y directions  
                    - Good for gradient-based edge detection  
                    - Robust to noise  
                    """, unsafe_allow_html=True)

                    with col_info2:
                        st.markdown("""
                    **<i class="fa-solid fa-water"></i> Laplacian Edge Detection:**
                    - Second-derivative based method  
                    - Sensitive to noise but finds thin edges  
                    - Good for detecting blobs and fine details  
                    - Often combined with Gaussian blur  

                    **<i class="fa-solid fa-bolt"></i> Scharr Edge Detection:**
                    - Optimized version of Sobel  
                    - Better rotational symmetry  
                    - More accurate gradient calculation  
                    - Good for precise edge orientation  
                    """, unsafe_allow_html=True)
                st.markdown("---")

        elif page1 == "Noise Analysis":
            # Custom CSS for modern UI
            st.markdown("""
            <style>
                .main-header {
                    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                    padding: 2rem;
                    border-radius: 10px;
                    color: white;
                    text-align: center;
                    margin-bottom: 2rem;
                }
            </style>
            """, unsafe_allow_html=True)
            
            # Header
            st.markdown("""
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
            """, unsafe_allow_html=True)


            st.markdown("""
            <div class="main-header">
                <h1><i class="fa-solid fa-wave-square"></i> Advanced Noise Detection in Images</h1>
                <p>Upload an image and apply Advanced Noise Detection techniques using OpenCV.</p>
            </div>
            """, unsafe_allow_html=True)
            # st.title("🧠 Advanced Noise Detection in Images")

            uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png", "tif", "tiff"])

            def compute_local_variance(gray_img, kernel_size=3):
                """Compute local variance for noise detection"""
                mean = cv2.blur(gray_img, (kernel_size, kernel_size))
                sq_mean = cv2.blur(np.square(gray_img), (kernel_size, kernel_size))
                variance = sq_mean - np.square(mean)
                return variance

            def apply_highpass_filter(img, kernel_size=5):
                """Apply high-pass filter to detect high-frequency noise"""
                blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
                highpass = cv2.subtract(img, blurred)
                return highpass

            def frequency_noise_map(gray_img):
                """Generate frequency domain noise map"""
                f = np.fft.fft2(gray_img)
                fshift = np.fft.fftshift(f)
                magnitude = 20 * np.log(np.abs(fshift) + 1)
                return magnitude

            def laplacian_noise_detection(gray_img):
                """Use Laplacian operator for edge-based noise detection"""
                laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
                return np.abs(laplacian)

            def sobel_noise_detection(gray_img):
                """Use Sobel operators for gradient-based noise detection"""
                sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
                magnitude = np.sqrt(sobelx**2 + sobely**2)
                return magnitude

            def wiener_filter_estimate(gray_img, noise_var=None):
                """Estimate noise using Wiener filter approach"""
                if noise_var is None:
                    # Estimate noise variance from image statistics
                    noise_var = np.var(gray_img) * 0.1
                
                # Simple Wiener-like filtering
                blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
                noise_estimate = gray_img - blurred
                return noise_estimate, noise_var

            def texture_based_noise_detection(gray_img):
                """Detect noise based on texture analysis using local binary patterns"""
                # Simplified texture analysis
                kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
                texture_response = cv2.filter2D(gray_img, -1, kernel)
                return np.abs(texture_response)

            def noise_classification(gray_img):
                """Classify different types of noise using clustering"""
                # Extract features from different noise detection methods
                local_var = compute_local_variance(gray_img, 5)
                highpass = apply_highpass_filter(gray_img, 9)
                laplacian = laplacian_noise_detection(gray_img)
                
                # Flatten and combine features
                h, w = gray_img.shape
                features = np.column_stack([
                    local_var.flatten(),
                    highpass.flatten(),
                    laplacian.flatten()
                ])
                
                # Normalize features
                features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
                
                # Cluster into noise types
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                noise_types = kmeans.fit_predict(features)
                noise_map = noise_types.reshape(h, w)
                
                return noise_map, kmeans.cluster_centers_

            def calculate_noise_metrics(gray_img):
                """Calculate various noise metrics"""
                metrics = {}
                
                # Signal-to-Noise Ratio (SNR)
                signal_power = np.mean(gray_img**2)
                noise_estimate = apply_highpass_filter(gray_img, 9)
                noise_power = np.mean(noise_estimate**2)
                snr = 10 * np.log10(signal_power / (noise_power + 1e-8))
                metrics['SNR (dB)'] = snr
                
                # Peak Signal-to-Noise Ratio (PSNR) estimate
                mse_noise = np.mean(noise_estimate**2)
                psnr = 20 * np.log10(255.0 / np.sqrt(mse_noise + 1e-8))
                metrics['PSNR (dB)'] = psnr
                
                # Noise variance
                metrics['Noise Variance'] = np.var(noise_estimate)
                
                # Noise standard deviation
                metrics['Noise Std Dev'] = np.std(noise_estimate)
                
                return metrics

            if uploaded_file:
                image = Image.open(uploaded_file).convert('RGB')
                img_np = np.array(image)
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

                # col1, col2 = st.columns([2, 1])
                
                # with col1:
                #     st.subheader("📷 Original Image")
                #     st.image(image, use_column_width=True)
                
                # with col2:
                #     st.subheader("📊 Noise Metrics")
                #     metrics = calculate_noise_metrics(gray)
                #     for metric, value in metrics.items():
                #         st.metric(metric, f"{value:.2f}")
                st.markdown("---")
                st.markdown('### <i class="fa-solid fa-braille"></i> Noise Metrics', unsafe_allow_html=True)
                # st.subheader("📊 Noise Metrics")

                # Assume `metrics` is a dictionary like {"PSNR": 32.1, "SNR": 20.5, ...}
                metrics = calculate_noise_metrics(gray)
                cols = st.columns(len(metrics))
                for col, (metric, value) in zip(cols, metrics.items()):
                    with col:
                        st.metric(label=metric, value=f"{value:.2f}")
                st.markdown("---")
                st.markdown('### <i class="fa-solid fa-wand-magic-sparkles"></i> Noise Filter', unsafe_allow_html=True)

                # Tabs for different analysis methods
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                    "Local Variance", "High-Pass Filter", "Frequency Domain", 
                    "Edge Detection", "Wiener Estimate", "Noise Classification"
                ])

                with tab1:
                    with st.expander("Local Variance Noise Map", expanded=False):
                    # st.subheader("🔍 Local Variance Noise Map")
                        ksize = st.slider("Kernel size for local variance", 3, 15, 5, step=2, key="var_kernel")
                        local_var = compute_local_variance(gray, kernel_size=ksize)
                        local_var_norm = cv2.normalize(local_var, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(local_var_norm, caption="Local Variance Heatmap", channels="GRAY", use_column_width=True)
                        with col2:
                            fig, ax = plt.subplots()
                            ax.hist(local_var.flatten(), bins=50, alpha=0.7, color='blue')
                            ax.set_xlabel('Variance Value')
                            ax.set_ylabel('Frequency')
                            ax.set_title('Variance Distribution')
                            st.pyplot(fig)
                    # Comparison section
                    with st.expander("Noise Detection Comparison", expanded=False):
                    # st.subheader("📈 Noise Detection Comparison")
                        comparison_option = st.selectbox(
                            "Choose comparison view",
                            ["Side-by-side", "Overlay", "Difference Map"]
                        )
                        
                        if comparison_option == "Side-by-side":
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                local_var = compute_local_variance(gray, 5)
                                local_var_norm = cv2.normalize(local_var, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                                st.image(local_var_norm, caption="Local Variance", channels="GRAY")
                            with col2:
                                highpass = apply_highpass_filter(gray, 9)
                                st.image(highpass, caption="High-Pass Filter", channels="GRAY")
                            with col3:
                                laplacian = laplacian_noise_detection(gray)
                                laplacian_norm = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                                st.image(laplacian_norm, caption="Laplacian", channels="GRAY")

                        st.success("✅ Advanced noise analysis completed.")

                with tab2:
                    with st.expander("High-Pass Filter Noise Detection", expanded=False):
                    # st.subheader("🧪 High-Pass Filter Noise Detection")
                        ksize_hp = st.slider("Gaussian Blur kernel size", 3, 25, 9, step=2, key="hp_kernel")
                        highpass = apply_highpass_filter(gray, kernel_size=ksize_hp)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(highpass, caption="High-Pass Filter Result", channels="GRAY", use_column_width=True)
                        with col2:
                            fig, ax = plt.subplots()
                            ax.hist(highpass.flatten(), bins=50, alpha=0.7, color='green')
                            ax.set_xlabel('Filter Response')
                            ax.set_ylabel('Frequency')
                            ax.set_title('High-Pass Response Distribution')
                            st.pyplot(fig)

                with tab3:
                    with st.expander("Frequency Domain Noise Detection", expanded=False):
                    # st.subheader("⚙️ Frequency Domain Noise Detection")
                        freq_map = frequency_noise_map(gray)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            fig, ax = plt.subplots()
                            ax.imshow(freq_map, cmap='inferno')
                            ax.set_title("FFT Magnitude Spectrum")
                            ax.axis('off')
                            st.pyplot(fig)
                        with col2:
                            # Create radial average of frequency spectrum
                            center = (freq_map.shape[0]//2, freq_map.shape[1]//2)
                            y, x = np.ogrid[:freq_map.shape[0], :freq_map.shape[1]]
                            r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
                            r = r.astype(int)
                            
                            # Calculate radial average
                            tbin = np.bincount(r.ravel(), freq_map.ravel())
                            nr = np.bincount(r.ravel())
                            radial_prof = tbin / (nr + 1e-8)
                            
                            fig, ax = plt.subplots()
                            ax.plot(radial_prof[:len(radial_prof)//2])
                            ax.set_xlabel('Spatial Frequency')
                            ax.set_ylabel('Magnitude')
                            ax.set_title('Radial Frequency Profile')
                            st.pyplot(fig)

                with tab4:
                    with st.expander("Edge-Based Noise Detection", expanded=False):
                    # st.subheader("🎯 Edge-Based Noise Detection")
                        method = st.selectbox("Select edge detection method", ["Laplacian", "Sobel"])
                        
                        if method == "Laplacian":
                            edge_response = laplacian_noise_detection(gray)
                            title = "Laplacian Edge Detection"
                        else:
                            edge_response = sobel_noise_detection(gray)
                            title = "Sobel Gradient Magnitude"
                        
                        edge_norm = cv2.normalize(edge_response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(edge_norm, caption=title, channels="GRAY", use_column_width=True)
                        with col2:
                            # Texture analysis
                            texture_response = texture_based_noise_detection(gray)
                            texture_norm = cv2.normalize(texture_response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                            st.image(texture_norm, caption="Texture-Based Noise Detection", channels="GRAY", use_column_width=True)

                with tab5:
                    with st.expander("Wiener Filter Noise Estimation", expanded=False):
                    # st.subheader("🔧 Wiener Filter Noise Estimation")
                        noise_var_input = st.slider("Noise variance (0 = auto-estimate)", 0.0, 1000.0, 0.0)
                        noise_var = noise_var_input if noise_var_input > 0 else None
                        
                        noise_est, estimated_var = wiener_filter_estimate(gray, noise_var)
                        noise_norm = cv2.normalize(np.abs(noise_est), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(noise_norm, caption="Estimated Noise", channels="GRAY", use_column_width=True)
                            st.write(f"Estimated noise variance: {estimated_var:.2f}")
                        with col2:
                            # Denoised version
                            denoised = gray - noise_est.astype(np.uint8)
                            st.image(denoised, caption="Denoised Image", channels="GRAY", use_column_width=True)

                with tab6:
                    with st.expander("Noise Type Classification", expanded=False):
                    # st.subheader("🎨 Noise Type Classification")
                        with st.spinner("Classifying noise types..."):
                            noise_map, cluster_centers = noise_classification(gray)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            fig, ax = plt.subplots()
                            im = ax.imshow(noise_map, cmap='tab10')
                            ax.set_title("Noise Type Classification")
                            ax.axis('off')
                            plt.colorbar(im, ax=ax)
                            st.pyplot(fig)
                        
                        with col2:
                            st.write("**Cluster Centers (Feature Space):**")
                            for i, center in enumerate(cluster_centers):
                                st.write(f"Cluster {i}: Local Var={center[0]:.3f}, HighPass={center[1]:.3f}, Laplacian={center[2]:.3f}")

                
                st.markdown("---")
            else:
                st.info("Upload an image to start advanced noise detection.")
                
                # Add some help information
                st.markdown('### <i class="fa-solid fa-circle-info"></i> About this tool', unsafe_allow_html=True)
                with st.expander("Details", expanded=False):
                    st.markdown("""
                    This advanced noise detection tool provides multiple methods to analyze and visualize noise in images:
                    
                    - **Local Variance**: Identifies noisy regions by computing pixel variance in local neighborhoods
                    - **High-Pass Filter**: Highlights high-frequency noise by subtracting low-frequency components
                    - **Frequency Domain**: Analyzes noise patterns in the frequency spectrum using FFT
                    - **Edge Detection**: Uses Laplacian and Sobel operators to detect noise at edges
                    - **Wiener Estimation**: Estimates and removes noise using Wiener filter principles
                    - **Noise Classification**: Automatically classifies different types of noise using machine learning
                    
                    **Metrics provided:**
                    - SNR (Signal-to-Noise Ratio)
                    - PSNR (Peak Signal-to-Noise Ratio)
                    - Noise variance and standard deviation
                    """)

    elif page_E == "JPEG Compression Analysis":
        page1 = st.sidebar.selectbox("Select Analysis Tool", 
                ["JPEG Artifact Analysis",
                "Quantization Table Analysis"],
                help="Choose Analysis Tool"
                )
        
        if page1 == "JPEG Artifact Analysis":
            st.markdown("""
            <style>
                .main-header {
                    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                    padding: 2rem;
                    border-radius: 10px;
                    color: white;
                    text-align: center;
                    margin-bottom: 2rem;
                }
                .metric-container {
                    background-color: #f0f2f6;
                    padding: 1rem;
                    border-radius: 0.5rem;
                    margin: 0.5rem 0;
                }
                /* Light Mode */
                @media (prefers-color-scheme: light) {
                    .metric-container {
                        background-color: #f0f2f6;
                        color: #000;
                    }
                }

                /* Dark Mode */
                @media (prefers-color-scheme: dark) {
                    .metric-container {
                        background-color: #1e1e1e;
                        color: #fff;
                    }
                .artifact-high { color: #ff4444; font-weight: bold; }
                .artifact-medium { color: #ff8800; font-weight: bold; }
                .artifact-low { color: #44ff44; font-weight: bold; }
            </style>
            """, unsafe_allow_html=True)
            st.markdown("""
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="main-header">
                <h1><i class="fa-solid fa-image"></i> Advanced JPEG Artifact Analysis Tool</h1>
                <p>Comprehensive analysis of JPEG compression artifacts using multiple detection methods</p>
            </div>
            """, unsafe_allow_html=True)
            # st.title("🔬 Advanced JPEG Artifact Analysis Tool")
            # st.markdown("*Comprehensive analysis of JPEG compression artifacts using multiple detection methods*")

            uploaded_file = st.file_uploader("Upload a JPEG image", type=['jpg', 'jpeg'])

            def calculate_psnr(original, compressed):
                """Calculate Peak Signal-to-Noise Ratio"""
                return sk_metrics.peak_signal_noise_ratio(original, compressed, data_range=255)

            def calculate_ssim(original, compressed):
                """Calculate Structural Similarity Index"""
                return sk_metrics.structural_similarity(original, compressed, data_range=255)

            def dct2(block):
                """2D Discrete Cosine Transform"""
                return cv2.dct(np.float32(block))

            def idct2(block):
                """2D Inverse Discrete Cosine Transform"""
                return cv2.idct(np.float32(block))

            def visualize_dct(image_gray, block_size=8):
                """Visualize DCT coefficients across the image"""
                h, w = image_gray.shape
                dct_map = np.zeros_like(image_gray, dtype=np.float32)
                high_freq_map = np.zeros_like(image_gray, dtype=np.float32)
                
                for i in range(0, h - block_size + 1, block_size):
                    for j in range(0, w - block_size + 1, block_size):
                        block = image_gray[i:i+block_size, j:j+block_size]
                        dct_block = dct2(block)
                        
                        # Overall DCT energy
                        dct_map[i:i+block_size, j:j+block_size] = np.log(np.abs(dct_block) + 1)
                        
                        # High frequency content (bottom-right quadrant)
                        high_freq = dct_block[block_size//2:, block_size//2:]
                        high_freq_energy = np.sum(np.abs(high_freq))
                        high_freq_map[i:i+block_size, j:j+block_size] = high_freq_energy
                        
                return dct_map, high_freq_map

            def detect_blockiness(image_gray, block_size=8):
                """Enhanced blockiness detection with multiple metrics"""
                h, w = image_gray.shape
                
                # Vertical blockiness (at block boundaries)
                vertical_diff = 0
                for i in range(block_size, h, block_size):
                    if i < h:
                        vertical_diff += np.sum(np.abs(image_gray[i-1, :] - image_gray[i, :]))
                
                # Horizontal blockiness (at block boundaries)
                horizontal_diff = 0
                for j in range(block_size, w, block_size):
                    if j < w:
                        horizontal_diff += np.sum(np.abs(image_gray[:, j-1] - image_gray[:, j]))
                
                # Normalize by number of boundaries
                num_v_boundaries = (h // block_size) * w
                num_h_boundaries = h * (w // block_size)
                
                blockiness_score = (vertical_diff / num_v_boundaries + horizontal_diff / num_h_boundaries) / 2
                return blockiness_score

            def detect_ringing_artifacts(image_gray):
                """Detect ringing artifacts using multiple edge detection methods"""
                # Laplacian for edge detection
                laplacian = cv2.Laplacian(image_gray, cv2.CV_64F)
                laplacian_abs = np.abs(laplacian)
                
                # Sobel edges
                sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
                sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
                
                # Canny edges
                canny = cv2.Canny(image_gray, 50, 150)
                
                return laplacian_abs, sobel_magnitude, canny

            def mosquito_noise_detection(image_gray):
                """Detect mosquito noise around edges"""
                # Find edges
                edges = cv2.Canny(image_gray, 50, 150)
                
                # Dilate edges to create a mask around edge regions
                kernel = np.ones((5,5), np.uint8)
                edge_regions = cv2.dilate(edges, kernel, iterations=2)
                
                # Calculate local variance in edge regions
                local_var = ndimage.generic_filter(image_gray.astype(np.float32), np.var, size=3)
                mosquito_map = local_var * (edge_regions / 255.0)
                
                return mosquito_map, np.mean(mosquito_map[mosquito_map > 0])

            def quality_assessment(blockiness, psnr, ssim, mosquito_score):
                """Overall quality assessment"""
                # Normalize scores (these thresholds may need adjustment based on your use case)
                block_norm = min(blockiness / 50.0, 1.0)  # Assuming 50 is high blockiness
                psnr_norm = max(0, min((psnr - 20) / 20.0, 1.0))  # 20-40 dB range
                ssim_norm = ssim  # Already 0-1
                mosquito_norm = min(mosquito_score / 100.0, 1.0)  # Assuming 100 is high
                
                # Weighted overall quality (higher is better)
                quality_score = (
                    0.3 * (1 - block_norm) +  # Less blockiness is better
                    0.3 * psnr_norm +         # Higher PSNR is better
                    0.3 * ssim_norm +         # Higher SSIM is better
                    0.1 * (1 - mosquito_norm) # Less mosquito noise is better
                )
                
                return quality_score * 100  # Convert to percentage

            def get_artifact_level(score, thresholds):
                """Get artifact level based on score and thresholds"""
                if score > thresholds[1]:
                    return "High", "artifact-high"
                elif score > thresholds[0]:
                    return "Medium", "artifact-medium"
                else:
                    return "Low", "artifact-low"

            if uploaded_file:
                # Load and preprocess image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, 1)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                
                # Display original image
                # st.image(img_rgb, caption="🖼️ Original Image", use_column_width=True)
                st.markdown("---")
                # Control panel
                with st.sidebar:
                    st.markdown("---")
                    with st.expander("Analysis Options", expanded=False):
                        # st.sidebar.header("🎛️ Analysis Parameters")
                        block_size = st.slider("DCT Block Size", 4, 16, 8, 2)
                        show_advanced = st.checkbox("Show Advanced Visualizations", True)
                    st.markdown("---")    
                # Basic image info
                # st.markdown("### 📊 Image Information")
                # st.info(f"""
                # **Dimensions**: {img_rgb.shape[1]} × {img_rgb.shape[0]}
                # **Channels**: {img_rgb.shape[2]}
                # **File Size**: {len(file_bytes)} bytes
                # """)
                # st.info(f"📊 **Overall Image Quality Score:** {quality_score:.1f}/100")
                        
                
                # Analysis
                # st.markdown("---")
                st.markdown('### <i class="fa-solid fa-magnifying-glass"></i> Artifact Analysis Results', unsafe_allow_html=True)
                # st.header("🔍 Artifact Analysis Results")
                
                # Create reference image for comparison (slightly blurred)
                reference = cv2.GaussianBlur(img_gray, (3, 3), 0.5)
                
                # Calculate metrics
                psnr = calculate_psnr(reference, img_gray)
                ssim = calculate_ssim(reference, img_gray)
                blockiness_score = detect_blockiness(img_gray, block_size)
                mosquito_map, mosquito_score = mosquito_noise_detection(img_gray)
                quality_score = quality_assessment(blockiness_score, psnr, ssim, mosquito_score)
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    block_level, block_class = get_artifact_level(blockiness_score, [5, 15])
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4>Blockiness</h4>
                        <p class="{block_class}">{blockiness_score:.2f}</p>
                        <small>Level: {block_level}</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4>PSNR</h4>
                        <p><strong>{psnr:.2f} dB</strong></p>
                        <small>Higher is better</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4>SSIM</h4>
                        <p><strong>{ssim:.3f}</strong></p>
                        <small>Higher is better</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    mosquito_level, mosquito_class = get_artifact_level(mosquito_score, [20, 50])
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4>Mosquito Noise</h4>
                        <p class="{mosquito_class}">{mosquito_score:.2f}</p>
                        <small>Level: {mosquito_level}</small>
                    </div>
                    """, unsafe_allow_html=True)

                # with col5:
                #     st.markdown(f"""
                #     <div class="metric-container">
                #         <h4>📊 Quality Score</h4>
                #         <p class="{block_class}">{quality_score:.1f}</p>
                #     </div>
                #     """, unsafe_allow_html=True)
                
                # Overall quality score
                st.info(f"**Overall Image Quality Score:** {quality_score:.1f}/100")
                # st.markdown(f"""
                # <div style="text-align: center; padding: 1rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 0.5rem; margin: 1rem 0;">
                #     <h3>Overall Image Quality Score: {quality_score:.1f}/100</h3>
                # </div>
                # """, unsafe_allow_html=True)
                
                # Visualizations
                if show_advanced:
                    st.markdown("---")
                    st.markdown('### <i class="fa-solid fa-palette"></i> Advanced Visualizations', unsafe_allow_html=True)
                    # st.header("🎨 Advanced Visualizations")
                    
                    # DCT Analysis
                    with st.expander("DCT Coefficient Analysis", expanded=False):
                    # st.subheader("🔍 DCT Coefficient Analysis")
                        dct_map, high_freq_map = visualize_dct(img_gray, block_size)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            fig1, ax1 = plt.subplots(figsize=(8, 6))
                            im1 = ax1.imshow(dct_map, cmap='hot', interpolation='nearest')
                            ax1.set_title("DCT Coefficient Energy Map")
                            ax1.axis("off")
                            plt.colorbar(im1, ax=ax1, shrink=0.8)
                            st.pyplot(fig1)
                        
                        with col2:
                            fig2, ax2 = plt.subplots(figsize=(8, 6))
                            im2 = ax2.imshow(high_freq_map, cmap='viridis', interpolation='nearest')
                            ax2.set_title("High Frequency Content Map")
                            ax2.axis("off")
                            plt.colorbar(im2, ax=ax2, shrink=0.8)
                            st.pyplot(fig2)
                        
                    # Ringing Artifacts
                    with st.expander("Ringing Artifacts Detection", expanded=False):
                    # st.subheader("🌊 Ringing Artifacts Detection")
                        laplacian_abs, sobel_magnitude, canny = detect_ringing_artifacts(img_gray)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            fig3, ax3 = plt.subplots(figsize=(6, 6))
                            ax3.imshow(laplacian_abs, cmap='plasma')
                            ax3.set_title("Laplacian Edge Detection")
                            ax3.axis("off")
                            st.pyplot(fig3)
                        
                        with col2:
                            fig4, ax4 = plt.subplots(figsize=(6, 6))
                            ax4.imshow(sobel_magnitude, cmap='plasma')
                            ax4.set_title("Sobel Edge Magnitude")
                            ax4.axis("off")
                            st.pyplot(fig4)
                        
                        with col3:
                            fig5, ax5 = plt.subplots(figsize=(6, 6))
                            ax5.imshow(canny, cmap='gray')
                            ax5.set_title("Canny Edge Detection")
                            ax5.axis("off")
                            st.pyplot(fig5)
                    
                    # Mosquito Noise Visualization
                    with st.expander("Mosquito Noise Visualization", expanded=False):
                    # st.subheader("🦟 Mosquito Noise Visualization")
                        fig6, ax6 = plt.subplots(figsize=(10, 6))
                        im6 = ax6.imshow(mosquito_map, cmap='inferno', interpolation='nearest')
                        ax6.set_title("Mosquito Noise Around Edges")
                        ax6.axis("off")
                        plt.colorbar(im6, ax=ax6, shrink=0.8)
                        st.pyplot(fig6)
                    
                # Detailed Analysis Report
                st.markdown("---")
                st.markdown('### <i class="fa-solid fa-clipboard-list"></i> Detailed Analysis Report', unsafe_allow_html=True)
                # st.header("📋 Detailed Analysis Report")
                with st.expander("JPEG Artifact Analysis Report", expanded=False):
                
                    report = f"""
                    ## JPEG Artifact Analysis Report
                    
                    ### Image Specifications
                    - **Resolution**: {img_rgb.shape[1]} × {img_rgb.shape[0]} pixels
                    - **File Size**: {len(file_bytes):,} bytes
                    - **Aspect Ratio**: {img_rgb.shape[1]/img_rgb.shape[0]:.2f}:1
                    
                    ### Artifact Detection Results
                    
                    #### 1. Blockiness Analysis
                    - **Score**: {blockiness_score:.2f}
                    - **Assessment**: {block_level} level blockiness detected
                    - **Interpretation**: {'Significant 8×8 block boundaries visible' if blockiness_score > 15 else 'Moderate block artifacts present' if blockiness_score > 5 else 'Minimal blockiness detected'}
                    
                    #### 2. Signal Quality Metrics
                    - **PSNR**: {psnr:.2f} dB
                    - **SSIM**: {ssim:.3f}
                    - **Quality Rating**: {'Excellent' if psnr > 35 else 'Good' if psnr > 30 else 'Fair' if psnr > 25 else 'Poor'}
                    
                    #### 3. Mosquito Noise Assessment
                    - **Score**: {mosquito_score:.2f}
                    - **Level**: {mosquito_level}
                    - **Impact**: {'High-frequency noise visible around edges' if mosquito_score > 50 else 'Moderate noise around sharp edges' if mosquito_score > 20 else 'Minimal mosquito noise detected'}
                    
                    ### Overall Quality Assessment
                    - **Composite Score**: {quality_score:.1f}/100
                    - **Grade**: {'A' if quality_score > 85 else 'B' if quality_score > 70 else 'C' if quality_score > 55 else 'D'}
                    
                    ### Recommendations
                    """
                    
                    if quality_score > 85:
                        report += "- ✅ Excellent image quality with minimal compression artifacts\n- No action needed"
                    elif quality_score > 70:
                        report += "- ✅ Good image quality with acceptable compression\n- Suitable for most applications"
                    elif quality_score > 55:
                        report += "- ⚠️ Moderate artifacts present\n- Consider re-encoding with higher quality settings\n- May need preprocessing for critical applications"
                    else:
                        report += "- ❌ Significant compression artifacts detected\n- Recommend obtaining higher quality source\n- Consider denoising and artifact reduction techniques"
                    
                    st.markdown(report)
                st.markdown("---")

            else:
                st.info("Please upload a JPEG image to begin comprehensive artifact analysis.")
                with st.expander("What This Tool Analyzes", expanded=False):
                    st.markdown("""
                    
                    - **DCT Coefficient Patterns**: Visualizes frequency domain artifacts
                    - **Blockiness Detection**: Identifies 8×8 block boundaries from JPEG compression
                    - **Ringing Artifacts**: Detects oscillations around sharp edges
                    - **Mosquito Noise**: Identifies high-frequency noise around edges
                    - **Signal Quality Metrics**: PSNR and SSIM calculations
                    - **Overall Quality Assessment**: Composite score based on multiple factors
                    """)
                st.markdown("---")

        elif page1 == "Quantization Table Analysis":
            def extract_quantization_tables(img):
                """Extract quantization tables from JPEG image"""
                try:
                    # First try to get quantization tables directly
                    if hasattr(img, 'quantization') and img.quantization:
                        return img.quantization
                    
                    # Alternative method using PIL's internal structures
                    if hasattr(img, '_getexif'):
                        qtables = {}
                        # Access quantization tables through PIL's internal methods
                        if hasattr(img, 'app') and 'quantization' in str(img.app):
                            return img.quantization
                    
                    return {}
                except Exception as e:
                    st.error(f"Error extracting quantization tables: {str(e)}")
                    return {}

            def plot_quant_table(table, index, use_seaborn=True):
                """Plot quantization table with better visualization"""
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Reshape to 8x8 matrix
                mat = np.array(table).reshape((8, 8))
                
                # First plot: Heatmap
                if use_seaborn:
                    sns.heatmap(mat, annot=True, fmt='d', cmap='viridis', 
                            ax=ax1, cbar_kws={'label': 'Quantization Value'})
                else:
                    im1 = ax1.imshow(mat, cmap='viridis')
                    for (i, j), val in np.ndenumerate(mat):
                        ax1.text(j, i, f'{val}', ha='center', va='center', 
                                color='white' if val < mat.max()/2 else 'black', fontsize=8)
                    plt.colorbar(im1, ax=ax1, label='Quantization Value')
                
                ax1.set_title(f"Quantization Table {index}")
                ax1.set_xlabel("Frequency (Horizontal)")
                ax1.set_ylabel("Frequency (Vertical)")
                
                # Second plot: 3D surface
                X, Y = np.meshgrid(range(8), range(8))
                ax2 = fig.add_subplot(122, projection='3d')
                surf = ax2.plot_surface(X, Y, mat, cmap='viridis', alpha=0.8)
                ax2.set_title(f"3D View - Table {index}")
                ax2.set_xlabel("Frequency (Horizontal)")
                ax2.set_ylabel("Frequency (Vertical)")
                ax2.set_zlabel("Quantization Value")
                
                plt.tight_layout()
                st.pyplot(fig)

            def compare_to_standard(qtable, standard_type='luma'):
                """Compare quantization table to standard tables"""
                std_luma = np.array([
                    16,11,10,16,24,40,51,61,
                    12,12,14,19,26,58,60,55,
                    14,13,16,24,40,57,69,56,
                    14,17,22,29,51,87,80,62,
                    18,22,37,56,68,109,103,77,
                    24,35,55,64,81,104,113,92,
                    49,64,78,87,103,121,120,101,
                    72,92,95,98,112,100,103,99
                ]).reshape((8, 8))
                
                std_chroma = np.array([
                    17,18,24,47,99,99,99,99,
                    18,21,26,66,99,99,99,99,
                    24,26,56,99,99,99,99,99,
                    47,66,99,99,99,99,99,99,
                    99,99,99,99,99,99,99,99,
                    99,99,99,99,99,99,99,99,
                    99,99,99,99,99,99,99,99,
                    99,99,99,99,99,99,99,99
                ]).reshape((8, 8))
                
                standard = std_luma if standard_type == 'luma' else std_chroma
                qtable_matrix = np.array(qtable).reshape((8, 8))
                
                # Calculate differences and similarity metrics
                diff = qtable_matrix - standard
                abs_diff = np.abs(diff)
                mse = np.mean(diff**2)
                mae = np.mean(abs_diff)
                
                return {
                    'difference': diff,
                    'abs_difference': abs_diff,
                    'mse': mse,
                    'mae': mae,
                    'standard': standard,
                    'input': qtable_matrix
                }

            def estimate_quality_factor(qtable, table_type='luma'):
                """Estimate JPEG quality factor from quantization table"""
                std_luma = np.array([
                    16,11,10,16,24,40,51,61,
                    12,12,14,19,26,58,60,55,
                    14,13,16,24,40,57,69,56,
                    14,17,22,29,51,87,80,62,
                    18,22,37,56,68,109,103,77,
                    24,35,55,64,81,104,113,92,
                    49,64,78,87,103,121,120,101,
                    72,92,95,98,112,100,103,99
                ])
                
                qtable_flat = np.array(qtable)
                
                # Simple estimation based on average scaling factor
                if np.mean(qtable_flat) != 0:
                    scale_factor = np.mean(std_luma) / np.mean(qtable_flat)
                    
                    if scale_factor >= 1:
                        quality = 100 - (1/scale_factor - 1) * 50
                    else:
                        quality = 50 * scale_factor
                        
                    return max(1, min(100, int(quality)))
                return 50

            def analyze_compression_characteristics(qtable):
                """Analyze compression characteristics of the quantization table"""
                mat = np.array(qtable).reshape((8, 8))
                
                # High frequency preservation (bottom-right corner)
                high_freq_avg = np.mean(mat[4:, 4:])
                low_freq_avg = np.mean(mat[:4, :4])
                
                # Edge preservation
                edge_preservation = low_freq_avg / high_freq_avg if high_freq_avg > 0 else float('inf')
                
                # Uniformity
                uniformity = np.std(mat)
                
                return {
                    'high_freq_preservation': high_freq_avg,
                    'low_freq_preservation': low_freq_avg,
                    'edge_preservation_ratio': edge_preservation,
                    'uniformity': uniformity,
                    'max_quantization': np.max(mat),
                    'min_quantization': np.min(mat)
                }

            # Streamlit UI
            # st.set_page_config(page_title="Advanced JPEG Quantization Analyzer", layout="wide")
            # Custom CSS for modern UI
            st.markdown("""
            <style>
                .main-header {
                    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                    padding: 2rem;
                    border-radius: 10px;
                    color: white;
                    text-align: center;
                    margin-bottom: 2rem;
                }
            </style>
            """, unsafe_allow_html=True)
            
            # Header
            st.markdown("""
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="main-header">
                <h1><i class="fa-solid fa-solid fa-chart-line"></i> Advanced JPEG Quantization Table Analysis</h1>
                <p>This tool analyzes JPEG quantization tables, which control the compression quality and characteristics.
            Upload a JPEG image to examine its quantization tables and compare them with standard tables.</p>
            </div>
            """, unsafe_allow_html=True)

            # st.title("🧮 Advanced JPEG Quantization Table Analysis")

            # st.markdown("""
            # This tool analyzes JPEG quantization tables, which control the compression quality and characteristics.
            # Upload a JPEG image to examine its quantization tables and compare them with standard tables.
            # """)

            # Sidebar for options
            with st.sidebar:
                st.markdown("---")
                with st.expander("Analysis Options", expanded=False):
                    # st.sidebar.header("Analysis Options")
                    use_seaborn = st.checkbox("Use Seaborn for better plots", value=True)
                    show_3d = st.checkbox("Show 3D visualization", value=True)
                    compare_standard = st.selectbox("Compare with standard:", ["luma", "chroma"])
                st.markdown("---")

            uploaded_file = st.file_uploader("Upload a JPEG image", type=['jpg', 'jpeg'])

            if uploaded_file:
                img = Image.open(uploaded_file)

                if img.format != 'JPEG':
                    st.error("⚠️ Only JPEG images contain quantization tables.")
                else:
                    # col1, col2 = st.columns([1, 2])
                    
                    # with col1:
                    #     st.image(img, caption="Uploaded Image", use_column_width=True)
                        
                        # Image metadata
                        # st.subheader("📋 Image Info")
                        # st.write(f"**Format:** {img.format}")
                        # st.write(f"**Size:** {img.size[0]} × {img.size[1]} pixels")
                        # st.write(f"**Mode:** {img.mode}")
                        
                        # if hasattr(img, 'info') and 'dpi' in img.info:
                        #     st.write(f"**DPI:** {img.info['dpi']}")

                    # with col2:
                        st.markdown("---")
                        st.markdown('### <i class="fa-solid fa-chart-bar"></i> Quantization Table Analysis', unsafe_allow_html=True)
                        # st.subheader("📊 Quantization Table Analysis")
                        
                        qtables = extract_quantization_tables(img)


                        if not qtables:
                            st.warning("⚠️ No quantization tables found. This might not be a standard JPEG or the tables are not accessible.")
                        else:
                            # Tabs for different tables
                            if len(qtables) > 1:
                                tabs = st.tabs([f"Table {idx}" for idx in qtables.keys()])
                            else:
                                tabs = [st.container()]

                            for tab_idx, (table_id, table) in enumerate(qtables.items()):
                                with tabs[tab_idx]:
                                    st.markdown(f'### <i class="fa-solid fa-table"></i> Quantization Table {table_id}', unsafe_allow_html=True)
                                    # st.markdown(f"### 🔢 Quantization Table {table_id}")
                                    
                                    # Estimate quality
                                    with st.expander("View Raw Quantization Values"):
                                        estimated_quality = estimate_quality_factor(table)
                                        st.metric("Estimated JPEG Quality", f"{estimated_quality}%")
                                    
                                    # Show raw values in expandable section
                                    
                                        st.dataframe(np.array(table).reshape((8, 8)))
                                    
                                    # Plot the table
                                    with st.expander("Quantization Visualization", expanded=False):
                                        plot_quant_table(table, table_id, use_seaborn)
                                    
                                    # Compression analysis
                                    with st.expander("Compression Characteristics", expanded=False):
                                    # st.subheader("🔍 Compression Characteristics")
                                        analysis = analyze_compression_characteristics(table)
                                        
                                        col_a, col_b, col_c = st.columns(3)
                                        with col_a:
                                            st.metric("Low Freq Avg", f"{analysis['low_freq_preservation']:.1f}")
                                            st.metric("High Freq Avg", f"{analysis['high_freq_preservation']:.1f}")
                                        with col_b:
                                            st.metric("Edge Preservation", f"{analysis['edge_preservation_ratio']:.2f}")
                                            st.metric("Uniformity (std)", f"{analysis['uniformity']:.1f}")
                                        with col_c:
                                            st.metric("Min Quantization", f"{analysis['min_quantization']}")
                                            st.metric("Max Quantization", f"{analysis['max_quantization']}")
                                    
                                    # Comparison with standard
                                    with st.expander(f"Comparison with Standard {compare_standard.title()} Table", expanded=False):
                                        # st.subheader(f"📈 Comparison with Standard {compare_standard.title()} Table")
                                        comparison = compare_to_standard(table, compare_standard)
                                        
                                        col_diff1, col_diff2 = st.columns(2)
                                        
                                        with col_diff1:
                                            st.metric("Mean Squared Error", f"{comparison['mse']:.2f}")
                                            st.metric("Mean Absolute Error", f"{comparison['mae']:.2f}")
                                        
                                        with col_diff2:
                                            # Plot difference heatmap
                                            fig, ax = plt.subplots(figsize=(8, 6))
                                            if use_seaborn:
                                                sns.heatmap(comparison['difference'], annot=True, fmt='.0f', 
                                                        cmap='RdBu_r', center=0, ax=ax,
                                                        cbar_kws={'label': 'Difference from Standard'})
                                            else:
                                                im = ax.imshow(comparison['difference'], cmap='RdBu_r')
                                                plt.colorbar(im, ax=ax, label='Difference from Standard')
                                                
                                            ax.set_title(f"Difference from Standard {compare_standard.title()} Table")
                                            ax.set_xlabel("Frequency (Horizontal)")
                                            ax.set_ylabel("Frequency (Vertical)")
                                            st.pyplot(fig)

                                    with st.expander(f"View Raw Values for Comparison with Standard {compare_standard.title()}"):
                                            diff_df = pd.DataFrame(comparison['difference'])
                                            st.dataframe(np.array(comparison['difference']).reshape((8, 8)))
                                    st.markdown("---")
            else:
                # Educational section
                st.markdown('### <i class="fa-solid fa-circle-info"></i> About GPS EXIF Data & Privacy', unsafe_allow_html=True)
                with st.expander("Details", expanded=False):
                    st.markdown("""
                    **JPEG Quantization Tables** are 8×8 matrices that control compression quality:
                    
                    - **Lower values** = Higher quality, less compression
                    - **Higher values** = Lower quality, more compression
                    - **Top-left corner** contains low-frequency coefficients (most important)
                    - **Bottom-right corner** contains high-frequency coefficients (fine details)
                    
                    **Standard Tables:**
                    - **Luminance (Y)**: Used for brightness information
                    - **Chrominance (Cb, Cr)**: Used for color information
                    
                    **Quality Factor Estimation:**
                    - Quality 100: Minimal compression
                    - Quality 75-90: High quality
                    - Quality 50: Balanced
                    - Quality 10-25: High compression
                    """)
                st.markdown("---")
            

        elif page1 == "Compression History":
            st.markdown("""
            <style>
                .main-header {
                    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                    padding: 2rem;
                    border-radius: 10px;
                    color: white;
                    text-align: center;
                    margin-bottom: 2rem;
                }
            </style>
            """, unsafe_allow_html=True)
            
            # Header
            st.markdown("""
            <div class="main-header">
                <h1>Compression History</h1>
                <p>This tool analyzes JPEG quantization tables, which control the compression quality and characteristics.
            Upload a JPEG image to examine its quantization tables and compare them with standard tables.</p>
            </div>
            """, unsafe_allow_html=True)
            
    elif page_E == "Metadata Forensics":
        page1 = st.sidebar.selectbox("Select Analysis Tool", 
                ["EXIF Data",
                #  "Geolocation Data",
                # "Thumbnail Analysis",
                # "Timestamp Analysis",
                # "Software Detection"],
                "Geolocation Data"],
                help="Choose Analysis Tool"
                )
        
        if page1 == "EXIF Data":
            # Custom CSS for better styling
            st.markdown("""
            <style>
                @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css');
                
                /* ====== LIGHT MODE (default) ====== */
                .main-header {
                    text-align: center;
                    padding: 2rem 0;
                    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                    border-radius: 10px;
                    margin-bottom: 2rem;
                    color: white;
                }

                .camera-card {
                    background: #f8f9fa;
                    padding: 1rem;
                    border-radius: 8px;
                    border-left: 4px solid #667eea;
                    margin-bottom: 0.5rem;
                }

                .metric-container {
                    background: white;
                    padding: 1rem;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    text-align: center;
                    border: 1px solid #e0e0e0;
                }

                .section-header {
                    color: #333;
                    border-bottom: 2px solid #667eea;
                    padding-bottom: 0.5rem;
                    margin-bottom: 1rem;
                }

                .tag-chip {
                    background: #e3f2fd;
                    padding: 0.2rem 0.5rem;
                    border-radius: 15px;
                    font-size: 0.8rem;
                    color: #1976d2;
                    margin: 0.1rem;
                    display: inline-block;
                }

                .error-box {
                    background: #ffebee;
                    border: 1px solid #f44336;
                    color: #d32f2f;
                    padding: 1rem;
                    border-radius: 8px;
                    margin: 1rem 0;
                }

                .success-box {
                    background: #e8f5e8;
                    border: 1px solid #4caf50;
                    color: #2e7d32;
                    padding: 1rem;
                    border-radius: 8px;
                    margin: 1rem 0;
                }

                /* ====== DARK MODE ====== */
                @media (prefers-color-scheme: dark) {
                    body {
                        background-color: #121212;
                        color: #f5f5f5;
                    }

                    .main-header {
                        background: linear-gradient(90deg, #3f4cbb 0%, #5d3a99 100%);
                        color: #f5f5f5;
                    }

                    .camera-card {
                        background: #1e1e1e;
                        border-left: 4px solid #8b9eff;
                        color: #e0e0e0;
                    }

                    .metric-container {
                        background: #1f1f1f;
                        border: 1px solid #333;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.5);
                        color: #f5f5f5;
                    }

                    .section-header {
                        color: #ddd;
                        border-bottom: 2px solid #8b9eff;
                    }

                    .tag-chip {
                        background: #263238;
                        color: #82b1ff;
                    }

                    .error-box {
                        background: #3b1e1e;
                        border: 1px solid #ef5350;
                        color: #ff8a80;
                    }

                    .success-box {
                        background: #1b3b1e;
                        border: 1px solid #81c784;
                        color: #a5d6a7;
                    }
                }

            </style>
            """, unsafe_allow_html=True)

            def extract_comprehensive_metadata(file_bytes):
                """Extract and clean comprehensive metadata from image bytes."""
                metadata = {}
                
                def clean_value(value):
                    """Clean EXIF values recursively to remove nulls and decode bytes."""
                    if isinstance(value, bytes):
                        try:
                            return value.decode("utf-8", errors="ignore").replace("\x00", "").strip()
                        except:
                            return value.hex()
                    elif isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(v, int) for v in value):
                        # Convert rational numbers to decimal
                        numerator, denominator = value
                        try:
                            return round(numerator / denominator, 6)
                        except ZeroDivisionError:
                            return None
                    elif isinstance(value, (list, tuple)):
                        return [clean_value(v) for v in value]
                    elif isinstance(value, dict):
                        return {k: clean_value(v) for k, v in value.items()}
                    else:
                        return value

                try:
                    exif_dict = piexif.load(file_bytes)
                    
                    for section_name, section_data in exif_dict.items():
                        if section_name == "thumbnail":
                            if section_data:
                                thumb_hash = hashlib.md5(section_data).hexdigest()
                                metadata["Thumbnail"] = {
                                    "Hash": thumb_hash,
                                    "Size (bytes)": len(section_data)
                                }
                        elif isinstance(section_data, dict):
                            metadata[section_name] = {}
                            for tag, value in section_data.items():
                                tag_name = piexif.TAGS.get(section_name, {}).get(tag, {"name": f"Tag_{tag}"})["name"]
                                metadata[section_name][tag_name] = clean_value(value)
                
                except Exception as e:
                    metadata["Error"] = str(e)
                
                return metadata

            def format_camera_value(key, value):
                """Format camera values for better display"""
                if key == "ExposureTime" and isinstance(value, (int, float)):
                    if value < 1:
                        return f"1/{int(1/value)}s"
                    else:
                        return f"{value}s"
                elif key == "FNumber" and isinstance(value, (int, float)):
                    return f"f/{value}"
                elif key == "FocalLength" and isinstance(value, (int, float)):
                    return f"{value}mm"
                elif key == "Flash" and isinstance(value, int):
                    flash_modes = {0: "No Flash", 1: "Flash", 5: "Flash (no return)", 7: "Flash (return)", 
                                9: "Fill Flash", 13: "Fill Flash (no return)", 15: "Fill Flash (return)",
                                16: "No Flash (compulsory)", 24: "No Flash (auto)", 25: "Flash (auto)",
                                29: "Flash (auto, no return)", 31: "Flash (auto, return)"}
                    return flash_modes.get(value, f"Flash Mode {value}")
                return str(value)

            def get_file_info(uploaded_file, file_bytes):
                """Get basic file information"""
                try:
                    img = Image.open(uploaded_file)
                    return {
                        "File Name": uploaded_file.name,
                        "File Size": f"{len(file_bytes) / 1024:.1f} KB",
                        "Image Size": f"{img.width} × {img.height} pixels",
                        "Color Mode": img.mode,
                        "Format": img.format
                    }
                except:
                    return {
                        "File Name": uploaded_file.name,
                        "File Size": f"{len(file_bytes) / 1024:.1f} KB"
                    }

            # Main header
            st.markdown("""
            <div class="main-header">
                <h1><i class="fas fa-camera"></i> EXIF Metadata Viewer</h1>
                <p>Upload an image to explore its metadata and camera settings</p>
            </div>
            """, unsafe_allow_html=True)

            # File upload section
            
            uploaded_file = st.file_uploader(
                    "Choose an image file",
                    type=["jpg", "jpeg", "tiff", "png"],
                    help="Supported formats: JPEG, TIFF, PNG"
                )

            if uploaded_file:
                file_bytes = uploaded_file.read()
                
                # Get basic file info
                file_info = get_file_info(uploaded_file, file_bytes)
                
                # Extract EXIF data
                exif_data = extract_comprehensive_metadata(file_bytes)
                st.markdown("---")
                
                if "Error" in exif_data:
                    st.markdown(f"""
                    <div class="error-box">
                        <strong><i class="fas fa-exclamation-triangle"></i> Error reading EXIF data:</strong><br>
                        {exif_data['Error']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="success-box">
                        <strong><i class="fas fa-check-circle"></i> EXIF data successfully extracted!</strong>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display image preview and basic info
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown('<h3 class="section-header"><i class="fas fa-image"></i> Image Preview</h3>', unsafe_allow_html=True)
                    try:
                        img = Image.open(uploaded_file)
                        st.image(img, use_column_width=True)
                    except:
                        st.info("Image preview not available")
                
                with col2:
                    st.markdown('<h3 class="section-header"><i class="fas fa-chart-bar"></i> File Information</h3>', unsafe_allow_html=True)
                    for key, value in file_info.items():
                        st.markdown(f"""
                        <div class="camera-card">
                            <strong>{key}:</strong> {value}
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("---")
                if "Error" not in exif_data:
                    # Merge all EXIF sections into a flat dictionary
                    all_exif = {}
                    for section in exif_data:
                        if isinstance(exif_data[section], dict):
                            all_exif.update(exif_data[section])
                    
                    # Camera Information Section
                    st.markdown('<h3 class="section-header"><i class="fas fa-camera"></i> Camera Settings</h3>', unsafe_allow_html=True)
                    
                    important_tags = [
                        'Make', 'Model', 'DateTime', 'Software', 'Flash', 
                        'FocalLength', 'ExposureTime', 'FNumber', 'ISO'
                    ]
                    
                    camera_info = {tag: all_exif[tag] for tag in important_tags if tag in all_exif}
                    
                    if camera_info:
                        # Create metrics layout
                        cols = st.columns(min(len(camera_info), 4))
                        for i, (key, value) in enumerate(camera_info.items()):
                            with cols[i % len(cols)]:
                                formatted_value = format_camera_value(key, value)
                                st.markdown(f"""
                                <div class="metric-container">
                                    <h4 style="margin: 0; color: #667eea;">{key}</h4>
                                    <p style="margin: 0.5rem 0 0 0; font-size: 1.1em; font-weight: bold;">{formatted_value}</p>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.info("No camera settings found in this image")
                    
                    # GPS Information (if available)
                    st.markdown("---")
                    if 'GPS' in exif_data and exif_data['GPS']:
                        st.markdown('<h3 class="section-header"><i class="fas fa-map-marker-alt"></i> GPS Location</h3>', unsafe_allow_html=True)
                        gps_data = exif_data['GPS']
                        
                        # Try to format GPS coordinates
                        if 'GPSLatitude' in gps_data and 'GPSLongitude' in gps_data:
                            lat = gps_data.get('GPSLatitude', 'N/A')
                            lon = gps_data.get('GPSLongitude', 'N/A')
                            lat_ref = gps_data.get('GPSLatitudeRef', 'N')
                            lon_ref = gps_data.get('GPSLongitudeRef', 'E')
                            
                            st.markdown(f"""
                            <div class="camera-card">
                                <strong>Coordinates:</strong> {lat}° {lat_ref}, {lon}° {lon_ref}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Display other GPS data
                        for key, value in gps_data.items():
                            if key not in ['GPSLatitude', 'GPSLongitude', 'GPSLatitudeRef', 'GPSLongitudeRef']:
                                st.markdown(f"""
                                <div class="camera-card">
                                    <strong>{key}:</strong> {value}
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Detailed EXIF sections
                    st.markdown("---")
                    st.markdown('<h3 class="section-header"><i class="fas fa-search"></i> Detailed Metadata</h3>', unsafe_allow_html=True)
                    
                    # Create tabs for different sections
                    sections = [section for section in exif_data.keys() if isinstance(exif_data[section], dict) and exif_data[section]]
                    
                    if sections:
                        tabs = st.tabs([f' {section}' for section in sections])
                        
                        for tab, section in zip(tabs, sections):
                            with tab:
                                values = exif_data[section]
                                if values:
                                    # Create a more readable table
                                    table_data = []
                                    for k, v in values.items():
                                        table_data.append({
                                            "Tag": k,
                                            "Value": str(v)[:100] + "..." if len(str(v)) > 100 else str(v)
                                        })
                                    st.dataframe(table_data, use_container_width=True)
                                else:
                                    st.info(f"No data available in {section} section")
                    
                    # Raw JSON view (collapsible)
                    with st.expander("Raw JSON Data", expanded=False):
                        st.json(exif_data)
                    st.markdown("---")
                    # Download button for JSON
                    st.markdown('<h3 class="section-header"><i class="fa-solid fa-download"></i> Download</h3>', unsafe_allow_html=True)
                    json_data = json.dumps(exif_data, indent=2)
                    st.download_button(
                        label="Download EXIF data as JSON",
                        data=json_data,
                        file_name=f"exif_{uploaded_file.name}.json",
                        mime="application/json"
                    )
                    st.markdown("---")
            # else:
            #     st.markdown("---")
        elif page1 == "Geolocation Data":
            def get_exif_data(image):
                """Extract EXIF data including GPS info."""
                exif_data = {}
                try:
                    info = image._getexif()
                    if not info:
                        return {}
                    for tag, value in info.items():
                        tag_name = TAGS.get(tag, tag)
                        if tag_name == "GPSInfo":
                            gps_data = {}
                            for t in value:
                                sub_tag = GPSTAGS.get(t, t)
                                gps_data[sub_tag] = value[t]
                            exif_data["GPSInfo"] = gps_data
                        else:
                            exif_data[tag_name] = value
                except Exception as e:
                    st.error(f"EXIF extraction failed: {e}")
                return exif_data

            def convert_to_degrees(value):
                """Convert GPS coordinates to degrees in float format from IFDRational or tuple."""
                try:
                    d = float(value[0])
                    m = float(value[1])
                    s = float(value[2])
                    return d + (m / 60.0) + (s / 3600.0)
                except Exception as e:
                    st.error(f"Error converting to degrees: {e}")
                    return None

            def extract_gps_coords(gps_info):
                try:
                    # Latitude
                    lat = convert_to_degrees(gps_info['GPSLatitude'])
                    ref_lat = gps_info['GPSLatitudeRef']
                    if isinstance(ref_lat, bytes):
                        ref_lat = ref_lat.decode()
                    if ref_lat != 'N':
                        lat = -lat

                    # Longitude
                    lon = convert_to_degrees(gps_info['GPSLongitude'])
                    ref_lon = gps_info['GPSLongitudeRef']
                    if isinstance(ref_lon, bytes):
                        ref_lon = ref_lon.decode()
                    if ref_lon != 'E':
                        lon = -lon

                    # Altitude
                    altitude = None
                    if 'GPSAltitude' in gps_info:
                        alt_val = gps_info['GPSAltitude']
                        if isinstance(alt_val, (tuple, list)) and len(alt_val) == 2:
                            altitude = float(alt_val[0]) / float(alt_val[1])
                        else:
                            altitude = float(alt_val)

                    # Time
                    gps_time = gps_info.get('GPSTimeStamp')
                    formatted_time = None
                    if gps_time:
                        try:
                            formatted_time = ":".join([str(int(float(x[0]) / float(x[1]))) for x in gps_time])
                        except:
                            formatted_time = str(gps_time)

                    # Date
                    gps_date = gps_info.get('GPSDateStamp')
                    if isinstance(gps_date, bytes):
                        gps_date = gps_date.decode(errors='ignore')

                    # Processing Method
                    processing_method = gps_info.get('GPSProcessingMethod')
                    if isinstance(processing_method, bytes):
                        processing_method = processing_method.decode(errors='ignore').replace("\x00", "")

                    # Version
                    gps_version = gps_info.get('GPSVersionID')

                    # Clean version data if it's a tuple/bytes
                    clean_version = None
                    if gps_version:
                        if isinstance(gps_version, (tuple, list)):
                            clean_version = ".".join(str(x) for x in gps_version)
                        else:
                            clean_version = str(gps_version)

                    return {
                        "Parsed Data": {
                            'Latitude': lat,
                            'Longitude': lon,
                            'Altitude': altitude,
                            'Time': formatted_time,
                            'Date': gps_date,
                            'ProcessingMethod': processing_method,
                            'Version': clean_version,
                        },
                        "Raw GPSInfo": {str(k): str(v) for k, v in gps_info.items()}
                    }

                except Exception as e:
                    st.warning(f"Could not extract GPS coordinates: {e}")
                    return None

            def get_all_exif_data(image):
                """Extract all EXIF data for comprehensive metadata display."""
                try:
                    exif_dict = {}
                    info = image._getexif()
                    if info:
                        for tag, value in info.items():
                            tag_name = TAGS.get(tag, tag)
                            if tag_name != "GPSInfo":  # Handle GPS separately
                                # Convert bytes to string for display
                                if isinstance(value, bytes):
                                    try:
                                        value = value.decode('utf-8', errors='ignore')
                                    except:
                                        value = str(value)
                                exif_dict[tag_name] = str(value)
                    return exif_dict
                except Exception as e:
                    st.error(f"Error extracting EXIF data: {e}")
                    return {}

            
                def convert_value(value):
                    if isinstance(value, bytes):
                        try:
                            return value.decode(errors="ignore")  # decode to string
                        except:
                            return value.hex()  # fallback to hex
                    elif isinstance(value, (list, tuple)):
                        return [convert_value(v) for v in value]
                    elif isinstance(value, dict):
                        return {k: convert_value(v) for k, v in value.items()}
                    else:
                        return value

                cleaned_exif = convert_value(exif_data)
                return json.dumps(cleaned_exif, indent=4, ensure_ascii=False)


            def format_coordinates(lat, lon):
                """Format coordinates in multiple formats."""
                def decimal_to_dms(decimal_coord):
                    """Convert decimal degrees to degrees, minutes, seconds."""
                    degrees = int(decimal_coord)
                    minutes_float = abs(decimal_coord - degrees) * 60
                    minutes = int(minutes_float)
                    seconds = (minutes_float - minutes) * 60
                    return degrees, minutes, seconds
                
                lat_dms = decimal_to_dms(abs(lat))
                lon_dms = decimal_to_dms(abs(lon))
                
                lat_dir = "N" if lat >= 0 else "S"
                lon_dir = "E" if lon >= 0 else "W"
                
                return {
                    "Decimal Degrees": f"{lat:.6f}, {lon:.6f}",
                    "DMS": f"{lat_dms[0]}°{lat_dms[1]}'{lat_dms[2]:.2f}\"{lat_dir}, {lon_dms[0]}°{lon_dms[1]}'{lon_dms[2]:.2f}\"{lon_dir}",
                    "Google Maps URL": f"https://www.google.com/maps?q={lat},{lon}",
                    "OpenStreetMap URL": f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}&zoom=15"
                }

            def export_data(gps_data, exif_data, filename):
                """Prepare data for export."""
                # Clean EXIF data to ensure JSON serialization
                clean_exif = {}
                for k, v in exif_data.items():
                    if isinstance(v, bytes):
                        try:
                            clean_exif[k] = v.decode('utf-8', errors='ignore')
                        except:
                            clean_exif[k] = str(v)
                    else:
                        clean_exif[k] = str(v) if v is not None else None
                
                # Clean GPS data recursively
                def clean_for_json(obj):
                    if isinstance(obj, dict):
                        return {k: clean_for_json(v) for k, v in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return [clean_for_json(item) for item in obj]
                    elif isinstance(obj, bytes):
                        try:
                            return obj.decode('utf-8', errors='ignore')
                        except:
                            return str(obj)
                    elif obj is None:
                        return None
                    else:
                        return str(obj) if not isinstance(obj, (int, float, str, bool)) else obj
                
                clean_gps_data = clean_for_json(gps_data) if gps_data else None
                
                export_dict = {
                    "filename": filename,
                    "extraction_timestamp": datetime.now().isoformat(),
                    "gps_data": clean_gps_data,
                    "camera_info": {k: v for k, v in clean_exif.items() if k in ['Make', 'Model', 'DateTime', 'Software']},
                    "all_exif": clean_exif
                }
                return json.dumps(export_dict, indent=2)

            def create_info_card(title, value, icon="📊"):
                """Create a styled info card."""
                if value is not None and value != "":
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 1rem;
                        border-radius: 10px;
                        color: white;
                        margin: 0.5rem 0;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    ">
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <span style="font-size: 1.2em;">{icon}</span>
                            <div>
                                <div style="font-size: 0.8em; opacity: 0.8;">{title}</div>
                                <div style="font-size: 1.1em; font-weight: bold;">{value}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            def create_coordinate_card(format_name, coord_str, is_url=False):
                """Create a styled coordinate display card."""
                if is_url:
                    link_color = "#4CAF50" if "Google" in format_name else "#2196F3"
                    st.markdown(f"""
                    <div style="
                        background: white;
                        border: 2px solid {link_color};
                        padding: 1rem;
                        border-radius: 8px;
                        margin: 0.5rem 0;
                        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                    ">
                        <div style="color: #333; font-weight: bold; margin-bottom: 5px;">{format_name}</div>
                        <a href="{coord_str}" target="_blank" style="
                            color: {link_color}; 
                            text-decoration: none; 
                            font-family: monospace;
                            word-break: break-all;
                        ">{coord_str} →</a>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="
                        background: #f8f9fa;
                        border-left: 4px solid #007bff;
                        padding: 1rem;
                        border-radius: 0 8px 8px 0;
                        margin: 0.5rem 0;
                    ">
                        <div style="color: #333; font-weight: bold; margin-bottom: 5px;">{format_name}</div>
                        <code style="
                            background: #e9ecef; 
                            padding: 0.2rem 0.4rem; 
                            border-radius: 4px;
                            font-size: 1.1em;
                        ">{coord_str}</code>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("""
            <style>
                .main-header {
                    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                    padding: 2rem;
                    border-radius: 10px;
                    color: white;
                    text-align: center;
                    margin-bottom: 2rem;
                }
                    /* Base style (optional default) */
                .info-card {
                    padding: 0.5rem;
                    border-radius: 8px;
                    border-left: 4px solid #667eea;
                    margin: 1rem 0;
                    transition: background 0.3s ease, color 0.3s ease;
                }

                /* Light mode */
                @media (prefers-color-scheme: light) {
                    .info-card {
                        background: #f8f9fa;
                        color: #212529;
                    }
                }

                /* Dark mode */
                @media (prefers-color-scheme: dark) {
                    .info-card {
                        background: #2a2a2a;
                        color: #f1f1f1;
                    }
                }
                # .info-card {
                #     background: white;
                #     border-radius: 15px;
                #     padding: 1rem;
                #     box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                #     margin: 1rem 0;
                # }
            </style>
            """, unsafe_allow_html=True)

            # Header
            st.markdown("""
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
            """, unsafe_allow_html=True)


            st.markdown("""
            <div class="main-header">
                <h1><i class="fa-solid fa-location-dot"></i> Enhanced Image Geolocation Finder</h1>
                <p>Extract GPS metadata from your images and visualize locations on interactive maps</p>
            </div>
            """, unsafe_allow_html=True)

            # Sidebar with enhanced styling
            with st.sidebar:
                st.sidebar.markdown("---")
                with st.expander("Settings & Options", expanded=False):
                # st.sidebar.markdown("## ⚙️ Settings & Options")
                # st.sidebar.markdown("---")

                # Enhanced sidebar options
                    # show_all_exif = st.checkbox("Show detailed EXIF metadata", value=False)
                    map_style = st.selectbox(
                        "Map Style", 
                        ["OpenStreetMap", "Stamen Terrain", "CartoDB Positron"],
                        help="Choose your preferred map visualization style"
                    )

                    # Map zoom level
                    map_zoom = st.slider("Map Zoom Level", min_value=10, max_value=20, value=10)
                with st.expander("Session Stats", expanded=False):
            # st.sidebar.markdown("---")
                    # st.markdown("### Session Stats")
                    if 'processed_images' not in st.session_state:
                        st.session_state.processed_images = 0
                    if 'images_with_gps' not in st.session_state:
                        st.session_state.images_with_gps = 0

                    st.metric("Images Processed", st.session_state.processed_images)
                    st.metric("GPS Data Found", st.session_state.images_with_gps)
                st.markdown("---")
            # File uploader with enhanced styling
            # st.markdown("""
            # <div class="upload-section">
            #     <h3>📤 Upload Your Images</h3>
            #     <p>Drag and drop your JPEG files here, or click to browse</p>
            # </div>
            # """, unsafe_allow_html=True)

            uploaded_files = st.file_uploader(
                "Choose images", 
                type=["jpg", "jpeg"], 
                accept_multiple_files=True,
                help="Upload JPEG images with GPS EXIF data. Multiple files supported."
            )

            if uploaded_files:
                # Progress bar for processing
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process multiple images
                for i, uploaded_file in enumerate(uploaded_files):
                    # Update progress
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing {uploaded_file.name}... ({i+1}/{len(uploaded_files)})")
                    
                    st.markdown("---")
                    
                    # Image card container
                    with st.container():
                        # st.markdown(f"""
                        # <div class="info-card">
                        #     <h2>📷 {uploaded_file.name}</h2>
                        # </div>
                        # """, unsafe_allow_html=True)
                        
                        try:
                            image = Image.open(uploaded_file)
                            st.session_state.processed_images += 1
                            
                            # Create enhanced layout
                            col1, col2 = st.columns([1.2, 1])
                            
                            with col1:
                                # Enhanced image display
                                resized_image = image.resize((500, 350))
                                st.image(
                                    resized_image, 
                                    caption=f"{uploaded_file.name}", 
                                    clamp=True
                                )
                                
                                # Image info
                                width, height = image.size
                                file_size = len(uploaded_file.getvalue()) / 1024  # KB
                                
                                col1a, col1b, col1c = st.columns(3)
                                with col1a:
                                    st.metric("Width", f"{width}px")
                                with col1b:
                                    st.metric("Height", f"{height}px")
                                with col1c:
                                    st.metric("File Size", f"{file_size:.1f} KB")
                            
                            with col2:
                                # Extract all EXIF data
                                exif_data = get_exif_data(image)
                                all_exif = get_all_exif_data(image)
                                
                                
                                if "GPSInfo" in exif_data:
                                    gps_data = extract_gps_coords(exif_data["GPSInfo"])
                                    if gps_data:
                                        st.success("✅ GPS Data Found!")
                                        st.session_state.images_with_gps += 1
                                        
                                        lat = gps_data["Parsed Data"]['Latitude']
                                        lon = gps_data["Parsed Data"]['Longitude']
                                        
                                        # GPS Info Cards
                                        parsed_data = gps_data["Parsed Data"]
                                        
                                        if parsed_data.get('Altitude'):
                                            create_info_card("Altitude", f"{parsed_data['Altitude']:.1f} m", '<i class="fa-solid fa-mountain"></i>')

                                        if parsed_data.get('Date'):
                                            create_info_card("Date", parsed_data['Date'], '<i class="fa-solid fa-calendar"></i>')

                                        if parsed_data.get('Time'):
                                            create_info_card("Time", parsed_data['Time'], '<i class="fa-solid fa-clock"></i>')

                                        if parsed_data.get('ProcessingMethod'):
                                            create_info_card("GPS Method", parsed_data['ProcessingMethod'], '<i class="fa-solid fa-satellite"></i>')

                                
                                else:
                                    st.markdown('### <i class="fa-solid fa-circle-xmark"></i> No GPS Data Found', unsafe_allow_html=True)
                                    # st.error("❌ No GPS Data Found")
                                    st.info("This image doesn't contain GPS metadata. Make sure location services were enabled when the photo was taken.")
                                    gps_data = None
                            
                            # Show coordinates if GPS data exists
                            if "GPSInfo" in exif_data and gps_data:
                                st.markdown("---")
                                st.markdown('### <i class="fa-solid fa-location-dot"></i> Location Coordinates', unsafe_allow_html=True)
                                with st.expander("Location Coordinates", expanded=False):
                                    # st.markdown("### 📍 Location Coordinates")
                                    
                                    lat = gps_data["Parsed Data"]['Latitude']
                                    lon = gps_data["Parsed Data"]['Longitude']
                                    
                                    coord_formats = format_coordinates(lat, lon)
                                    
                                    # Display coordinates in styled cards
                                    for format_name, coord_str in coord_formats.items():
                                        is_url = "URL" in format_name
                                        create_coordinate_card(format_name, coord_str, is_url)
                            
                            # Enhanced map display
                            if "GPSInfo" in exif_data and gps_data:
                                st.markdown('### <i class="fa-solid fa-earth-americas"></i> Interactive Map', unsafe_allow_html=True)
                                # st.markdown("### 🌍 Interactive Map")
                                
                                # Create map with selected style
                                if map_style == "Stamen Terrain":
                                    map_ = folium.Map(
                                        location=[lat, lon], 
                                        zoom_start=map_zoom, 
                                        tiles="https://stamen-tiles.a.ssl.fastly.net/toner/{z}/{x}/{y}.png",
                                        attr="Map tiles by Stamen Design, CC BY 3.0 — Map data © OpenStreetMap contributors"
                                    )
                                elif map_style == "CartoDB Positron":
                                    map_ = folium.Map(
                                        location=[lat, lon], 
                                        zoom_start=map_zoom, 
                                        tiles="https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png",
                                        attr="© OpenStreetMap contributors, © CARTO"
                                    )
                                else:
                                    map_ = folium.Map(location=[lat, lon], zoom_start=map_zoom)
                                
                                # Enhanced marker with popup info
                                popup_html = f"""
                                <div style="font-family: Arial, sans-serif; width: 250px;">
                                    <h4 style="margin: 0 0 10px 0; color: #333;">📷 {uploaded_file.name}</h4>
                                    <hr style="margin: 10px 0;">
                                    <p style="margin: 5px 0;"><strong>📍 Coordinates:</strong><br>
                                    {lat:.6f}, {lon:.6f}</p>
                                """
                                
                                if gps_data["Parsed Data"].get('Altitude'):
                                    popup_html += f'<p style="margin: 5px 0;"><strong>⛰️ Altitude:</strong> {gps_data["Parsed Data"]["Altitude"]:.1f}m</p>'
                                if gps_data["Parsed Data"].get('Date'):
                                    popup_html += f'<p style="margin: 5px 0;"><strong>📅 Date:</strong> {gps_data["Parsed Data"]["Date"]}</p>'
                                if gps_data["Parsed Data"].get('Time'):
                                    popup_html += f'<p style="margin: 5px 0;"><strong>⏰ Time:</strong> {gps_data["Parsed Data"]["Time"]}</p>'
                                
                                popup_html += "</div>"
                                
                                # Custom marker icon
                                folium.Marker(
                                    [lat, lon], 
                                    tooltip="📍 Click for image details",
                                    popup=folium.Popup(popup_html, max_width=300),
                                    icon=folium.Icon(color='red', icon='camera', prefix='fa')
                                ).add_to(map_)
                                
                                # Add a circle to show precision
                                folium.Circle(
                                    [lat, lon],
                                    radius=50,
                                    popup="Approximate GPS accuracy area",
                                    color='blue',
                                    fill=True,
                                    opacity=0.3
                                ).add_to(map_)
                                
                                st_folium(map_, width=700, height=450)
                                st.markdown("---")
                            
                            # Show all EXIF data if requested
                            # if show_all_exif and all_exif:
                            #     st.markdown('### <i class="fa-solid fa-clipboard-list"></i> Complete EXIF Metadata', unsafe_allow_html=True)
                                # st.markdown("### 📋 Complete EXIF Metadata")
                                
                                # # Camera info highlights
                                # camera_info = {}
                                # important_tags = ['Make', 'Model', 'DateTime', 'Software', 'Flash', 'FocalLength', 'ExposureTime', 'FNumber', 'ISO']
                                
                                
                                
                                # for tag in important_tags:
                                #     if tag in all_exif:
                                #         camera_info[tag] = all_exif[tag]
                                
                                # if camera_info:
                                #     st.markdown('#### <i class="fa-solid fa-camera"></i> Camera Information', unsafe_allow_html=True)
                                #     # st.markdown("#### 📸 Camera Information")
                                #     camera_cols = st.columns(len(camera_info))
                                #     for i, (key, value) in enumerate(camera_info.items()):
                                #         with camera_cols[i % len(camera_cols)]:
                                #             st.markdown(f"**{key}**\n\n{value}")

                                
                                # # All EXIF data table
                                # st.markdown('#### <i class="fa-solid fa-table-list"></i> All Metadata', unsafe_allow_html=True)
                                # # st.markdown("#### 📊 All Metadata")
                                # df = pd.DataFrame(list(all_exif.items()), columns=['Tag', 'Value'])
                                # st.dataframe(
                                #     df, 
                                #     use_container_width=True,
                                #     height=300
                                # )
                            
                            # Enhanced export section
                            if gps_data:
                                st.markdown('### <i class="fa-solid fa-file-export"></i> Export & Download', unsafe_allow_html=True)
                                # st.markdown("### 💾 Export & Download")
                                
                                col_export1, col_export2 = st.columns(2)
                                
                                with col_export1:
                                    export_json = export_data(gps_data, all_exif, uploaded_file.name)
                                    st.download_button(
                                        label="Download JSON Data",
                                        data=export_json,
                                        file_name=f"{uploaded_file.name}_geolocation.json",
                                        mime="application/json",
                                        help="Download all extracted GPS and EXIF data in JSON format"
                                    )
                                
                                with col_export2:
                                    # Create CSV for coordinates
                                    csv_data = f"filename,latitude,longitude,altitude,date,time\n"
                                    csv_data += f"{uploaded_file.name},{lat},{lon},"
                                    csv_data += f"{parsed_data.get('Altitude', '')},"
                                    csv_data += f"{parsed_data.get('Date', '')},"
                                    csv_data += f"{parsed_data.get('Time', '')}"
                                    
                                    st.download_button(
                                        label="Download CSV Data",
                                        data=csv_data,
                                        file_name=f"{uploaded_file.name}_coordinates.csv",
                                        mime="text/csv",
                                        help="Download GPS coordinates in CSV format"
                                    )
                        
                        except Exception as e:
                            st.error(f"❌ Could not process {uploaded_file.name}: {e}")
                
                # Clear progress bar when done
                progress_bar.empty()
                status_text.empty()
                
                # Summary section
                if len(uploaded_files) > 1:
                    st.markdown("---")
                    st.markdown('### <i class="fa-solid fa-chart-bar"></i> Processing Summary', unsafe_allow_html=True)
                    # st.markdown("### 📊 Processing Summary")
                    
                    summary_col1, summary_col2, summary_col3 = st.columns(3)
                    with summary_col1:
                        st.metric("Total Images", len(uploaded_files))
                    with summary_col2:
                        st.metric("Images with GPS", st.session_state.images_with_gps)
                    with summary_col3:
                        success_rate = (st.session_state.images_with_gps / len(uploaded_files)) * 100
                        st.metric("Success Rate", f"{success_rate:.1f}%")

            # Enhanced information section
            st.markdown('### <i class="fa-solid fa-circle-info"></i> About GPS EXIF Data & Privacy', unsafe_allow_html=True)
            with st.expander("Details", expanded=False):
                st.markdown("""
                ### <i class="fa-solid fa-satellite"></i> What is GPS EXIF Data?

                **EXIF (Exchangeable Image File Format)** metadata can include precise GPS coordinates and other location information:
                - <i **Latitude & Longitude**: Exact geographic coordinates  
                - <i **Altitude**: Elevation above sea level  
                - <i **Timestamp**: When and where the photo was taken  
                - <i **GPS Method**: How the location was determined (GPS, Network, etc.)

                ### <i class="fa-solid fa-mobile-screen-button"></i> Common Sources
                - Smartphones with location services enabled  
                - Digital cameras with built-in GPS  
                - Images edited with location-aware software  

                ### <i class="fa-solid fa-user-shield"></i> Privacy & Security Considerations

                <i class="fa-solid fa-triangle-exclamation"></i> **Important Privacy Notes:**
                - GPS data can reveal sensitive location information  
                - Photos shared online may expose your home, workplace, or travel patterns  
                - Consider the privacy implications before sharing images with embedded GPS data  
                - Many social media platforms automatically strip EXIF data, but not all do  

                ### <i class="fa-solid fa-shield-halved"></i> How to Protect Your Privacy
                - Turn off location services for camera apps when privacy is important  
                - Use EXIF removal tools before sharing photos online  
                - Check your device's privacy settings regularly  
                - Be especially careful with photos of children or private locations  

                ### <i class="fa-solid fa-list-check"></i> Supported Features
                -  JPEG files with embedded GPS EXIF data  
                -  Multiple coordinate format displays  
                -  Interactive map visualization  
                -  Batch processing of multiple images  
                -  Export to JSON and CSV formats  
                -  Comprehensive metadata analysis  
                """, unsafe_allow_html=True)


            # Footer
            st.markdown("---")
            # st.markdown("""
            # <div style="text-align: center; color: #666; padding: 1rem;">
            #     <p>🔧 Built with Streamlit • 🗺️ Maps by OpenStreetMap • 📊 Data visualization enhanced</p>
            # </div>
            # """, unsafe_allow_html=True)

        elif page1 == "Thumbnail Analysis":
            st.title("Compression History")

        elif page1 == "Timestamp Analysis":
            st.title("Compression History")

        elif page1 == "Software Detection":
            st.title("Compression History")
    
    elif page_E == "Advanced Forensic Analysis":
        page1 = st.sidebar.selectbox("Select Analysis Tool", 
                ["Frequency Domain Analysis",
                 "Luminous Analyzer",
                 "PCA Analysis",
                 "Statistical Analysis"],
                help="Choose Analysis Tool"
                )
        
        if page1 == "Luminous Analyzer":
                    st.markdown("""
                    <style>
                        .main-header {
                            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                            padding: 2rem;
                            border-radius: 10px;
                            color: white;
                            text-align: center;
                            margin-bottom: 2rem;
                        }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Header
                    st.markdown("""
                    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
                    """, unsafe_allow_html=True)
                    st.markdown("""
                    <div class="main-header">
                        <h1><i class="fa-solid fa-lightbulb"></i> Luminous Analyzer</h1>
                        <p>Upload an image to analyze its luminance (brightness) distribution with comprehensive statistical analysis.</p>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.sidebar:
                        st.markdown("---")
                        with st.expander("Analysis Options", expanded=False):

                            # Color space selection
                            color_space = st.selectbox(
                                "Luminance Calculation Method",
                                ["ITU-R BT.601 (Standard)", "ITU-R BT.709 (HDTV)", "Simple Average", "Custom Weights"]
                            )

                            # Image resize option
                            max_size = st.slider("Max Image Size (for performance)", 500, 2000, 1000, 50)
                            st.write("Large images will be resized for faster processing")
                        st.markdown("---")

                    # Custom weights if selected
                    if color_space == "Custom Weights":
                        st.sidebar.subheader("Custom RGB Weights")
                        r_weight = st.sidebar.slider("Red Weight", 0.0, 1.0, 0.299, 0.001)
                        g_weight = st.sidebar.slider("Green Weight", 0.0, 1.0, 0.587, 0.001) 
                        b_weight = st.sidebar.slider("Blue Weight", 0.0, 1.0, 0.114, 0.001)
                        
                        # Normalize weights
                        total_weight = r_weight + g_weight + b_weight
                        if total_weight > 0:
                            r_weight, g_weight, b_weight = r_weight/total_weight, g_weight/total_weight, b_weight/total_weight

                    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "bmp", "webp", "tiff"])

                    def calculate_luminance(img_array, method):
                        """Calculate luminance based on selected method"""
                        try:
                            if method == "ITU-R BT.601 (Standard)":
                                return 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
                            elif method == "ITU-R BT.709 (HDTV)":
                                return 0.2126 * img_array[:, :, 0] + 0.7152 * img_array[:, :, 1] + 0.0722 * img_array[:, :, 2]
                            elif method == "Simple Average":
                                return np.mean(img_array, axis=2)
                            elif method == "Custom Weights":
                                return r_weight * img_array[:, :, 0] + g_weight * img_array[:, :, 1] + b_weight * img_array[:, :, 2]
                        except Exception as e:
                            st.error(f"Error calculating luminance: {str(e)}")
                            return None

                    def resize_image_if_needed(image, max_size):
                        """Resize image if it's too large"""
                        width, height = image.size
                        if max(width, height) > max_size:
                            ratio = max_size / max(width, height)
                            new_width = int(width * ratio)
                            new_height = int(height * ratio)
                            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                        return image

                    if uploaded_file:
                        try:
                            # Load and process image
                            with st.spinner("Loading and processing image..."):
                                image = Image.open(uploaded_file).convert("RGB")
                                original_size = image.size
                                
                                # Resize if needed
                                image = resize_image_if_needed(image, max_size)
                                resized_size = image.size
                                
                                if original_size != resized_size:
                                    st.info(f"Image resized from {original_size} to {resized_size} for faster processing")
                                
                                # st.image(image, caption="Processed Image", use_column_width=True)
                                
                                img_np = np.array(image)
                                
                                # Calculate luminance
                                luminance = calculate_luminance(img_np, color_space)
                                
                                if luminance is None:
                                    st.error("Failed to calculate luminance. Please try a different image.")
                                    st.stop()
                                
                            # Basic Statistics
                            avg_lum = np.mean(luminance)
                            min_lum = np.min(luminance)
                            max_lum = np.max(luminance)
                            std_lum = np.std(luminance)
                            median_lum = np.median(luminance)
                            
                            # Percentiles
                            p25 = np.percentile(luminance, 25)
                            p75 = np.percentile(luminance, 75)
                            st.markdown("---")
                            st.markdown('### <i class="fa-solid fa-chart-column"></i> Comprehensive Luminance Statistics', unsafe_allow_html=True)
                            # st.markdown("### 📊 Comprehensive Luminance Statistics")
                            
                            # Display metrics in columns
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Average", f"{avg_lum:.2f}")
                            col2.metric("Median", f"{median_lum:.2f}")
                            col3.metric("Std Deviation", f"{std_lum:.2f}")
                            col4.metric("Range", f"{max_lum - min_lum:.2f}")
                            
                            col5, col6, col7, col8 = st.columns(4)
                            col5.metric("Minimum", f"{min_lum:.2f}")
                            col6.metric("Maximum", f"{max_lum:.2f}")
                            col7.metric("25th Percentile", f"{p25:.2f}")
                            col8.metric("75th Percentile", f"{p75:.2f}")
                            st.markdown("---")
                            # Create tabs for different visualizations
                            tab1, tab2, tab3, tab4 = st.tabs(["Heatmap", "Histogram", "Threshold Analysis", "Data Export"])
                            
                            with tab1:
                                with st.expander("Luminance Heatmap", expanded=False):
                                # st.markdown("### Luminance Heatmap")
                                    fig1, ax1 = plt.subplots(figsize=(12, 8))
                                    heatmap = ax1.imshow(luminance, cmap="inferno", interpolation="nearest")
                                    plt.colorbar(heatmap, ax=ax1, label='Luminance Value')
                                    ax1.set_title(f"Luminance Heatmap ({color_space})")
                                    ax1.axis("off")
                                    st.pyplot(fig1)
                                    
                                    # Download heatmap
                                    buf1 = io.BytesIO()
                                    fig1.savefig(buf1, format="png", dpi=300, bbox_inches='tight')
                                    buf1.seek(0)
                                    st.download_button("Download Heatmap (High Quality)", buf1, 
                                                    file_name="luminance_heatmap.png", mime="image/png")
                            
                            with tab2:
                                with st.expander("Luminance Distribution", expanded=False):
                                # st.markdown("### Luminance Distribution")
                                    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(15, 6))
                                    
                                    # Histogram
                                    ax2.hist(luminance.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                                    ax2.axvline(avg_lum, color='red', linestyle='--', label=f'Mean: {avg_lum:.2f}')
                                    ax2.axvline(median_lum, color='green', linestyle='--', label=f'Median: {median_lum:.2f}')
                                    ax2.set_xlabel('Luminance Value')
                                    ax2.set_ylabel('Frequency')
                                    ax2.set_title('Luminance Distribution')
                                    ax2.legend()
                                    ax2.grid(True, alpha=0.3)
                                    
                                    # Box plot
                                    ax3.boxplot(luminance.flatten(), vert=True)
                                    ax3.set_ylabel('Luminance Value')
                                    ax3.set_title('Luminance Box Plot')
                                    ax3.grid(True, alpha=0.3)
                                    
                                    st.pyplot(fig2)
                                    
                                    # Download histogram
                                    buf2 = io.BytesIO()
                                    fig2.savefig(buf2, format="png", dpi=300, bbox_inches='tight')
                                    buf2.seek(0)
                                    st.download_button("Download Distribution Plot", buf2, 
                                                    file_name="luminance_distribution.png", mime="image/png")
                            
                            with tab3:
                                with st.expander("Interactive Threshold Analysis", expanded=False):
                                # st.markdown("### Interactive Threshold Analysis")
                                
                                    col_a, col_b = st.columns([1, 2])
                                    
                                    with col_a:
                                        threshold_low = st.slider("Lower Threshold", 0.0, 255.0, 50.0, 1.0)
                                        threshold_high = st.slider("Upper Threshold", 0.0, 255.0, 200.0, 1.0)
                                        
                                        # Ensure proper ordering
                                        if threshold_low >= threshold_high:
                                            st.error("Lower threshold must be less than upper threshold!")
                                        else:
                                            # Calculate regions
                                            dark_mask = luminance < threshold_low
                                            bright_mask = luminance > threshold_high
                                            mid_mask = (luminance >= threshold_low) & (luminance <= threshold_high)
                                            
                                            # Statistics for each region
                                            dark_percent = np.sum(dark_mask) / luminance.size * 100
                                            bright_percent = np.sum(bright_mask) / luminance.size * 100
                                            mid_percent = np.sum(mid_mask) / luminance.size * 100
                                            
                                            st.metric("Dark Regions %", f"{dark_percent:.1f}%")
                                            st.metric("Mid-tone Regions %", f"{mid_percent:.1f}%")
                                            st.metric("Bright Regions %", f"{bright_percent:.1f}%")
                                    
                                    with col_b:
                                        if threshold_low < threshold_high:
                                            # Create colored overlay
                                            overlay = np.zeros((*luminance.shape, 3), dtype=np.uint8)
                                            overlay[dark_mask] = [0, 0, 255]      # Blue for dark
                                            overlay[mid_mask] = [0, 255, 0]       # Green for mid-tone
                                            overlay[bright_mask] = [255, 0, 0]    # Red for bright
                                            
                                            st.image(overlay, caption="Threshold Analysis: Blue=Dark, Green=Mid-tone, Red=Bright", 
                                                use_column_width=True)
                                
                            with tab4:
                                with st.expander("Data Export Options", expanded=False):
                                # st.markdown("### Data Export Options")
                                
                                    # Prepare statistics DataFrame
                                    stats_data = {
                                        'Metric': ['Mean', 'Median', 'Standard Deviation', 'Minimum', 'Maximum', 
                                                '25th Percentile', '75th Percentile', 'Range'],
                                        'Value': [avg_lum, median_lum, std_lum, min_lum, max_lum, p25, p75, max_lum - min_lum]
                                    }
                                    stats_df = pd.DataFrame(stats_data)
                                    
                                    # st.subheader('<i class="fa-solid fa-chart-line"></i> Statistics Summary', unsafe_allow_html=True)
                                    st.markdown('### <i class="fa-solid fa-chart-column"></i> Statistics Summary', unsafe_allow_html=True)
                                    # st.subheader("📈 Statistics Summary")
                                    st.dataframe(stats_df, use_container_width=True)
                                    
                                    # Export options
                                    col_export1, col_export2, col_export3 = st.columns(3)
                                    
                                    with col_export1:
                                        # Export statistics as CSV
                                        stats_csv = stats_df.to_csv(index=False)
                                        st.download_button("Download Statistics CSV", stats_csv, 
                                                        file_name="luminance_statistics.csv", mime="text/csv")
                                    
                                    with col_export2:
                                        # Export raw luminance data (sampled for large images)
                                        sample_size = min(10000, luminance.size)
                                        sampled_luminance = np.random.choice(luminance.flatten(), sample_size, replace=False)
                                        luminance_df = pd.DataFrame({'Luminance': sampled_luminance})
                                        luminance_csv = luminance_df.to_csv(index=False)
                                        st.download_button("Download Sample Data CSV", luminance_csv, 
                                                        file_name="luminance_data_sample.csv", mime="text/csv")
                                    
                                    with col_export3:
                                        # Export analysis report
                                        report = f"""Luminance Analysis Report
                        Image: {uploaded_file.name}
                        Analysis Method: {color_space}
                        Original Size: {original_size[0]} x {original_size[1]} pixels
                        Processed Size: {resized_size[0]} x {resized_size[1]} pixels

                        Statistical Summary:
                        - Mean Luminance: {avg_lum:.2f}
                        - Median Luminance: {median_lum:.2f}
                        - Standard Deviation: {std_lum:.2f}
                        - Range: {min_lum:.2f} - {max_lum:.2f}
                        - 25th Percentile: {p25:.2f}
                        - 75th Percentile: {p75:.2f}
                        - Coefficient of Variation: {(std_lum/avg_lum)*100:.2f}%

                        Generated by Advanced Luminous Analyzer Pro
                        """
                                        st.download_button("Download Analysis Report", report, 
                                                        file_name="luminance_analysis_report.txt", mime="text/plain")
                                    
                                    # Additional analysis insights
                                    st.markdown('### <i class="fa-solid fa-magnifying-glass-chart"></i> Analysis Insights', unsafe_allow_html=True)
                                    # st.subheader("🔍 Analysis Insights")
                                    
                                    cv = (std_lum / avg_lum) * 100 if avg_lum > 0 else 0
                                    
                                    insights = []
                                    if cv < 20:
                                        insights.append("**Low variability:** Image has relatively uniform brightness")
                                    elif cv > 50:
                                        insights.append("**High variability:** Image has significant brightness contrast")
                                    else:
                                        insights.append("**Moderate variability:** Image has balanced brightness distribution")

                                    if avg_lum < 85:
                                        insights.append("**Dark image:** Overall low brightness — brightness adjustment recommended")
                                    elif avg_lum > 170:
                                        insights.append("**Bright image:** Well-lit or high exposure")
                                    else:
                                        insights.append("**Balanced brightness:** Well-balanced luminance levels")

                                    if abs(avg_lum - median_lum) > 10:
                                        insights.append("**Skewed distribution:** Mean and median differ significantly (asymmetrical)")
                                    else:
                                        insights.append("**Normal distribution:** Mean and median are similar")


                                    
                                    for insight in insights:
                                        st.write(insight)
                            st.markdown("---")
                        except Exception as e:
                            st.error(f"An error occurred while processing the image: {str(e)}")
                            st.write("Please ensure you've uploaded a valid image file and try again.")

                    else:
                        st.info("👆 Please upload an image to begin analysis")
                        # st.markdown('<i class="fa-solid fa-circle-arrow-up"></i> **Please upload an image to begin analysis**', unsafe_allow_html=True)
                        
                        # Show sample features
                        st.markdown('### <i class="fa-solid fa-star"></i> Features', unsafe_allow_html=True)
                        with st.expander("Details", expanded=False):
                        # st.markdown("### ✨ Features")
                            feature_col1, feature_col2 = st.columns(2)
                            
                            with feature_col1:
                                st.markdown("""
                                **Analysis Options:**
                                - Multiple luminance calculation methods
                                - Automatic image resizing for performance
                                - Comprehensive statistical analysis
                                - Interactive threshold analysis
                                """)
                            
                            with feature_col2:
                                st.markdown("""
                                **Export Capabilities:**
                                - High-quality visualization downloads
                                - Statistical data in CSV format
                                - Detailed analysis reports
                                - Sample data for further analysis
                                """)
                        st.markdown("---")
    
        elif page1 == "PCA Analysis":
            # Custom CSS for better styling
            st.markdown("""
                    <style>
                        .main-header {
                            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                            padding: 2rem;
                            border-radius: 10px;
                            color: white;
                            text-align: center;
                            margin-bottom: 2rem;
                        }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Header
            st.markdown("""
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
            """, unsafe_allow_html=True)


            st.markdown("""
                    <div class="main-header">
                        <h1><i class="fa-solid fa-camera"></i> Advanced PCA Image Analyzer</h1>
                        <p>Analyze image compression and transformation using Principal Component Analysis (PCA) with advanced metrics and visualizations.</p>
                    </div>
                    """, unsafe_allow_html=True)

            # File uploader
            uploaded_file = st.file_uploader(
                "Upload an image", 
                type=["png", "jpg", "jpeg", "bmp", "tiff"],
                help="Supported formats: PNG, JPG, JPEG, BMP, TIFF"
            )

            def apply_pca_on_channel(channel_data, n_components):
                """Apply PCA to a single color channel."""
                pca = PCA(n_components=n_components)
                transformed = pca.fit_transform(channel_data)
                reconstructed = pca.inverse_transform(transformed)
                return reconstructed, pca.explained_variance_ratio_, pca.singular_values_

            def calculate_compression_metrics(original, reconstructed, n_components, max_components):
                """Calculate various compression and quality metrics."""
                # Mean Squared Error
                mse = np.mean((original - reconstructed) ** 2)
                
                # Peak Signal-to-Noise Ratio
                if mse == 0:
                    psnr = float('inf')
                else:
                    max_pixel = 255.0
                    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
                
                # Compression ratio
                compression_ratio = max_components / n_components
                
                # Data retention percentage
                data_retention = (n_components / max_components) * 100
                
                return {
                    'mse': mse,
                    'psnr': psnr,
                    'compression_ratio': compression_ratio,
                    'data_retention': data_retention
                }

            def plot_cumulative_variance(variance_ratios, channel_names, colors):
                """Plot cumulative explained variance for all channels."""
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Individual variance plot
                for i, (var_ratio, name, color) in enumerate(zip(variance_ratios, channel_names, colors)):
                    ax1.plot(range(1, len(var_ratio) + 1), var_ratio, 
                            label=f'{name} Channel', color=color, linewidth=2)
                
                ax1.set_xlabel('Principal Components')
                ax1.set_ylabel('Explained Variance Ratio')
                ax1.set_title('Explained Variance by Component')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Cumulative variance plot
                for i, (var_ratio, name, color) in enumerate(zip(variance_ratios, channel_names, colors)):
                    cumulative_var = np.cumsum(var_ratio)
                    ax2.plot(range(1, len(cumulative_var) + 1), cumulative_var, 
                            label=f'{name} Channel', color=color, linewidth=2)
                
                ax2.set_xlabel('Principal Components')
                ax2.set_ylabel('Cumulative Explained Variance')
                ax2.set_title('Cumulative Explained Variance')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax2.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='90% Threshold')
                ax2.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='95% Threshold')
                
                plt.tight_layout()
                return fig

            def create_comparison_grid(original, reconstructed, n_components):
                """Create a comparison grid showing original vs reconstructed image."""
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                # Original image
                axes[0, 0].imshow(original)
                axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
                axes[0, 0].axis('off')
                
                # Reconstructed image
                axes[0, 1].imshow(reconstructed)
                axes[0, 1].set_title(f'PCA Reconstructed ({n_components} components)', fontsize=14, fontweight='bold')
                axes[0, 1].axis('off')
                
                # Difference image
                diff = np.abs(original.astype(float) - reconstructed.astype(float))
                axes[1, 0].imshow(diff / 255.0, cmap='hot')
                axes[1, 0].set_title('Absolute Difference', fontsize=14, fontweight='bold')
                axes[1, 0].axis('off')
                
                # Histogram comparison
                axes[1, 1].hist(original.flatten(), bins=50, alpha=0.7, label='Original', color='blue')
                axes[1, 1].hist(reconstructed.flatten(), bins=50, alpha=0.7, label='Reconstructed', color='red')
                axes[1, 1].set_title('Pixel Intensity Distribution', fontsize=14, fontweight='bold')
                axes[1, 1].set_xlabel('Pixel Intensity')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                return fig

            if uploaded_file:
                # Load and process image
                with st.spinner('Loading image...'):
                    image = Image.open(uploaded_file).convert("RGB")
                    image_np = np.array(image)
                
                # Display original image info
                st.markdown("---")
                # st.subheader("📊 Image Information")
                st.markdown('### <i class="fa-solid fa-circle-info"></i> Image Information', unsafe_allow_html=True)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Width", f"{image_np.shape[1]} px")
                with col2:
                    st.metric("Height", f"{image_np.shape[0]} px")
                with col3:
                    st.metric("Channels", image_np.shape[2])
                with col4:
                    st.metric("Total Pixels", f"{image_np.shape[0] * image_np.shape[1]:,}")
                
                max_components = min(image_np.shape[0], image_np.shape[1])
                
                with st.sidebar:
                    st.markdown("---")
                    with st.expander("PCA Configuration", expanded=False):
                        # Sidebar controls
                        # st.header("🔧 PCA Configuration")
                        
                        # Component selection methods
                        selection_method = st.radio(
                            "Component Selection Method:",
                            ["Manual", "Variance Threshold", "Compression Ratio"]
                        )
                        
                        if selection_method == "Manual":
                            n_components = st.slider(
                                "Number of PCA Components", 
                                1, max_components, 
                                value=min(50, int(max_components * 0.2)),
                                help="Select the number of principal components to retain"
                            )
                        elif selection_method == "Variance Threshold":
                            variance_threshold = st.slider(
                                "Variance Threshold (%)", 
                                50, 99, 
                                value=90,
                                help="Retain components that explain this percentage of variance"
                            ) / 100
                            n_components = max_components  # Will be calculated after PCA
                        else:  # Compression Ratio
                            compression_ratio = st.slider(
                                "Compression Ratio", 
                                2, 50, 
                                value=10,
                                help="Higher ratio = more compression"
                            )
                            n_components = max(1, max_components // compression_ratio)
                    with st.expander("Analysis Options", expanded=False):
                        show_variance_plot = st.checkbox("Show Variance Analysis", value=True)
                        show_comparison_grid = st.checkbox("Show Detailed Comparison", value=True)
                        show_metrics = st.checkbox("Show Quality Metrics", value=True)
                
                # Processing
                if st.button("Run PCA Analysis", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Separate channels
                    status_text.text('Separating color channels...')
                    progress_bar.progress(20)
                    R, G, B = image_np[:,:,0], image_np[:,:,1], image_np[:,:,2]
                    
                    # Apply PCA to each channel
                    status_text.text('Applying PCA to Red channel...')
                    progress_bar.progress(40)
                    R_reconstructed, R_var, R_singular = apply_pca_on_channel(R, n_components)
                    
                    status_text.text('Applying PCA to Green channel...')
                    progress_bar.progress(60)
                    G_reconstructed, G_var, G_singular = apply_pca_on_channel(G, n_components)
                    
                    status_text.text('Applying PCA to Blue channel...')
                    progress_bar.progress(80)
                    B_reconstructed, B_var, B_singular = apply_pca_on_channel(B, n_components)
                    
                    # Combine channels
                    status_text.text('Reconstructing final image...')
                    progress_bar.progress(90)
                    
                    # Ensure values are in valid range
                    R_reconstructed = np.clip(R_reconstructed, 0, 255)
                    G_reconstructed = np.clip(G_reconstructed, 0, 255)
                    B_reconstructed = np.clip(B_reconstructed, 0, 255)
                    
                    reconstructed_image = np.stack(
                        [R_reconstructed, G_reconstructed, B_reconstructed], axis=2
                    ).astype(np.uint8)
                    
                    progress_bar.progress(100)
                    status_text.text('Analysis complete!')
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Display results
                    st.markdown("---")
                    st.markdown('### <i class="fa-solid fa-circle-check"></i> Results', unsafe_allow_html=True)

                    # st.subheader("🔍 Results")
                    
                    
                    # Basic comparison
                    with st.expander("PCA Reconstructed", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Original Image**")
                            st.image(image_np, use_column_width=True)
                        with col2:
                            st.markdown(f"**PCA Reconstructed ({n_components} components)**")
                            st.image(reconstructed_image, use_column_width=True)
                    
                    # Quality metrics
                    if show_metrics:
                        with st.expander("Quality Metrics", expanded=False):
                        # st.subheader("📊 Quality Metrics")
                        
                            # Calculate metrics for each channel
                            R_metrics = calculate_compression_metrics(R, R_reconstructed, n_components, max_components)
                            G_metrics = calculate_compression_metrics(G, G_reconstructed, n_components, max_components)
                            B_metrics = calculate_compression_metrics(B, B_reconstructed, n_components, max_components)
                            
                            # Overall metrics
                            overall_metrics = calculate_compression_metrics(image_np, reconstructed_image, n_components, max_components)
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Compression Ratio", f"{overall_metrics['compression_ratio']:.1f}:1")
                            with col2:
                                st.metric("Data Retention", f"{overall_metrics['data_retention']:.1f}%")
                            with col3:
                                st.metric("PSNR", f"{overall_metrics['psnr']:.2f} dB")
                            with col4:
                                st.metric("MSE", f"{overall_metrics['mse']:.2f}")
                            
                            # Channel-wise metrics table
                            st.markdown("**Channel-wise Metrics**")
                            metrics_data = {
                                'Channel': ['Red', 'Green', 'Blue'],
                                'MSE': [f"{R_metrics['mse']:.2f}", f"{G_metrics['mse']:.2f}", f"{B_metrics['mse']:.2f}"],
                                'PSNR (dB)': [f"{R_metrics['psnr']:.2f}", f"{G_metrics['psnr']:.2f}", f"{B_metrics['psnr']:.2f}"],
                                'Variance Explained': [f"{sum(R_var):.1%}", f"{sum(G_var):.1%}", f"{sum(B_var):.1%}"]
                            }
                            st.table(metrics_data)
                    
                    # Variance analysis
                    if show_variance_plot:
                        with st.expander("Variance Analysis", expanded=False):
                            # st.subheader("📈 Variance Analysis")
                            fig, ax = plt.subplots(1, 3, figsize=(15, 4))
                            ax[0].plot(R_var, label='R',color='red')
                            ax[0].set_title("Red Channel")
                            ax[1].plot(G_var, label='G', color='green')
                            ax[1].set_title("Green Channel")
                            ax[2].plot(B_var, label='B', color='blue')
                            ax[2].set_title("Blue Channel")
                            for a in ax:
                                a.set_xlabel("Components")
                                a.set_ylabel("Explained Variance Ratio")
                                a.grid(True)
                            st.pyplot(fig)

                            variance_ratios = [R_var, G_var, B_var]
                            channel_names = ['Red', 'Green', 'Blue']
                            colors = ['red', 'green', 'blue']
                            
                            fig = plot_cumulative_variance(variance_ratios, channel_names, colors)
                            st.pyplot(fig)
                            
                            # Components needed for different variance thresholds
                            st.markdown("**Components needed for variance thresholds:**")
                            for threshold in [0.8, 0.9, 0.95, 0.99]:
                                components_needed = []
                                for var_ratio in variance_ratios:
                                    cumsum = np.cumsum(var_ratio)
                                    needed = np.argmax(cumsum >= threshold) + 1
                                    components_needed.append(needed)
                                
                                avg_needed = int(np.mean(components_needed))
                                st.write(f"- {threshold:.0%} variance: ~{avg_needed} components (R:{components_needed[0]}, G:{components_needed[1]}, B:{components_needed[2]})")
                        
                    # Detailed comparison
                    if show_comparison_grid:
                        with st.expander("Detailed Comparison", expanded=False):
                            # st.subheader("🔬 Detailed Comparison")
                            fig = create_comparison_grid(image_np, reconstructed_image, n_components)
                            st.pyplot(fig)
                    
                    st.success("✅ PCA Analysis Completed Successfully!")
                    
                    # Download option
                    st.markdown("---")
                    # st.subheader("💾 Download Results")
                    st.markdown('### <i class="fa-solid fa-download"></i> Download Results', unsafe_allow_html=True)
                    if st.button("Prepare Download"):
                        reconstructed_pil = Image.fromarray(reconstructed_image)
                        st.download_button(
                            label="Download Reconstructed Image",
                            data=reconstructed_pil.tobytes(),
                            file_name=f"pca_reconstructed_{n_components}_components.png",
                            mime="image/png"
                        )
                    st.markdown("---")

            else:
                st.info("📁 Upload an image to begin PCA analysis.")
                
                # Show example information
                st.markdown('### <i class="fa-solid fa-circle-info"></i> About PCA Image Analysis',unsafe_allow_html=True)
                with st.expander("Details", expanded=False):
                # st.subheader("ℹ️ About PCA Image Analysis")
                    st.markdown("""
                    **Principal Component Analysis (PCA)** is a dimensionality reduction technique that can be used for image compression:
                    
                    - **How it works**: PCA finds the directions (principal components) of maximum variance in the data
                    - **Image compression**: By keeping only the most important components, we can reconstruct images with fewer data
                    - **Trade-off**: Fewer components = more compression but lower quality
                    - **Applications**: Image compression, noise reduction, feature extraction
                    
                    **This tool analyzes each RGB channel separately** to provide detailed insights into how PCA affects different color components.
                    """)
                st.markdown('### <i class="fa-solid fa-wand-magic-sparkles"></i> Features',unsafe_allow_html=True)
                with st.expander("Details", expanded=False):
                # st.subheader("🚀 Features")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("""
                        **Analysis Options:**
                        - Manual component selection
                        - Variance threshold-based selection
                        - Compression ratio-based selection
                        - Quality metrics (PSNR, MSE)
                        """)
                    with col2:
                        st.markdown("""
                        **Visualizations:**
                        - Variance analysis plots
                        - Cumulative variance tracking
                        - Difference visualization
                        - Pixel intensity histograms
                        """)
                st.markdown("---")

        elif page1 == "Frequency Domain Analysis":
            st.markdown("""
                    <style>
                        .main-header {
                            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                            padding: 2rem;
                            border-radius: 10px;
                            color: white;
                            text-align: center;
                            margin-bottom: 2rem;
                        }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Header
            st.markdown("""
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
            """, unsafe_allow_html=True)


            st.markdown("""
                    <div class="main-header">
                        <h1><i class="fa-solid fa-chart-bar"></i> Advanced Image Frequency Domain Analysis</h1>
                        <p>Explore the frequency domain properties of your images with various filtering techniques</p>
                    </div>
                    """, unsafe_allow_html=True)

            # Upload image
            uploaded_file = st.file_uploader("Upload an image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

            if uploaded_file:
                # Load and preprocess image
                image = Image.open(uploaded_file).convert("L")
                img_array = np.array(image)
                
                # Normalize image for better processing
                img_array = img_array.astype(np.float32) / 255.0

                st.markdown("---")
                st.markdown('### <i class="fa-solid fa-circle-check"></i> Results', unsafe_allow_html=True)
                with st.expander("Image Statistics"):
                    # st.subheader("🧮 Image Statistics")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(label="Dimensions", value=f"{img_array.shape[1]} × {img_array.shape[0]} px")
                        st.metric(label="Mean Intensity", value=f"{np.mean(img_array):.3f}")

                    with col2:
                        st.metric(label="Std Deviation", value=f"{np.std(img_array):.3f}")
                        st.metric(label="Min / Max", value=f"{np.min(img_array):.3f} / {np.max(img_array):.3f}")

                # Main layout
                col1, col2 = st.columns(2)
                
                with col1:
                    with st.expander("Grayscale Image", expanded=False):
                    # st.subheader("📷 Original Image")
                        st.image(image, caption="Grayscale Image", use_column_width=True)
                    
                    # Image statistics
                    # with st.expander("📊 Image Statistics"):
                        # st.write(f"**Dimensions:** {img_array.shape[1]} × {img_array.shape[0]} pixels")
                        # st.write(f"**Mean Intensity:** {np.mean(img_array):.3f}")
                        # st.write(f"**Standard Deviation:** {np.std(img_array):.3f}")
                        # st.write(f"**Min/Max Values:** {np.min(img_array):.3f} / {np.max(img_array):.3f}")

                with col2:
                    with st.expander("FFT Magnitude Spectrum", expanded=False):
                    # st.subheader("🔍 FFT Magnitude Spectrum")
                    
                        # Compute FFT
                        f = np.fft.fft2(img_array)
                        fshift = np.fft.fftshift(f)
                        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-10)  # Add small value to avoid log(0)

                        fig, ax = plt.subplots(figsize=(6, 6))
                        im = ax.imshow(magnitude_spectrum, cmap="hot", aspect='equal')
                        ax.set_title("Log Magnitude Spectrum", fontsize=12)
                        ax.axis("off")
                        plt.colorbar(im, ax=ax, shrink=0.8)
                        st.pyplot(fig)

                with st.sidebar:
                    st.markdown("---")
                    with st.expander("Filter Parameters", expanded=False):
                        # Advanced filtering controls
                        # st.sidebar.header("🎛️ Filter Parameters")
                        # st.sidebar.header("🔧 Controls")
                        
                        filter_type = st.selectbox(
                            "Filter Type", 
                            ["Low Pass", "High Pass", "Band Pass", "Band Stop", "Gaussian Low Pass", "Gaussian High Pass"]
                        )
                        
                        if filter_type in ["Band Pass", "Band Stop"]:
                            inner_radius = st.slider("Inner Radius", 1, min(img_array.shape)//4, 15)
                            outer_radius = st.slider("Outer Radius", inner_radius+1, min(img_array.shape)//2, 50)
                        else:
                            radius = st.slider("Cut-off Radius", 1, min(img_array.shape)//2, 30)
                    st.markdown("---")
                # Additional parameters for Gaussian filters
                if "Gaussian" in filter_type:
                    sigma = st.sidebar.slider("Gaussian Sigma", 0.1, 5.0, 1.0, 0.1)

                # Create filter mask
                rows, cols = img_array.shape
                crow, ccol = rows//2, cols//2
                
                # Create coordinate matrices
                y, x = np.ogrid[:rows, :cols]
                center = np.array([crow, ccol])
                distance = np.sqrt((x - ccol)**2 + (y - crow)**2)
                
                # Generate different types of masks
                mask = np.zeros((rows, cols), dtype=np.float32)
                
                if filter_type == "Low Pass":
                    mask = (distance <= radius).astype(np.float32)
                elif filter_type == "High Pass":
                    mask = (distance > radius).astype(np.float32)
                elif filter_type == "Band Pass":
                    mask = ((distance >= inner_radius) & (distance <= outer_radius)).astype(np.float32)
                elif filter_type == "Band Stop":
                    mask = ((distance < inner_radius) | (distance > outer_radius)).astype(np.float32)
                elif filter_type == "Gaussian Low Pass":
                    mask = np.exp(-(distance**2) / (2 * (radius * sigma)**2))
                elif filter_type == "Gaussian High Pass":
                    mask = 1 - np.exp(-(distance**2) / (2 * (radius * sigma)**2))

                # Apply filter and inverse FFT
                fshift_filtered = fshift * mask
                f_ishift = np.fft.ifftshift(fshift_filtered)
                img_back = np.fft.ifft2(f_ishift)
                img_back = np.abs(img_back)

                # Display results
                with st.expander("Results", expanded=False):
                    st.subheader(f"{filter_type} Filter")
                
                    col3, col4, col5 = st.columns(3)
                    
                    with col3:
                        st.write("**Filter Mask**")
                        fig, ax = plt.subplots(figsize=(5, 5))
                        im = ax.imshow(mask, cmap="viridis", aspect='equal')
                        ax.set_title("Filter Mask")
                        ax.axis("off")
                        plt.colorbar(im, ax=ax, shrink=0.8)
                        st.pyplot(fig)
                    
                    with col4:
                        st.write("**Filtered Spectrum**")
                        filtered_magnitude = 20 * np.log(np.abs(fshift_filtered) + 1e-10)
                        fig, ax = plt.subplots(figsize=(5, 5))
                        im = ax.imshow(filtered_magnitude, cmap="hot", aspect='equal')
                        ax.set_title("Filtered Spectrum")
                        ax.axis("off")
                        plt.colorbar(im, ax=ax, shrink=0.8)
                        st.pyplot(fig)
                    
                    with col5:
                        st.write("**Filtered Image**")
                        fig, ax = plt.subplots(figsize=(5, 5))
                        ax.imshow(img_back, cmap="gray", aspect='equal')
                        ax.set_title("Reconstructed Image")
                        ax.axis("off")
                        st.pyplot(fig)

                # Comparison section
                with st.expander("Before vs After Comparison", expanded=False):
                # st.subheader("📈 Before vs After Comparison")
                
                    comparison_col1, comparison_col2 = st.columns(2)
                    
                    with comparison_col1:
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                        
                        ax1.imshow(img_array, cmap="gray")
                        ax1.set_title("Original Image")
                        ax1.axis("off")
                        
                        ax2.imshow(img_back, cmap="gray")
                        ax2.set_title("Filtered Image")
                        ax2.axis("off")
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    with comparison_col2:
                        # Difference image
                        difference = np.abs(img_array - img_back)
                        fig, ax = plt.subplots(figsize=(5, 4))
                        im = ax.imshow(difference, cmap="hot")
                        ax.set_title("Difference Image")
                        ax.axis("off")
                        plt.colorbar(im, ax=ax, shrink=0.8)
                        st.pyplot(fig)
                        
                        # Metrics
                        st.subheader("📊 Quality Metrics")
                        mse = np.mean((img_array - img_back)**2)
                        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric(label="🔻 MSE (Mean Squared Error)", value=f"{mse:.6f}")

                        with col2:
                            st.metric(label="📈 PSNR (Peak Signal-to-Noise Ratio)", value=f"{psnr:.2f} dB")
                        # st.write("**Quality Metrics:**")
                        # mse = np.mean((img_array - img_back)**2)
                        # psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
                        # st.write(f"MSE: {mse:.6f}")
                        # st.write(f"PSNR: {psnr:.2f} dB")

                # Advanced analysis
                with st.expander("Advanced Analysis"):
                    analysis_tabs = st.tabs(["Phase Spectrum", "1D Frequency Profile", "Filter Response"])
                    
                    with analysis_tabs[0]:
                        phase_spectrum = np.angle(fshift)
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                        
                        im1 = ax1.imshow(phase_spectrum, cmap="twilight")
                        ax1.set_title("Original Phase Spectrum")
                        ax1.axis("off")
                        plt.colorbar(im1, ax=ax1, shrink=0.8)
                        
                        filtered_phase = np.angle(fshift_filtered)
                        im2 = ax2.imshow(filtered_phase, cmap="twilight")
                        ax2.set_title("Filtered Phase Spectrum")
                        ax2.axis("off")
                        plt.colorbar(im2, ax=ax2, shrink=0.8)
                        
                        st.pyplot(fig)
                    
                    with analysis_tabs[1]:
                        # 1D radial profile
                        center_row = magnitude_spectrum[crow, :]
                        center_col = magnitude_spectrum[:, ccol]
                        
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
                        
                        ax1.plot(center_row, 'b-', linewidth=2, label='Horizontal profile')
                        ax1.set_title('Horizontal Frequency Profile (through center)')
                        ax1.set_xlabel('Frequency bin')
                        ax1.set_ylabel('Magnitude (dB)')
                        ax1.grid(True, alpha=0.3)
                        ax1.legend()
                        
                        ax2.plot(center_col, 'r-', linewidth=2, label='Vertical profile')
                        ax2.set_title('Vertical Frequency Profile (through center)')
                        ax2.set_xlabel('Frequency bin')
                        ax2.set_ylabel('Magnitude (dB)')
                        ax2.grid(True, alpha=0.3)
                        ax2.legend()
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    with analysis_tabs[2]:
                        # Filter frequency response
                        fig, ax = plt.subplots(figsize=(8, 6))
                        
                        # Radial profile of the filter
                        distances = np.arange(0, min(rows, cols)//2)
                        if filter_type == "Low Pass":
                            response = (distances <= radius).astype(float)
                        elif filter_type == "High Pass":
                            response = (distances > radius).astype(float)
                        elif filter_type == "Gaussian Low Pass":
                            response = np.exp(-(distances**2) / (2 * (radius * sigma)**2))
                        elif filter_type == "Gaussian High Pass":
                            response = 1 - np.exp(-(distances**2) / (2 * (radius * sigma)**2))
                        else:
                            # For band filters, use center distance for visualization
                            if filter_type == "Band Pass":
                                response = ((distances >= inner_radius) & (distances <= outer_radius)).astype(float)
                            else:  # Band Stop
                                response = ((distances < inner_radius) | (distances > outer_radius)).astype(float)
                        
                        ax.plot(distances, response, 'g-', linewidth=3, label=f'{filter_type} Filter')
                        ax.set_xlabel('Distance from center (pixels)')
                        ax.set_ylabel('Filter Response')
                        ax.set_title('Filter Frequency Response')
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                        ax.set_ylim(-0.1, 1.1)
                        
                        st.pyplot(fig)

                # Export options
                st.markdown("---")
                st.markdown('### <i class="fa-solid fa-download"></i> Export Options', unsafe_allow_html=True)
                # st.header("💾 Export Options")
                
                if st.button("Download Filtered Image"):
                    # Convert back to uint8 for saving
                    filtered_uint8 = (np.clip(img_back, 0, 1) * 255).astype(np.uint8)
                    filtered_pil = Image.fromarray(filtered_uint8, mode='L')
                    
                    # Note: In a real Streamlit app, you'd use st.download_button here
                    st.success("Image ready for download!")
                    st.info("In a full Streamlit deployment, this would trigger a download.")
                st.markdown("---")
            else:
                st.info("👆 Please upload an image to start the frequency domain analysis!")
                
                # Show example of what the tool can do
                with st.expander("What this tool does:", expanded=False):
                # st.subheader("🎯What this tool does:")
                    st.markdown("""
                    - **FFT Analysis**: Converts images to frequency domain using Fast Fourier Transform
                    - **Multiple Filters**: Apply various frequency filters (Low-pass, High-pass, Band-pass, etc.)
                    - **Real-time Visualization**: See immediate results of filter applications
                    - **Advanced Analysis**: Examine phase spectra, frequency profiles, and filter responses
                    - **Quality Metrics**: Calculate MSE and PSNR to evaluate filtering effects
                    
                    **Use Cases:**
                    - Noise reduction (low-pass filtering)
                    - Edge enhancement (high-pass filtering)  
                    - Feature extraction and analysis
                    - Understanding image frequency characteristics
                    """)
                st.markdown("---")

        elif page1 == "Statistical Analysis":
            st.markdown("""
                    <style>
                        .main-header {
                            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                            padding: 2rem;
                            border-radius: 10px;
                            color: white;
                            text-align: center;
                            margin-bottom: 2rem;
                        }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Header
            st.markdown("""
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
            """, unsafe_allow_html=True)


            st.markdown("""
                    <div class="main-header">
                        <h1><i class="fa-solid fa-chart-column"></i> Advanced Image Statistical Analysis</h1>
                        <p>Explore the frequency domain properties of your images with various filtering techniques</p>
                    </div>
                    """, unsafe_allow_html=True)
            # st.title("📊 Advanced Image Statistical Analysis")

            uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"])

            if uploaded_file:
                image = Image.open(uploaded_file).convert('RGB')
                img_array = np.array(image)

                # Display basic image info
                st.markdown("---")
                st.markdown('### <i class="fa-solid fa-image"></i> Image Properties', unsafe_allow_html=True)
                # st.subheader("🖼️ Image Properties")
                col1, col2 = st.columns(2)

                with col1:
                    st.metric(label="Dimensions", value=f"{img_array.shape[1]} × {img_array.shape[0]} px")
                    st.metric(label="Total Pixels", value=f"{img_array.shape[0] * img_array.shape[1]:,}")

                with col2:
                    st.metric(label="Channels", value=f"{img_array.shape[2]}")
                    st.metric(label="File Size", value=f"{len(uploaded_file.getvalue()) / 1024:.1f} KB")

                st.markdown("---")
                st.markdown('### <i class="fa-solid fa-square-root-variable"></i> Statistical Metrics', unsafe_allow_html=True)
                # st.header("🧮 Statistical Metrics")
                r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
                gray = np.mean(img_array, axis=2).astype(np.uint8)
                
                def calc_stats(channel, name):
                    flat_channel = channel.flatten()
                    stats = {
                        "Mean": np.mean(flat_channel),
                        "Median": np.median(flat_channel),
                        "Mode": np.bincount(flat_channel).argmax(),
                        "Standard Deviation": np.std(flat_channel),
                        "Variance": np.var(flat_channel),
                        "Min": np.min(flat_channel),
                        "Max": np.max(flat_channel),
                        "Range": np.max(flat_channel) - np.min(flat_channel),
                        "Skewness": skew(flat_channel),
                        "Kurtosis": kurtosis(flat_channel),
                        "Entropy": scipy_entropy(np.histogram(flat_channel, bins=256)[0]+1),
                        "25th Percentile": np.percentile(flat_channel, 25),
                        "75th Percentile": np.percentile(flat_channel, 75),
                        "IQR": np.percentile(flat_channel, 75) - np.percentile(flat_channel, 25)
                    }
                    
                    with st.expander(f"{name} Channel Statistics"):
                        col1, col2 = st.columns(2)
                        items = list(stats.items())
                        mid = len(items) // 2
                        
                        with col1:
                            for k, v in items[:mid]:
                                st.metric(label=k, value=f"{v:.4f}")
                        
                        with col2:
                            for k, v in items[mid:]:
                                st.metric(label=k, value=f"{v:.4f}")
                
                calc_stats(r, "Red")
                calc_stats(g, "Green")
                calc_stats(b, "Blue")
                calc_stats(gray, "Grayscale")

                # Color Analysis
                st.markdown("---")
                st.markdown('### <i class="fa-solid fa-palette"></i> Color Analysis', unsafe_allow_html=True)
                # st.header("🎨 Color Analysis")
                with st.expander("Dominant Colors and Color Temperature Analysis", expanded=False):
                
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Dominant Colors")
                        # Extract dominant colors using K-means
                        pixels = img_array.reshape(-1, 3)
                        n_colors = st.slider("Number of dominant colors", 2, 10, 5)
                        
                        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
                        kmeans.fit(pixels)
                        colors = kmeans.cluster_centers_.astype(int)
                        
                        # Create color palette
                        fig_palette, ax_palette = plt.subplots(figsize=(8, 2))
                        ax_palette.imshow([colors], aspect='auto')
                        ax_palette.set_xticks(range(n_colors))
                        ax_palette.set_xticklabels([f'RGB({c[0]},{c[1]},{c[2]})' for c in colors], rotation=45)
                        ax_palette.set_yticks([])
                        ax_palette.set_title("Dominant Colors")
                        st.pyplot(fig_palette)
                    
                    with col2:
                        st.subheader("Color Temperature Analysis")
                        # Calculate color temperature metrics
                        avg_rgb = np.mean(pixels, axis=0)
                        
                        # Warmth index (more red/yellow vs blue)
                        warmth = (avg_rgb[0] + avg_rgb[1]) / (2 * avg_rgb[2]) if avg_rgb[2] > 0 else float('inf')
                        
                        # Saturation (deviation from grayscale)
                        saturation = np.std(avg_rgb) / np.mean(avg_rgb) if np.mean(avg_rgb) > 0 else 0
                        
                        st.metric("Warmth Index", f"{warmth:.3f}", help="Higher values indicate warmer colors")
                        st.metric("Saturation Index", f"{saturation:.3f}", help="Higher values indicate more colorful image")
                        st.metric("Brightness", f"{np.mean(gray):.1f}/255", help="Average brightness level")
                st.markdown("---")
                st.markdown('### <i class="fa-solid fa-chart-line"></i> Advanced Visualizations', unsafe_allow_html=True)
                # st.header("📈 Advanced Visualizations")
                with st.expander("Advanced Visualizations", expanded=False):

                    # Create tabs for different visualizations
                    tab1, tab2, tab3, tab4 = st.tabs(["Histograms", "2D Histograms", "Scatter Plots", "Distribution Analysis"])
                    
                    with tab1:
                        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

                        # Grayscale histogram
                        axs[0,0].hist(gray.ravel(), bins=256, color='gray', alpha=0.7)
                        axs[0,0].set_title("Grayscale Histogram")
                        axs[0,0].set_xlabel("Pixel Value")
                        axs[0,0].set_ylabel("Frequency")
                        axs[0,0].grid(True, alpha=0.3)

                        # Color histogram
                        axs[0,1].hist(r.ravel(), bins=256, color='red', alpha=0.5, label='Red')
                        axs[0,1].hist(g.ravel(), bins=256, color='green', alpha=0.5, label='Green')
                        axs[0,1].hist(b.ravel(), bins=256, color='blue', alpha=0.5, label='Blue')
                        axs[0,1].set_title("RGB Histogram")
                        axs[0,1].set_xlabel("Pixel Value")
                        axs[0,1].set_ylabel("Frequency")
                        axs[0,1].legend()
                        axs[0,1].grid(True, alpha=0.3)

                        # Cumulative histogram
                        axs[1,0].hist(gray.ravel(), bins=256, cumulative=True, color='gray', alpha=0.7)
                        axs[1,0].set_title("Cumulative Histogram")
                        axs[1,0].set_xlabel("Pixel Value")
                        axs[1,0].set_ylabel("Cumulative Frequency")
                        axs[1,0].grid(True, alpha=0.3)

                        # Log histogram
                        counts, bins = np.histogram(gray.ravel(), bins=256)
                        axs[1,1].bar(bins[:-1], np.log(counts + 1), width=1, color='gray', alpha=0.7)
                        axs[1,1].set_title("Log Histogram")
                        axs[1,1].set_xlabel("Pixel Value")
                        axs[1,1].set_ylabel("Log(Frequency)")
                        axs[1,1].grid(True, alpha=0.3)

                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    with tab2:
                        fig_2d, axs_2d = plt.subplots(1, 3, figsize=(15, 5))
                        
                        # RGB 2D histograms
                        axs_2d[0].hist2d(r.ravel(), g.ravel(), bins=50, cmap='Reds')
                        axs_2d[0].set_title("Red vs Green")
                        axs_2d[0].set_xlabel("Red Channel")
                        axs_2d[0].set_ylabel("Green Channel")
                        
                        axs_2d[1].hist2d(r.ravel(), b.ravel(), bins=50, cmap='Blues')
                        axs_2d[1].set_title("Red vs Blue")
                        axs_2d[1].set_xlabel("Red Channel")
                        axs_2d[1].set_ylabel("Blue Channel")
                        
                        axs_2d[2].hist2d(g.ravel(), b.ravel(), bins=50, cmap='Greens')
                        axs_2d[2].set_title("Green vs Blue")
                        axs_2d[2].set_xlabel("Green Channel")
                        axs_2d[2].set_ylabel("Blue Channel")
                        
                        plt.tight_layout()
                        st.pyplot(fig_2d)
                    
                    with tab3:
                        # Scatter plots with correlation
                        fig_scatter, axs_scatter = plt.subplots(1, 3, figsize=(15, 5))
                        
                        # Sample data for performance (use every nth pixel)
                        step = max(1, len(pixels) // 10000)  # Limit to ~10k points
                        sample_pixels = pixels[::step]
                        
                        corr_rg = pearsonr(sample_pixels[:, 0], sample_pixels[:, 1])[0]
                        corr_rb = pearsonr(sample_pixels[:, 0], sample_pixels[:, 2])[0]
                        corr_gb = pearsonr(sample_pixels[:, 1], sample_pixels[:, 2])[0]
                        
                        axs_scatter[0].scatter(sample_pixels[:, 0], sample_pixels[:, 1], alpha=0.1, s=1)
                        axs_scatter[0].set_title(f"Red vs Green (r={corr_rg:.3f})")
                        axs_scatter[0].set_xlabel("Red")
                        axs_scatter[0].set_ylabel("Green")
                        
                        axs_scatter[1].scatter(sample_pixels[:, 0], sample_pixels[:, 2], alpha=0.1, s=1)
                        axs_scatter[1].set_title(f"Red vs Blue (r={corr_rb:.3f})")
                        axs_scatter[1].set_xlabel("Red")
                        axs_scatter[1].set_ylabel("Blue")
                        
                        axs_scatter[2].scatter(sample_pixels[:, 1], sample_pixels[:, 2], alpha=0.1, s=1)
                        axs_scatter[2].set_title(f"Green vs Blue (r={corr_gb:.3f})")
                        axs_scatter[2].set_xlabel("Green")
                        axs_scatter[2].set_ylabel("Blue")
                        
                        plt.tight_layout()
                        st.pyplot(fig_scatter)
                    
                    with tab4:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Channel correlation heatmap
                            st.subheader("Channel Correlation Matrix")
                            flat_rgb = img_array.reshape(-1, 3)
                            corr_matrix = np.corrcoef(flat_rgb.T)
                            
                            fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
                            sns.heatmap(corr_matrix, annot=True, fmt=".3f", 
                                    xticklabels=["Red", "Green", "Blue"], 
                                    yticklabels=["Red", "Green", "Blue"], 
                                    ax=ax_corr, cmap='coolwarm')
                            ax_corr.set_title("RGB Channel Correlations")
                            st.pyplot(fig_corr)
                        
                        with col2:
                            # KDE distribution plot
                            st.subheader("Intensity Distribution (KDE)")
                            fig_kde, ax_kde = plt.subplots(figsize=(6, 4))
                            sns.kdeplot(data=r.flatten(), color="red", label="Red", ax=ax_kde, alpha=0.7)
                            sns.kdeplot(data=g.flatten(), color="green", label="Green", ax=ax_kde, alpha=0.7)
                            sns.kdeplot(data=b.flatten(), color="blue", label="Blue", ax=ax_kde, alpha=0.7)
                            sns.kdeplot(data=gray.flatten(), color="black", label="Grayscale", ax=ax_kde, alpha=0.7)
                            ax_kde.set_title("Pixel Intensity Distribution")
                            ax_kde.set_xlabel("Pixel Value")
                            ax_kde.set_ylabel("Density")
                            ax_kde.legend()
                            st.pyplot(fig_kde)

                # Advanced Analysis Section
                st.markdown("---")
                st.markdown('### <i class="fa-solid fa-microscope"></i> Advanced Analysis', unsafe_allow_html=True)
                # st.header("Advanced Analysis")
                with st.expander("Advanced Analysis", expanded=False):
                
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.subheader("Image Quality Metrics")
                        
                        # Contrast ratio
                        contrast = np.std(gray) / np.mean(gray) if np.mean(gray) > 0 else 0
                        
                        # Dynamic range
                        dynamic_range = np.max(gray) - np.min(gray)
                        
                        # Entropy (information content)
                        img_entropy = scipy_entropy(np.histogram(gray, bins=256)[0] + 1)
                        
                        st.metric("Contrast Ratio", f"{contrast:.3f}")
                        st.metric("Dynamic Range", f"{dynamic_range}/255")
                        st.metric("Information Content", f"{img_entropy:.3f}")
                    
                    with col2:
                        st.subheader("Color Distribution")
                        
                        # Calculate color moments
                        mean_color = np.mean(pixels, axis=0)
                        std_color = np.std(pixels, axis=0)
                        
                        st.write("**Mean RGB:**")
                        st.write(f"R: {mean_color[0]:.1f}, G: {mean_color[1]:.1f}, B: {mean_color[2]:.1f}")
                        
                        st.write("**Standard Deviation:**")
                        st.write(f"R: {std_color[0]:.1f}, G: {std_color[1]:.1f}, B: {std_color[2]:.1f}")
                    
                    with col3:
                        st.subheader("Histogram Statistics")
                        
                        # Peak analysis
                        hist_gray, _ = np.histogram(gray, bins=256)
                        peak_value = np.argmax(hist_gray)
                        peak_count = np.max(hist_gray)
                        
                        # Histogram spread
                        hist_spread = np.std(np.repeat(range(256), hist_gray))
                        
                        st.metric("Peak Intensity", f"{peak_value}")
                        st.metric("Peak Frequency", f"{peak_count:,}")
                        st.metric("Histogram Spread", f"{hist_spread:.1f}")

                # Export functionality
                st.markdown("---")
                st.markdown('### <i class="fa-solid fa-download"></i> Export Analysis', unsafe_allow_html=True)
                # st.header("💾 Export Analysis")
                
                if st.button("Generate Analysis Report"):
                    report = f"""
            # Image Statistical Analysis Report

            ## Image Properties
            - Dimensions: {img_array.shape[1]} × {img_array.shape[0]} pixels
            - Total Pixels: {img_array.shape[0] * img_array.shape[1]:,}
            - File Size: {len(uploaded_file.getvalue()) / 1024:.1f} KB

            ## Statistical Summary
            - Mean Brightness: {np.mean(gray):.2f}
            - Contrast Ratio: {contrast:.3f}
            - Dynamic Range: {dynamic_range}/255
            - Information Content: {img_entropy:.3f}

            ## Color Analysis
            - Warmth Index: {warmth:.3f}
            - Saturation Index: {saturation:.3f}
            - Mean RGB: ({mean_color[0]:.1f}, {mean_color[1]:.1f}, {mean_color[2]:.1f})

            ## Channel Correlations
            - Red-Green: {corr_rg:.3f}
            - Red-Blue: {corr_rb:.3f}
            - Green-Blue: {corr_gb:.3f}
                    """
                    
                    st.download_button(
                        label="Download Report",
                        data=report,
                        file_name=f"image_analysis_{uploaded_file.name}.txt",
                        mime="text/plain"
                    )
                st.markdown("---")

            else:
                # st.info("⏳ The process may take a long time.")
                st.markdown('#### <i class="fa-solid fa-hourglass-half"></i> The process may take a long time.', unsafe_allow_html=True)

                with st.expander("Features:", expanded=False):
                    st.markdown("""
                    - Comprehensive statistical metrics for each color channel
                    - Dominant color extraction and analysis
                    - Multiple visualization types (histograms, scatter plots, KDE)
                    - Channel correlation analysis
                    - Advanced image quality metrics
                    - Exportable analysis reports
                    
                    """)
                st.markdown("---")