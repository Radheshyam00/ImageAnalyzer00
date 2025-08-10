import streamlit as st

# Enhanced page configuration
st.set_page_config(
    page_title="Image Analyzer Suite",
    page_icon="./assets/image.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    /* Base Tool Card Styling */
    .tool-card {
        padding: 1.5rem;
        border-radius: 12px;
        transition: all 0.3s ease;
        height: 100%;
        margin-bottom: 1rem;
        border: 1px solid;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }

    /* Light mode */
    @media (prefers-color-scheme: light) {
        .tool-card {
            background: white;
            border-color: #e0e0e0;
            color: #000;
        }

        .tool-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
            border-color: #667eea;
        }
    }

    /* Dark mode */
    @media (prefers-color-scheme: dark) {
        .tool-card {
            background: #1e1e1e;
            border-color: #333;
            color: #eee;
            box-shadow: 0 4px 20px rgba(255, 255, 255, 0.05);
        }

        .tool-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 30px rgba(255, 255, 255, 0.1);
            border-color: #8895f6;
        }
    }

    
    /* Feature list styling */
    .feature-list {
        list-style: none;
        padding: 0;
    }
    
    .feature-item {
        padding: 0.5rem 0;
        border-bottom: 1px solid #f0f0f0;
        display: flex;
        align-items: center;
    }
    
    .feature-item:last-child {
        border-bottom: none;
    }
    
    .feature-icon {
        margin-right: 0.5rem;
        font-size: 1.1em;
    }
    
    /* Launch button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        margin-top: 1rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        text-align: center;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 2rem 0;
        border-top: 1px solid #e0e0e0;
        color: #666;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: #f8f9fa;
        border-radius: 8px;
    }
    
    /* Stats container */
    .stats-container {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 2rem 0;
        text-align: center;
    }
    
    .stat-item {
        display: inline-block;
        margin: 0 2rem;
        text-align: center;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #333;
        display: block;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    /* Light mode */
    @media (prefers-color-scheme: light) {
        /* Selected option text */
        .stSelectbox div[data-baseweb="select"] > div {
            background-color: #ffffff !important;
            color: #000000 !important;
        }

        /* Dropdown options */
        .stSelectbox .css-1wa3eu0 {
            background-color: #ffffff !important;
            color: #000000 !important;
        }

        /* Hover style */
        .stSelectbox .css-1wa3eu0:hover {
            background-color: #f0f0f0 !important;
        }
    }

    /* Dark mode */
    @media (prefers-color-scheme: dark) {
        /* Selected option text */
        .stSelectbox div[data-baseweb="select"] > div {
            background-color: #1e1e1e !important;
            color: #eeeeee !important;
        }

        /* Dropdown options */
        .stSelectbox .css-1wa3eu0 {
            background-color: #1e1e1e !important;
            color: #eeeeee !important;
        }

        /* Hover style */
        .stSelectbox .css-1wa3eu0:hover {
            background-color: #2e2e2e !important;
        }
    }
    /* Base styling */
    .feature-section {
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 1px solid;
        transition: all 0.3s ease;
    }

    /* Light Mode */
    @media (prefers-color-scheme: light) {
        .feature-section {
            background: white;
            border-color: #e0e0e0;
            color: #1f2937;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        }
    }

    /* Dark Mode */
    @media (prefers-color-scheme: dark) {
        .feature-section {
            background: #1e1e1e;
            border-color: #333;
            color: #e5e7eb;
            box-shadow: 0 2px 10px rgba(255, 255, 255, 0.05);
        }
    }

</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <link
      rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
    >
    """,
    unsafe_allow_html=True
)
# Main header
st.markdown("""
<div class="main-header">
    <h1><i class="fa-solid fa-magnifying-glass"></i> Image Analyzer Suite</h1>
    <h3><i class="fa-solid fa-images"></i> Unlock Hidden Details in Your Images</h3>
    <p>Professional toolkit for advanced image examination and digital forensics</p>
</div>
""", unsafe_allow_html=True)

# Welcome section with enhanced styling
st.markdown("""
<div class="info-box">
    <h4>Why Choose Image Analyzer Suite?</h4>
    <p>Comprehensive analysis tools for metadata extraction, forensic investigation, and file security inspection. Perfect for investigators, researchers, and security professionals.</p>
</div>
""", unsafe_allow_html=True)

# Stats section
st.markdown("""
<div class="stats-container">
    <div class="stat-item">
        <span class="stat-number">3</span>
        <span class="stat-label">Powerful Tools</span>
    </div>
    <div class="stat-item">
        <span class="stat-number">15+</span>
        <span class="stat-label">Analysis Types</span>
    </div>
    <div class="stat-item">
        <span class="stat-number">100%</span>
        <span class="stat-label">Privacy Safe</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Tools section with enhanced cards
st.markdown("## <i class=\"fa-solid fa-toolbox\"></i> Available Analysis Tools", unsafe_allow_html=True)


col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("""
    <div class="tool-card">
        <h3 style="color: #667eea; margin-bottom: 1rem;"><i class='fa-solid fa-scroll'></i> Metadata Analyzer</h3>
        <div class="feature-list">
            <div class="feature-item">
                <span class="feature-icon"><i class='fa-solid fa-chart-column'></i></span>
                <span>Extract comprehensive EXIF data</span>
            </div>
            <div class="feature-item">
                <span class="feature-icon"><i class='fa-solid fa-earth-asia'></i></span>
                <span>GPS location & mapping</span>
            </div>
            <div class="feature-item">
                <span class="feature-icon"><i class='fa-solid fa-camera-retro'></i></span>
                <span>Camera settings & device info</span>
            </div>
            <div class="feature-item">
                <span class="feature-icon"><i class='fa-solid fa-ruler'></i></span>
                <span>Detailed format analysis</span>
            </div>
            <div class="feature-item">
                <span class="feature-icon"><i class='fa-solid fa-tags'></i></span>
                <span>IPTC & XMP metadata</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="tool-card">
        <h3 style="color: #f5576c; margin-bottom: 1rem;"><i class='fa-solid fa-magnifying-glass'></i> Forensics Analyzer</h3>
        <div class="feature-list">
            <div class="feature-item">
                <span class="feature-icon"><i class='fa-solid fa-bolt'></i></span>
                <span>Error Level Analysis (ELA)</span>
            </div>
            <div class="feature-item">
                <span class="feature-icon"><i class='fa-solid fa-clone'></i></span>
                <span>Advanced clone detection</span>
            </div>
            <div class="feature-item">
                <span class="feature-icon"><i class='fa-solid fa-lightbulb'></i></span>
                <span>Luminance analysis</span>
            </div>
            <div class="feature-item">
                <span class="feature-icon"><i class='fa-solid fa-square-root-variable'></i></span>
                <span>Principal Component Analysis</span>
            </div>
            <div class="feature-item">
                <span class="feature-icon"><i class='fa-solid fa-shield'></i></span>
                <span>C2PA authentication check</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="tool-card">
        <h3 style="color: #764ba2; margin-bottom: 1rem;"><i class='fa-solid fa-shield-halved'></i> File Inspector Pro</h3>
        <div class="feature-list">
            <div class="feature-item">
                <span class="feature-icon"><i class='fa-solid fa-circle-check'></i></span>
                <span>File type validation</span>
            </div>
            <div class="feature-item">
                <span class="feature-icon"><i class='fa-solid fa-user-secret'></i></span>
                <span>Hidden payload detection</span>
            </div>
            <div class="feature-item">
                <span class="feature-icon"><i class='fa-solid fa-lock'></i></span>
                <span>Encryption analysis</span>
            </div>
            <div class="feature-item">
                <span class="feature-icon"><i class='fa-solid fa-mask'></i></span>
                <span>Steganography detection</span>
            </div>
            <div class="feature-item">
                <span class="feature-icon"><i class='fa-solid fa-triangle-exclamation'></i></span>
                <span>Security threat alerts</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# Enhanced tool selection section
st.markdown("---")
st.markdown("## <i class='fa-solid fa-rocket'></i> Launch Your Analysis", unsafe_allow_html=True)


# Create two columns for better layout
select_col1, select_col2 = st.columns([2, 1])

with select_col1:
    st.markdown("### Select Analysis Tool")
    tool = st.selectbox(
        "Choose the module that best fits your analysis needs:",
        [
            "Advanced Image Metadata Analyzer",
            "Advanced Image Forensics Analyzer", 
            "File Inspector"
        ],
        help="Each tool offers specialized analysis capabilities for different investigation needs."
    )

with select_col2:
    st.markdown("### <i class='fa-solid fa-bolt'></i> Quick Launch", unsafe_allow_html=True)

    # Button with icon
    if st.button("Launch Selected Tool", use_container_width=True):
        if tool == "Advanced Image Metadata Analyzer":
            st.switch_page("pages/metadata.py")
        elif tool == "Advanced Image Forensics Analyzer":
            st.switch_page("pages/forensic.py")
        elif tool == "File Inspector":
            st.switch_page("pages/file.py")

# Additional features section
st.markdown("---")
st.markdown("## <i class='fa-solid fa-star'></i> Key Features", unsafe_allow_html=True)


feature_col1, feature_col2 = st.columns(2)

with feature_col1:
    st.markdown("""
    <div class="feature-section">
        <h3><i class='fa-solid fa-lock'></i> Privacy & Security</h3>
        <ul>
            <li><strong>Local Processing</strong>: All analysis performed on your device</li>
            <li><strong>No Cloud Upload</strong>: Your images never leave your computer</li>
            <li><strong>Zero Data Collection</strong>: No tracking or data retention</li>
            <li><strong>Open Source</strong>: Transparent and auditable code</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with feature_col2:
    st.markdown("""
    <div class="feature-section">
        <h3><i class='fa-solid fa-bullseye'></i> Professional Grade</h3>
        <ul>
            <li><strong>Forensic Quality</strong>: Industry-standard analysis techniques</li>
            <li><strong>Comprehensive Reports</strong>: Detailed findings with visualizations</li>
            <li><strong>Multiple Formats</strong>: Support for all major image formats</li>
            <li><strong>Export Options</strong>: Save results in various formats</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Enhanced footer
st.markdown("""
<div class="footer">
    <h4>Image Analyzer Suite</h4>
    <p><strong>© 2025 Radheshyam Janwa</strong> | Professional Image Analysis Toolkit</p>
    <p><i class='fa-solid fa-lock'></i> <em>Secure • Private • Comprehensive</em> <i class='fa-solid fa-lock'></i></p>
    <hr style="width: 50%; margin: 1rem auto; border-color: #ddd;">
    <small>All Rights Reserved | Built with <i class='fa-solid fa-heart'></i> using Streamlit</small>

</div>
""", unsafe_allow_html=True)