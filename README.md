# 🔍 Advanced Image Metadata & Security Analyzer

## Overview

**ImageAnalyzer** is a powerful Streamlit-based application for analyzing image files. It supports viewing, editing, and removing metadata, detecting steganographic artifacts, and scanning files with VirusTotal for malware analysis. Ideal for digital forensics, cybersecurity, and privacy auditing.

---

## 🚀 Features

- 📁 File Information: Name, size, MIME type, MD5/SHA256 hash, magic bytes.
- 🧬 EXIF & Metadata Viewer: Camera info, timestamps, GPS, exposure settings.
- ✏️ Metadata Editor & Remover: Modify or erase metadata in JPEGs.
- 🛡️ VirusTotal Integration: Scan images for malware (API key required).
- 🧠 Steganography Detection: Check for hidden data, embedded ZIPs, and entropy anomalies.
- 🎨 ICC Profile & JFIF Metadata: Extract color space and format info.
- 🔎 Header Analysis: Validate file type and structure.
- 📄 Export JSON Report: Full analysis in downloadable format.

---

## 🖼️ Supported Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- TIFF (.tiff)
- BMP (.bmp)
- WebP (.webp)
- GIF (.gif)

---

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/image-analyzer.git
cd image-analyzer

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
