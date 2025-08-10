# Advanced Image Metadata & Security Analyzer(AIMSA)
<div align="center">
# IMAGEANALYZER00

_Unlock Insights, Transform Images, Empower Decisions_

![Last Commit](https://img.shields.io/github/last-commit/Radheshyam00/IMAGEANALYZER00?label=last%20commit)
![Commit Time](https://img.shields.io/badge/yesterday-blue)
![Python](https://img.shields.io/badge/python-100%25-blue)
![Languages](https://img.shields.io/badge/languages-1-lightgrey)

---

**Built with the tools and technologies:**

![Markdown](https://img.shields.io/badge/Markdown-000000?style=for-the-badge&logo=markdown&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Folium](https://img.shields.io/badge/Folium-77B829?style=for-the-badge)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

</div>

## Overview

**ImageAnalyzer** is a powerful Streamlit-based application for analyzing image files. It supports viewing, editing, and removing metadata, detecting steganographic artifacts, and scanning files with VirusTotal for malware analysis. Ideal for digital forensics, cybersecurity, and privacy auditing.

---

## Features

- 📁 File Information: Name, size, MIME type, MD5/SHA256 hash, magic bytes.
- 🧬 EXIF & Metadata Viewer: Camera info, timestamps, GPS, exposure settings.
- ✏️ Metadata Editor & Remover: Modify or erase metadata in JPEGs.
- 🛡️ VirusTotal Integration: Scan images for malware (API key required).
- 🧠 Steganography Detection: Check for hidden data, embedded ZIPs, and entropy anomalies.
- 🎨 ICC Profile & JFIF Metadata: Extract color space and format info.
- 🔎 Header Analysis: Validate file type and structure.
- 📄 Export JSON Report: Full analysis in downloadable format.

---

## Supported Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- TIFF (.tiff)
- BMP (.bmp)
- WebP (.webp)
- GIF (.gif)

---

## Installation


```bash
# Clone the repository
git clone https://github.com/Radheshyam00/ImageAnalyzer00.git
cd ImageAnalyzer00

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Create a .env file in the root directory and add
MONGO_URI=your_mongodb_uri  # optional, for File Inspector Pro
```
---
## Running the App
```bash
streamlit run ImageAnalyzer.py
```

Open your browser to `http://localhost:8501`

---

## Example Use Cases
- Audit photo metadata before sharing online

- Conduct forensic image analysis

- Remove GPS or timestamp information for privacy

- Detect steganographic payloads

- Check for malware using VirusTotal

## LICENSE

[BSD 3-Clause License](LICENSE)