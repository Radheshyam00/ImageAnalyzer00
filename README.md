# Advanced Image Metadata & Security Analyzer(AIMSA)
<div align="center">

_Unlock Insights, Transform Images, Empower Decisions_

![Last Commit](https://img.shields.io/github/last-commit/Radheshyam00/IMAGEANALYZER00?label=last%20commit)
![Commit Time](https://img.shields.io/badge/yesterday-blue)
![Python](https://img.shields.io/badge/python-100%25-blue)
![Languages](https://img.shields.io/badge/languages-1-lightgrey)


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

## Supported File Formats

| Format | Extensions | Features |
|--------|------------|----------|
| JPEG   | `.jpg`, `.jpeg` | Full metadata editing, EXIF extraction |
| PNG    | `.png` | Metadata analysis, transparency info |
| TIFF   | `.tiff`, `.tif` | Multi-page support, compression details |
| BMP    | `.bmp` | Basic analysis, header validation |
| WebP   | `.webp` | Modern format support |
| GIF    | `.gif` | Animation frame analysis |

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



# Advanced Image Metadata & Security Analyzer (AIMSA)

<div align="center">

*Unlock Insights, Transform Images, Empower Decisions*

[![Last Commit](https://img.shields.io/github/last-commit/Radheshyam00/IMAGEANALYZER00?label=last%20commit)](https://github.com/Radheshyam00/IMAGEANALYZER00)
[![Python](https://img.shields.io/badge/python-100%25-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-green)](LICENSE)

### Built with Modern Technologies

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

</div>

---

## Overview

**AIMSA (Advanced Image Metadata & Security Analyzer)** is a comprehensive Streamlit-based application designed for in-depth image analysis. Perfect for digital forensics professionals, cybersecurity experts, and privacy-conscious users who need to thoroughly examine image files for metadata, security threats, and hidden content.

## Key Features

### **Comprehensive Analysis**
- **File Information**: Complete file details including name, size, MIME type, MD5/SHA256 hashes, and magic bytes
- **EXIF & Metadata Extraction**: Camera information, timestamps, GPS coordinates, and exposure settings
- **Header Analysis**: File type validation and structural integrity checks

### **Metadata Management**
- **Metadata Editor**: Modify EXIF data in JPEG files
- **Metadata Removal**: Strip sensitive information for privacy protection
- **ICC Profile Analysis**: Extract color space and format specifications

### **Security Features**
- **VirusTotal Integration**: Real-time malware scanning (API key required)
- **Steganography Detection**: Identify hidden data, embedded archives, and entropy anomalies
- **JFIF Metadata Analysis**: Detailed format information extraction

### **Export & Reporting**
- **JSON Reports**: Export complete analysis results
- **Downloadable Formats**: Save cleaned images and reports

---



## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Radheshyam00/ImageAnalyzer00.git
   cd ImageAnalyzer00
   ```

2. **Set up virtual environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate (Linux/Mac)
   source venv/bin/activate
   
   # Activate (Windows)
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment** (Optional)
   ```bash
   # Create .env file for additional features
   echo "MONGO_URI=your_mongodb_uri" > .env
   echo "VIRUSTOTAL_API_KEY=your_api_key" >> .env
   ```

### Running the Application

```bash
streamlit run ImageAnalyzer.py
```

The application will open in your browser at `http://localhost:8501`

---

## Use Cases

### **Digital Forensics**
- Investigate image authenticity and origin
- Extract hidden metadata for evidence
- Analyze file structure for tampering

### **Privacy Protection**
- Remove GPS coordinates before sharing
- Strip camera information from photos
- Clean metadata for anonymous publishing

### **Cybersecurity**
- Detect steganographic payloads
- Scan for embedded malware
- Analyze suspicious image files

### **Content Management**
- Audit photo collections for sensitive data
- Batch process metadata removal
- Validate file integrity

---

## Configuration

### VirusTotal Integration
To enable malware scanning:
1. Sign up for a [VirusTotal API key](https://www.virustotal.com/gui/join-us)
2. Add your API key to the `.env` file or enter it in the application interface

### MongoDB Integration (Optional)
For enhanced file tracking with File Inspector Pro:
```env
MONGO_URI=mongodb://username:password@host:port/database
```

---

## Project Structure

```
ImageAnalyzer00/
├── ImageAnalyzer.py          # Main application file
├── requirements.txt          # Python dependencies
├── .env                     # Environment variables (create this)
├── LICENSE                  # BSD 3-Clause License
├── README.md               # This file
└── assets/                 # Additional resources
```

---

## Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the [BSD 3-Clause License](LICENSE) - see the LICENSE file for details.

---

## Support

If you encounter any issues or have questions:

- Open an issue on [GitHub Issues](https://github.com/Radheshyam00/ImageAnalyzer00/issues)
- Check existing issues for solutions
- Review the documentation

---

<div align="center">

**Made with dedication for the cybersecurity and digital forensics community**

Star this repository if you find it helpful!

</div>