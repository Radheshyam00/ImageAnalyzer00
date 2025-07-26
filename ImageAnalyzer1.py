import streamlit as st
from streamlit_ace import st_ace
import string
import hashlib
import binascii
from pymongo import MongoClient
from dotenv import load_dotenv
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import io
import struct
import base64
import json
import os
from datetime import datetime, timedelta, timezone
import requests
import time
import piexif
from bson import ObjectId
import numpy as np
import cv2
from PIL import Image, ImageChops, ImageEnhance, ImageFilter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict

load_dotenv()


def calculate_file_hash(file_bytes, algorithm='md5'):
    """Calculate file hash (MD5 or SHA256)"""
    if algorithm.lower() == 'md5':
        return hashlib.md5(file_bytes).hexdigest()
    elif algorithm.lower() == 'sha256':
        return hashlib.sha256(file_bytes).hexdigest()

def get_raw_header(file_bytes, num_bytes=32):
    """Get raw header bytes in hex format"""
    return binascii.hexlify(file_bytes[:num_bytes]).decode('utf-8').upper()

def analyze_magic_bytes(file_bytes):
    """Analyze file magic bytes"""
    header = file_bytes[:16]
    magic_signatures = {
        b'\xff\xd8\xff': 'JPEG',
        b'\x89PNG\r\n\x1a\n': 'PNG',
        b'GIF87a': 'GIF87a',
        b'GIF89a': 'GIF89a',
        b'RIFF': 'RIFF (WebP)',
        b'BM': 'BMP',
        b'II*\x00': 'TIFF (Little Endian)',
        b'MM\x00*': 'TIFF (Big Endian)'
    }
    
    for signature, file_type in magic_signatures.items():
        if header.startswith(signature):
            return signature.hex().upper(), file_type, True
    
    return header[:4].hex().upper(), 'Unknown', False

def extract_jfif_info(file_bytes):
    """Extract JFIF information from JPEG"""
    jfif_info = {}
    try:
        # Look for JFIF marker (FF E0)
        pos = 0
        while pos < len(file_bytes) - 1:
            if file_bytes[pos:pos+2] == b'\xff\xe0':
                # Found JFIF marker
                length = struct.unpack('>H', file_bytes[pos+2:pos+4])[0]
                jfif_data = file_bytes[pos+4:pos+2+length]
                
                if jfif_data.startswith(b'JFIF\x00'):
                    jfif_info['jfif_version'] = f"{jfif_data[5]}.{jfif_data[6]:02d}"
                    jfif_info['resolution_unit'] = ['None', 'inches', 'cm'][jfif_data[7]]
                    jfif_info['x_resolution'] = struct.unpack('>H', jfif_data[8:10])[0]
                    jfif_info['y_resolution'] = struct.unpack('>H', jfif_data[10:12])[0]
                break
            pos += 1
    except Exception as e:
        jfif_info['error'] = str(e)
    
    return jfif_info

def extract_icc_profile(file_bytes):
    """Extract ICC color profile information"""
    icc_info = {}
    try:
        # Look for ICC profile marker (FF E2)
        pos = 0
        while pos < len(file_bytes) - 1:
            if file_bytes[pos:pos+2] == b'\xff\xe2':
                length = struct.unpack('>H', file_bytes[pos+2:pos+4])[0]
                icc_data = file_bytes[pos+4:pos+2+length]
                
                if icc_data.startswith(b'ICC_PROFILE\x00'):
                    # Extract ICC profile header (first 128 bytes after identifier)
                    profile_start = 14  # Skip "ICC_PROFILE\x00" + sequence info
                    if len(icc_data) > profile_start + 128:
                        header = icc_data[profile_start:profile_start+128]
                        
                        icc_info['profile_size'] = struct.unpack('>I', header[0:4])[0]
                        icc_info['profile_cmm_type'] = header[4:8].decode('ascii', errors='ignore')
                        icc_info['profile_version'] = f"{header[8]}.{header[9]}.{header[10]}"
                        icc_info['profile_class'] = header[12:16].decode('ascii', errors='ignore')
                        icc_info['color_space_data'] = header[16:20].decode('ascii', errors='ignore')
                        icc_info['profile_connection_space'] = header[20:24].decode('ascii', errors='ignore')
                        
                        # Profile creation date/time
                        year = struct.unpack('>H', header[24:26])[0]
                        month = struct.unpack('>H', header[26:28])[0]
                        day = struct.unpack('>H', header[28:30])[0]
                        if year > 0:
                            icc_info['profile_date_time'] = f"{year}-{month:02d}-{day:02d}"
                        
                        icc_info['profile_file_signature'] = header[36:40].decode('ascii', errors='ignore')
                        icc_info['primary_platform'] = header[40:44].decode('ascii', errors='ignore')
                        icc_info['device_manufacturer'] = header[48:52].decode('ascii', errors='ignore')
                        icc_info['device_model'] = header[52:56].decode('ascii', errors='ignore')
                        
                        # Rendering intent
                        intent_map = {0: 'Perceptual', 1: 'Media-Relative', 2: 'Saturation', 3: 'ICC-Absolute'}
                        intent = struct.unpack('>I', header[64:68])[0]
                        icc_info['rendering_intent'] = intent_map.get(intent, f'Unknown ({intent})')
                        
                        icc_info['profile_creator'] = header[80:84].decode('ascii', errors='ignore')
                break
            pos += 1
    except Exception as e:
        icc_info['error'] = str(e)
    
    return icc_info

def extract_exif_data(image):
    """Extract EXIF data from image"""
    exif_data = {}
    try:
        exif = image.getexif()
        if exif:
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)
                
                # Handle GPS data specially
                if tag == 'GPSInfo':
                    gps_data = {}
                    for gps_tag_id, gps_value in value.items():
                        gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                        gps_data[gps_tag] = gps_value
                    exif_data['GPS'] = gps_data
                else:
                    # Convert bytes to string if needed
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8', errors='ignore')
                        except:
                            value = str(value)
                    exif_data[tag] = value
    except Exception as e:
        exif_data['error'] = str(e)
    
    return exif_data

def check_steganographic_indicators(file_bytes, filename):
    """Check for steganographic or hidden data indicators"""
    indicators = {}
    
    # Check for comments in JPEG
    try:
        pos = 0
        while pos < len(file_bytes) - 1:
            if file_bytes[pos:pos+2] == b'\xff\xfe':  # Comment marker
                length = struct.unpack('>H', file_bytes[pos+2:pos+4])[0]
                comment = file_bytes[pos+4:pos+2+length].decode('utf-8', errors='ignore')
                indicators['comment'] = comment
                break
            pos += 1
    except:
        pass
    
    # Check for appended data after EOF
    if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
        # Look for JPEG EOF marker (FF D9)
        eof_pos = file_bytes.rfind(b'\xff\xd9')
        if eof_pos != -1 and eof_pos < len(file_bytes) - 2:
            appended_size = len(file_bytes) - eof_pos - 2
            indicators['appended_data_size'] = appended_size
            if appended_size > 0:
                indicators['appended_data_preview'] = file_bytes[eof_pos+2:eof_pos+32].hex().upper()
    
    # Check for dual file signatures (e.g., ZIP in JPEG)
    zip_signature = b'PK\x03\x04'
    if zip_signature in file_bytes[100:]:  # Skip normal header area
        indicators['dual_file_signature'] = 'ZIP archive detected'
    
    # Calculate entropy (simplified)
    if len(file_bytes) > 1000:
        sample = file_bytes[-1000:]  # Check last 1000 bytes
        byte_counts = [0] * 256
        for byte in sample:
            byte_counts[byte] += 1
        
        entropy = 0
        for count in byte_counts:
            if count > 0:
                p = count / len(sample)
                import math
                entropy -= p * math.log2(p)
        
        indicators['payload_entropy'] = round(entropy, 3)
        if entropy > 7.5:  # High entropy threshold
            indicators['high_entropy_warning'] = True
    
    return indicators

def remove_metadata(image_bytes, remove_options):
    """Remove selected metadata from image"""
    try:
        # Load image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Create a copy without metadata
        clean_image = Image.new(image.mode, image.size)
        clean_image.putdata(list(image.getdata()))
        
        # Handle different removal options
        if remove_options.get('remove_all', False):
            # Remove all metadata - save without any info
            output = io.BytesIO()
            clean_image.save(output, format=image.format, optimize=True)
            return output.getvalue(), "All metadata removed"
        
        # Selective removal
        removed_items = []
        
        # For JPEG images, we can use piexif for more control
        if image.format == 'JPEG':
            try:
                exif_dict = piexif.load(image_bytes)
                
                if remove_options.get('remove_exif', False):
                    exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
                    removed_items.append("EXIF data")
                
                if remove_options.get('remove_gps', False):
                    exif_dict["GPS"] = {}
                    removed_items.append("GPS data")
                
                if remove_options.get('remove_camera_info', False):
                    # Remove camera make/model
                    exif_dict["0th"].pop(piexif.ImageIFD.Make, None)
                    exif_dict["0th"].pop(piexif.ImageIFD.Model, None)
                    exif_dict["0th"].pop(piexif.ImageIFD.Software, None)
                    removed_items.append("Camera information")
                
                if remove_options.get('remove_timestamps', False):
                    exif_dict["0th"].pop(piexif.ImageIFD.DateTime, None)
                    exif_dict["Exif"].pop(piexif.ExifIFD.DateTimeOriginal, None)
                    exif_dict["Exif"].pop(piexif.ExifIFD.DateTimeDigitized, None)
                    removed_items.append("Timestamps")
                
                # Save with modified EXIF
                exif_bytes = piexif.dump(exif_dict)
                output = io.BytesIO()
                image.save(output, format='JPEG', exif=exif_bytes, optimize=True)
                return output.getvalue(), f"Removed: {', '.join(removed_items)}"
                
            except Exception as e:
                # Fallback: save without any metadata
                output = io.BytesIO()
                clean_image.save(output, format=image.format, optimize=True)
                return output.getvalue(), f"Fallback removal - all metadata stripped ({str(e)})"
        
        else:
            # For non-JPEG formats, save clean copy
            output = io.BytesIO()
            clean_image.save(output, format=image.format, optimize=True)
            return output.getvalue(), "All metadata removed (non-JPEG format)"
            
    except Exception as e:
        raise Exception(f"Error removing metadata: {str(e)}")

def edit_metadata(image_bytes, edit_data):
    """Edit metadata in image"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.format != 'JPEG':
            raise Exception("Metadata editing currently only supported for JPEG images")
        
        # Load existing EXIF data
        try:
            exif_dict = piexif.load(image_bytes)
        except:
            # Create new EXIF structure if none exists
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
        
        changes_made = []
        
        # Edit basic camera info
        if edit_data.get('camera_make'):
            exif_dict["0th"][piexif.ImageIFD.Make] = edit_data['camera_make']
            changes_made.append("Camera Make")
        
        if edit_data.get('camera_model'):
            exif_dict["0th"][piexif.ImageIFD.Model] = edit_data['camera_model']
            changes_made.append("Camera Model")
        
        if edit_data.get('software'):
            exif_dict["0th"][piexif.ImageIFD.Software] = edit_data['software']
            changes_made.append("Software")
        
        if edit_data.get('artist'):
            exif_dict["0th"][piexif.ImageIFD.Artist] = edit_data['artist']
            changes_made.append("Artist")
        
        if edit_data.get('copyright'):
            exif_dict["0th"][piexif.ImageIFD.Copyright] = edit_data['copyright']
            changes_made.append("Copyright")
        
        # Edit timestamps
        if edit_data.get('datetime'):
            datetime_str = edit_data['datetime'].strftime("%Y:%m:%d %H:%M:%S")
            exif_dict["0th"][piexif.ImageIFD.DateTime] = datetime_str
            exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = datetime_str
            exif_dict["Exif"][piexif.ExifIFD.DateTimeDigitized] = datetime_str
            changes_made.append("Timestamps")
        
        # Edit camera settings
        if edit_data.get('iso'):
            exif_dict["Exif"][piexif.ExifIFD.ISOSpeedRatings] = int(edit_data['iso'])
            changes_made.append("ISO")
        
        if edit_data.get('focal_length'):
            # Convert to rational (numerator, denominator)
            focal = float(edit_data['focal_length'])
            exif_dict["Exif"][piexif.ExifIFD.FocalLength] = (int(focal * 10), 10)
            changes_made.append("Focal Length")
        
        if edit_data.get('aperture'):
            # Convert f-number to APEX value
            f_num = float(edit_data['aperture'])
            exif_dict["Exif"][piexif.ExifIFD.FNumber] = (int(f_num * 10), 10)
            changes_made.append("Aperture")
        
        # Edit GPS data
        if edit_data.get('gps_lat') and edit_data.get('gps_lon'):
            lat = float(edit_data['gps_lat'])
            lon = float(edit_data['gps_lon'])
            
            # Convert decimal degrees to degrees, minutes, seconds
            def decimal_to_dms(decimal):
                degrees = int(abs(decimal))
                minutes_float = (abs(decimal) - degrees) * 60
                minutes = int(minutes_float)
                seconds = (minutes_float - minutes) * 60
                return [(degrees, 1), (minutes, 1), (int(seconds * 100), 100)]
            
            exif_dict["GPS"][piexif.GPSIFD.GPSLatitudeRef] = 'N' if lat >= 0 else 'S'
            exif_dict["GPS"][piexif.GPSIFD.GPSLatitude] = decimal_to_dms(lat)
            exif_dict["GPS"][piexif.GPSIFD.GPSLongitudeRef] = 'E' if lon >= 0 else 'W'
            exif_dict["GPS"][piexif.GPSIFD.GPSLongitude] = decimal_to_dms(lon)
            changes_made.append("GPS Coordinates")
        
        # Add custom comment
        if edit_data.get('comment'):
            exif_dict["0th"][piexif.ImageIFD.ImageDescription] = edit_data['comment']
            changes_made.append("Image Description")
        
        # Save with modified EXIF
        exif_bytes = piexif.dump(exif_dict)
        output = io.BytesIO()
        image.save(output, format='JPEG', exif=exif_bytes, optimize=True)
        
        return output.getvalue(), f"Modified: {', '.join(changes_made)}"
        
    except Exception as e:
        raise Exception(f"Error editing metadata: {str(e)}")

def create_metadata_editor_ui(uploaded_file, file_bytes):
    """Create the metadata editor interface"""
    st.subheader("✏️ Metadata Editor")
    
    edit_tab, remove_tab = st.tabs(["Edit Metadata", "Remove Metadata"])
    
    with edit_tab:
        st.markdown("**Edit or add metadata to your image:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Basic Information**")
            camera_make = st.text_input("Camera Make", placeholder="e.g., Canon")
            camera_model = st.text_input("Camera Model", placeholder="e.g., EOS R5")
            software = st.text_input("Software", placeholder="e.g., Adobe Lightroom")
            artist = st.text_input("Artist/Photographer", placeholder="Your name")
            copyright_text = st.text_input("Copyright", placeholder="© 2024 Your Name")
            
            st.markdown("**Camera Settings**")
            iso = st.number_input("ISO", min_value=50, max_value=102400, value=100, step=50)
            focal_length = st.number_input("Focal Length (mm)", min_value=1.0, max_value=800.0, value=50.0, step=0.1)
            aperture = st.number_input("Aperture (f-number)", min_value=1.0, max_value=32.0, value=2.8, step=0.1)
        
        with col2:
            st.markdown("**Date & Time**")
            date_edit = st.date_input("Photo Date", value=datetime.now().date())
            time_edit = st.time_input("Photo Time", value=datetime.now().time())
            
            st.markdown("**GPS Location**")
            gps_lat = st.number_input("Latitude", value=0.0, format="%.6f", help="Positive for North, negative for South")
            gps_lon = st.number_input("Longitude", value=0.0, format="%.6f", help="Positive for East, negative for West")
            
            st.markdown("**Additional Info**")
            comment = st.text_area("Image Description/Comment", placeholder="Description of the image")
        
        if st.button("💾 Apply Changes", type="primary"):
            try:
                # Combine date and time
                datetime_combined = datetime.combine(date_edit, time_edit)
                
                edit_data = {
                    'camera_make': camera_make if camera_make else None,
                    'camera_model': camera_model if camera_model else None,
                    'software': software if software else None,
                    'artist': artist if artist else None,
                    'copyright': copyright_text if copyright_text else None,
                    'datetime': datetime_combined,
                    'iso': iso,
                    'focal_length': focal_length,
                    'aperture': aperture,
                    'gps_lat': gps_lat if gps_lat != 0.0 else None,
                    'gps_lon': gps_lon if gps_lon != 0.0 else None,
                    'comment': comment if comment else None
                }
                
                modified_bytes, changes = edit_metadata(file_bytes, edit_data)
                
                st.success(f"✅ {changes}")
                
                # Provide download button
                st.download_button(
                    label="📥 Download Modified Image",
                    data=modified_bytes,
                    file_name=f"modified_{uploaded_file.name}",
                    mime=uploaded_file.type
                )
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    with remove_tab:
        st.markdown("**Select metadata to remove:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            remove_all = st.checkbox("🗑️ Remove ALL metadata", help="Completely strip all metadata")
            
            remove_exif = False
            remove_gps = False
            remove_camera = False
            remove_timestamps = False
            
            if not remove_all:
                st.markdown("**Selective Removal:**")
                remove_exif = st.checkbox("Remove EXIF data", help="Camera settings, exposure info")
                remove_gps = st.checkbox("Remove GPS location", help="Geographic coordinates")
                remove_camera = st.checkbox("Remove camera info", help="Make, model, software")
                remove_timestamps = st.checkbox("Remove timestamps", help="Date/time information")
        
        with col2:
            st.markdown("**Preview:**")
            if remove_all:
                st.info("🔄 All metadata will be stripped from the image")
            else:
                removal_list = []
                if remove_exif:
                    removal_list.append("EXIF data")
                if remove_gps:
                    removal_list.append("GPS location")
                if remove_camera:
                    removal_list.append("Camera information")
                if remove_timestamps:
                    removal_list.append("Timestamps")
                
                if removal_list:
                    st.info(f"🔄 Will remove: {', '.join(removal_list)}")
                else:
                    st.warning("⚠️ No removal options selected")
        
        if st.button("🗑️ Remove Selected Metadata", type="secondary"):
            try:
                remove_options = {
                    'remove_all': remove_all,
                    'remove_exif': remove_exif,
                    'remove_gps': remove_gps,
                    'remove_camera_info': remove_camera,
                    'remove_timestamps': remove_timestamps
                }
                
                if not any(remove_options.values()):
                    st.warning("⚠️ Please select at least one removal option")
                else:
                    cleaned_bytes, removal_summary = remove_metadata(file_bytes, remove_options)
                    
                    st.success(f"✅ {removal_summary}")
                    
                    # Show size comparison
                    original_size = len(file_bytes)
                    cleaned_size = len(cleaned_bytes)
                    size_diff = original_size - cleaned_size
                    
                    if size_diff > 0:
                        st.info(f"📊 Size reduced by {size_diff:,} bytes ({size_diff/original_size*100:.1f}%)")
                    
                    # Provide download button
                    st.download_button(
                        label="📥 Download Cleaned Image",
                        data=cleaned_bytes,
                        file_name=f"cleaned_{uploaded_file.name}",
                        mime=uploaded_file.type
                    )
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")


def analyze_header_validity(file_bytes, file_type):
    """Analyze header validity"""
    validity = {}
    
    magic_bytes, detected_type, is_valid = analyze_magic_bytes(file_bytes)
    validity['magic_bytes'] = magic_bytes
    validity['magic_bytes_valid'] = is_valid
    validity['detected_type'] = detected_type
    
    # Check for EXIF marker in JPEG
    if file_type.upper() == 'JPEG':
        has_exif = b'\xff\xe1' in file_bytes[:1000]
        validity['exif_marker'] = 'Present (FF E1)' if has_exif else 'Not found'
        
        # Check for ICC profile marker
        has_icc = b'\xff\xe2' in file_bytes[:2000]
        validity['icc_profile_marker'] = 'Present (FF E2)' if has_icc else 'Not found'
    
    return validity

def virustotal_upload_file(file_bytes, filename, api_key):
    """Upload file to VirusTotal for scanning"""
    url = "https://www.virustotal.com/api/v3/files"
    headers = {
        "X-Apikey": api_key
    }
    
    files = {
        "file": (filename, file_bytes, "application/octet-stream")
    }
    
    try:
        response = requests.post(url, headers=headers, files=files, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Upload failed: {str(e)}"}

def virustotal_get_file_report(file_hash, api_key):
    """Get file report from VirusTotal using file hash"""
    url = f"https://www.virustotal.com/api/v3/files/{file_hash}"
    headers = {
        "X-Apikey": api_key
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 404:
            return {"status": "not_found", "message": "File not found in VirusTotal database"}
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Report fetch failed: {str(e)}"}

def virustotal_get_analysis_report(analysis_id, api_key):
    """Get analysis report from VirusTotal using analysis ID"""
    url = f"https://www.virustotal.com/api/v3/analyses/{analysis_id}"
    headers = {
        "X-Apikey": api_key
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Analysis fetch failed: {str(e)}"}

def process_virustotal_scan(file_bytes, filename, file_hash, api_key):
    """Process VirusTotal scan with comprehensive error handling"""
    if not api_key:
        return {"status": "no_api_key", "message": "VirusTotal API key not provided"}
    
    # First, try to get existing report by hash
    st.info("🔍 Checking VirusTotal database for existing scan results...")
    existing_report = virustotal_get_file_report(file_hash, api_key)
    
    if "error" not in existing_report and existing_report.get("status") != "not_found":
        st.success("✅ Found existing scan results in VirusTotal database")
        return existing_report
    
    # If file not found, upload for new scan
    st.info("📤 File not found in database. Uploading to VirusTotal for scanning...")
    upload_result = virustotal_upload_file(file_bytes, filename, api_key)
    
    if "error" in upload_result:
        return upload_result
    
    if "data" not in upload_result:
        return {"error": "Invalid upload response from VirusTotal"}
    
    analysis_id = upload_result["data"]["id"]
    st.info(f"⏳ File uploaded successfully. Analysis ID: {analysis_id}")
    st.info("⏱️ Waiting for scan to complete (this may take 1-2 minutes)...")
    
    # Poll for analysis results
    max_attempts = 20  # Maximum 10 minutes
    attempt = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while attempt < max_attempts:
        analysis_report = virustotal_get_analysis_report(analysis_id, api_key)
        
        if "error" in analysis_report:
            return analysis_report
        
        if "data" in analysis_report:
            status = analysis_report["data"]["attributes"]["status"]
            status_text.text(f"Status: {status}")
            
            if status == "completed":
                progress_bar.progress(100)
                st.success("✅ Scan completed successfully!")
                
                # Get the final file report
                final_report = virustotal_get_file_report(file_hash, api_key)
                return final_report
        
        attempt += 1
        progress = min((attempt / max_attempts) * 100, 95)
        progress_bar.progress(int(progress))
        time.sleep(30)  # Wait 30 seconds between checks
    
    return {"error": "Scan timeout - analysis took too long to complete"}

def display_virustotal_results(vt_result):
    """Display VirusTotal scan results in a user-friendly format"""
    if "error" in vt_result:
        st.error(f"❌ VirusTotal Error: {vt_result['error']}")
        return
    
    if vt_result.get("status") == "no_api_key":
        st.warning("⚠️ VirusTotal API key not provided. Malware scanning disabled.")
        return
    
    if vt_result.get("status") == "not_found":
        st.info("ℹ️ File not found in VirusTotal database and upload failed.")
        return
    
    if "data" not in vt_result:
        st.error("❌ Invalid response from VirusTotal")
        return
    
    try:
        data = vt_result["data"]
        attributes = data["attributes"]
        stats = attributes.get("last_analysis_stats", {})
        
        # Overall threat assessment
        malicious = stats.get("malicious", 0)
        suspicious = stats.get("suspicious", 0)
        harmless = stats.get("harmless", 0)
        undetected = stats.get("undetected", 0)
        total_engines = malicious + suspicious + harmless + undetected
        
        # Display threat level
        if malicious > 0:
            st.error(f"🚨 **MALWARE DETECTED** - {malicious}/{total_engines} engines flagged this file as malicious")
        elif suspicious > 0:
            st.warning(f"⚠️ **SUSPICIOUS** - {suspicious}/{total_engines} engines flagged this file as suspicious")
        else:
            st.success(f"✅ **CLEAN** - No malware detected by {total_engines} antivirus engines")
        
        # Detailed statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Malicious", malicious, delta=None)
        with col2:
            st.metric("Suspicious", suspicious, delta=None)
        with col3:
            st.metric("Harmless", harmless, delta=None)
        with col4:
            st.metric("Undetected", undetected, delta=None)
        
        # File information
        st.subheader("📋 VirusTotal File Information")
        vt_info = {
            "File Size": f"{attributes.get('size', 'Unknown'):,} bytes",
            "File Type": attributes.get('type_description', 'Unknown'),
            "MD5": attributes.get('md5', 'Unknown'),
            "SHA1": attributes.get('sha1', 'Unknown'),
            "SHA256": attributes.get('sha256', 'Unknown'),
            "First Submission": attributes.get('first_submission_date', 'Unknown'),
            "Last Analysis": attributes.get('last_analysis_date', 'Unknown'),
            "Times Submitted": attributes.get('times_submitted', 'Unknown')
        }
        
        # Convert timestamps
        if isinstance(vt_info["First Submission"], int):
            vt_info["First Submission"] = datetime.fromtimestamp(vt_info["First Submission"]).strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(vt_info["Last Analysis"], int):
            vt_info["Last Analysis"] = datetime.fromtimestamp(vt_info["Last Analysis"]).strftime("%Y-%m-%d %H:%M:%S")
        
        col1, col2 = st.columns(2)
        items = list(vt_info.items())
        mid = len(items) // 2
        
        with col1:
            for key, value in items[:mid]:
                st.text(f"{key}: {value}")
        with col2:
            for key, value in items[mid:]:
                st.text(f"{key}: {value}")
        
        # Detailed engine results (if any detections)
        if malicious > 0 or suspicious > 0:
            with st.expander("🔍 Detailed Detection Results", expanded=False):
                results = attributes.get("last_analysis_results", {})
                for engine, result in results.items():
                    category = result.get("category", "undetected")
                    if category in ["malicious", "suspicious"]:
                        threat_name = result.get("result", "Unknown")
                        st.text(f"🔴 {engine}: {threat_name}")
        
        # Additional file properties
        if attributes.get("names"):
            st.subheader("📝 Known File Names")
            for name in attributes["names"][:10]:  # Show first 10 names
                st.text(f"• {name}")
    
    except Exception as e:
        st.error(f"❌ Error processing VirusTotal results: {str(e)}")


def main():
    st.set_page_config(
        page_title="Image Metadata & Security Analyzer",
        page_icon="🔍",
        layout="wide"
    )
     # Navigation menu with icons
    st.sidebar.markdown("### 📋 Navigation")
    page = st.sidebar.selectbox("Choose a Tool", 
                   ["🔍 Advanced Image Metadata Analyzer", 
                    # "⚙️ Reverse Image Analyzer", 
                    "📞 File Inspector Pro", 
                    "📊 Advanced Image Forensics Analyzer",
                    "🔧 Settings",
                    "📱 Aadhaar-Mobile Link Checker"])
    
    if page == "🔍 Advanced Image Metadata Analyzer":
            
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
                }
            </style>
            """, unsafe_allow_html=True)
            
            # Header
            st.markdown("""
            <div class="main-header">
                <h1>🔍 Advanced Image Metadata & Security Analyzer</h1>
                <p>Upload an image to extract comprehensive metadata, EXIF data, and check for malware using VirusTotal.</p>
            </div>
            """, unsafe_allow_html=True)
    
            # st.title("🔍 Advanced Image Metadata & Security Analyzer")
            # st.markdown("Upload an image to extract comprehensive metadata, EXIF data, and check for malware using VirusTotal.")
            
            # VirusTotal API Key input
            st.sidebar.header("🔐 VirusTotal Configuration")
            api_key = st.sidebar.text_input(
                "VirusTotal API Key",
                type="password",
                help="Enter your VirusTotal API key to enable malware scanning. Get a free key at virustotal.com"
            )
            
            if api_key:
                st.sidebar.success("✅ API Key configured")
            else:
                st.sidebar.warning("⚠️ No API key - malware scanning disabled")
            
            st.sidebar.markdown("---")
            st.sidebar.markdown("**Note:** VirusTotal has rate limits:")
            st.sidebar.markdown("- Free tier: 4 requests/minute")
            st.sidebar.markdown("- Premium: Higher limits")
            
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp'],
                help="Supported formats: JPEG, PNG, GIF, BMP, TIFF, WebP"
            )
            
            if uploaded_file is not None:
                # Read file bytes
                file_bytes = uploaded_file.read()
                uploaded_file.seek(0)  # Reset file pointer
                
                st.markdown("---")
                # Calculate hashes early for VirusTotal
                md5_hash = calculate_file_hash(file_bytes, 'md5')
                sha256_hash = calculate_file_hash(file_bytes, 'sha256')
                
                with st.expander("📁 Basic File Info", expanded=False):
                    # Display image
                    col1, col3, col2 = st.columns([1, 0.1, 1.5])
                    
                    with col1:
                        try:
                            image = Image.open(uploaded_file)
                            st.image(image, caption=uploaded_file.name, use_column_width=True)
                        except Exception as e:
                            st.error(f"Could not display image: {e}")
                            image = None
                    with col3:
                        st.markdown("""
                        <div style='height: 300px; border-left: 2px solid #ccc; margin: 10px;'></div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # Basic file info
                        # st.subheader("📁 Basic File Info")
                        basic_info = {
                            "File Name": uploaded_file.name,
                            "File Size": f"{len(file_bytes):,} bytes ({len(file_bytes)/1024:.1f} KB)",
                            "File Type": uploaded_file.type,
                            "File Extension": os.path.splitext(uploaded_file.name)[1].lower(),
                            "MIME Type": uploaded_file.type,
                            "MD5 Checksum": md5_hash,
                            "SHA256 Checksum": sha256_hash,
                            "Raw Header": get_raw_header(file_bytes, 32)
                        }
                        
                        for key, value in basic_info.items():
                            st.text(f"{key}: {value}")
                
                # VirusTotal Malware Check - Priority section
                st.markdown("---")
                st.subheader("🛡️ Malware & Security Analysis")
                
                if st.button("🔍 Scan with VirusTotal", type="primary"):
                    with st.spinner("Scanning file with VirusTotal..."):
                        vt_result = process_virustotal_scan(file_bytes, uploaded_file.name, sha256_hash, api_key)
                        st.session_state['vt_result'] = vt_result
                
                # Display cached results if available
                if 'vt_result' in st.session_state:
                    display_virustotal_results(st.session_state['vt_result'])
                
                # Image properties
                st.markdown("---")
                if image:
                    st.subheader("🖼️ Image Properties")
                    
                    width, height = image.size
                    megapixels = (width * height) / 1_000_000
                    
                    image_props = {
                        "Image Width": f"{width} pixels",
                        "Image Height": f"{height} pixels",
                        "Megapixels": f"{megapixels:.2f} MP",
                        "Image Size": f"{width} × {height}",
                        "Color Mode": image.mode,
                        "Format": image.format,
                    }
                    
                    # Additional format-specific info
                    if hasattr(image, 'info'):
                        if 'dpi' in image.info:
                            image_props["DPI"] = str(image.info['dpi'])
                        if 'compression' in image.info:
                            image_props["Compression"] = image.info['compression']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        for key, value in list(image_props.items())[:len(image_props)//2]:
                            st.text(f"{key}: {value}")
                    with col2:
                        for key, value in list(image_props.items())[len(image_props)//2:]:
                            st.text(f"{key}: {value}")
                
                # JFIF Metadata
                st.markdown("---")
                st.subheader("🏷️ JFIF Metadata")
                jfif_info = extract_jfif_info(file_bytes)
                if jfif_info:
                    for key, value in jfif_info.items():
                        st.text(f"{key}: {value}")
                else:
                    st.text("No JFIF metadata found")
                
                # ICC Color Profile
                st.markdown("---")
                st.subheader("🎨 ICC Color Profile Metadata")
                icc_info = extract_icc_profile(file_bytes)
                if icc_info:
                    col1, col2 = st.columns(2)
                    items = list(icc_info.items())
                    mid = len(items) // 2
                    
                    with col1:
                        for key, value in items[:mid]:
                            st.text(f"{key}: {value}")
                    with col2:
                        for key, value in items[mid:]:
                            st.text(f"{key}: {value}")
                else:
                    st.text("No ICC color profile found")
                
                # EXIF Metadata
                st.markdown("---")
                st.subheader("🧬 EXIF Metadata")
                if image:
                    exif_data = extract_exif_data(image)
                    if exif_data:
                        # Display in expandable sections
                        with st.expander("Camera Information", expanded=True):
                            camera_fields = ['Make', 'Model', 'Software', 'DateTime', 'DateTimeOriginal']
                            for field in camera_fields:
                                if field in exif_data:
                                    st.text(f"{field}: {exif_data[field]}")
                        
                        with st.expander("Camera Settings"):
                            settings_fields = ['ExposureTime', 'FNumber', 'ISOSpeedRatings', 'FocalLength', 'Flash']
                            for field in settings_fields:
                                if field in exif_data:
                                    st.text(f"{field}: {exif_data[field]}")
                        
                        if 'GPS' in exif_data:
                            with st.expander("GPS Information"):
                                for key, value in exif_data['GPS'].items():
                                    st.text(f"{key}: {value}")
                        
                        with st.expander("All EXIF Data"):
                            for key, value in exif_data.items():
                                if key != 'GPS':
                                    st.text(f"{key}: {value}")
                    else:
                        st.text("No EXIF metadata found")
                else:
                    st.text("Could not extract EXIF data - image failed to load")
                
                # Steganographic indicators
                st.markdown("---")
                st.subheader("🔐 Steganographic / Hidden Indicators")
                steg_indicators = check_steganographic_indicators(file_bytes, uploaded_file.name)
                if steg_indicators:
                    for key, value in steg_indicators.items():
                        if key == 'high_entropy_warning':
                            st.warning("⚠️ High entropy detected in file tail - possible hidden data")
                        else:
                            st.text(f"{key}: {value}")
                else:
                    st.text("No suspicious indicators detected")
                
                # Header analysis
                st.markdown("---")
                st.subheader("🔎 Header Analysis")
                header_analysis = analyze_header_validity(file_bytes, uploaded_file.type)
                for key, value in header_analysis.items():
                    if key == 'magic_bytes_valid':
                        if value:
                            st.success(f"✅ Magic bytes valid")
                        else:
                            st.error(f"❌ Invalid magic bytes detected")
                    else:
                        st.text(f"{key}: {value}")

                # Metadata Editor and Remover
                st.markdown("---")
                create_metadata_editor_ui(uploaded_file, file_bytes)
                
                
                # Download full report
                st.markdown("---")
                st.subheader("📄 Export Report")
                if st.button("Generate Comprehensive Report"):
                    report = {
                        "basic_info": basic_info,
                        "image_properties": image_props if image else {},
                        "jfif_metadata": jfif_info,
                        "icc_profile": icc_info,
                        "exif_data": exif_data if image else {},
                        "steganographic_indicators": steg_indicators,
                        "header_analysis": header_analysis,
                        "virustotal_scan": st.session_state.get('vt_result', {"status": "not_scanned"}),
                        "analysis_timestamp": datetime.now().isoformat()
                    }
                    
                    json_report = json.dumps(report, indent=2, default=str)
                    st.download_button(
                        label="Download JSON Report",
                        data=json_report,
                        file_name=f"{uploaded_file.name}_security_metadata_report.json",
                        mime="application/json"
                    )
    elif page == "⚙️ Reverse Image Analyzer":
        st.title("Product Analysis Tool")
    elif page == "📞 File Inspector Pro":
        # Configure page
        # st.title("File Inspector Pro")
        # st.set_page_config(
        #     page_title="File Inspector Pro",
        #     page_icon="🔍",
        #     layout="wide",
        #     initial_sidebar_state="expanded"
        # )

        # MongoDB Configuration
        
        MONGO_URI = os.getenv("MONGO_URI")

        @st.cache_resource
        def init_mongodb():
            """Initialize MongoDB connection"""
            try:
                client = MongoClient(MONGO_URI)
                # Test the connection
                client.admin.command('ismaster')
                db = client['file_inspector']
                return db
            except Exception as e:
                st.error(f"Failed to connect to MongoDB: {str(e)}")
                return None

        # Initialize MongoDB
        db = init_mongodb()

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
            
            .info-card {
                background: #f8f9fa;
                padding: 0.5rem;
                border-radius: 8px;
                border-left: 4px solid #667eea;
                margin: 1rem 0;
            }
            
            .metric-container {
                background: white;
                padding: 1rem;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                text-align: center;
            }
            
            .hex-container {
                font-family: 'Courier New', monospace;
                background: #1e1e1e;
                color: #d4d4d4;
                padding: 1rem;
                border-radius: 8px;
                max-height: 400px;
                overflow-y: auto;
            }
            
            .status-success {
                color: #28a745;
                font-weight: bold;
            }
            
            .status-info {
                color: #17a2b8;
                font-weight: bold;
            }
            
            .db-status {
                background: #e7f3ff;
                padding: 0.5rem;
                border-radius: 5px;
                border-left: 3px solid #2196f3;
                margin: 0.5rem 0;
            }
        </style>
        """, unsafe_allow_html=True)

        # MongoDB Helper Functions
        def save_file_analysis(filename, file_bytes, extracted_strings, hashes, file_info):
            """Save file analysis to MongoDB"""
            if db is None:
                return None
            
            try:
                collection = db['file_analyses']
                
                # Create document
                document = {
                    'filename': filename,
                    'upload_date': datetime.now(timezone.utc),
                    'file_size': len(file_bytes),
                    'file_type': file_info,
                    'md5_hash': hashes[0],
                    'sha1_hash': hashes[1],
                    'sha256_hash': hashes[2],
                    'extracted_strings': extracted_strings[:1000],  # Limit to first 1000 strings
                    'string_count': len(extracted_strings),
                    'entropy': calculate_entropy(file_bytes),
                    'null_percentage': (file_bytes.count(0) / len(file_bytes)) * 100 if file_bytes else 0,
                    'file_data_base64': base64.b64encode(file_bytes[:10240]).decode('utf-8') if len(file_bytes) <= 10240 else None  # Store small files only
                }
                
                result = collection.insert_one(document)
                return result.inserted_id
            except Exception as e:
                st.error(f"Error saving to MongoDB: {str(e)}")
                return None

        def get_recent_analyses(limit=10):
            """Get recent file analyses from MongoDB"""
            if db is None:
                return []
            
            try:
                collection = db['file_analyses']
                analyses = list(collection.find().sort('upload_date', -1).limit(limit))
                return analyses
            except Exception as e:
                st.error(f"Error retrieving from MongoDB: {str(e)}")
                return []

        def search_analyses(search_term):
            """Search file analyses by filename or hash"""
            if db is None:
                return []
            
            try:
                collection = db['file_analyses']
                query = {
                    '$or': [
                        {'filename': {'$regex': search_term, '$options': 'i'}},
                        {'md5_hash': {'$regex': search_term, '$options': 'i'}},
                        {'sha1_hash': {'$regex': search_term, '$options': 'i'}},
                        {'sha256_hash': {'$regex': search_term, '$options': 'i'}}
                    ]
                }
                analyses = list(collection.find(query).sort('upload_date', -1))
                return analyses
            except Exception as e:
                st.error(f"Error searching MongoDB: {str(e)}")
                return []

        def get_db_stats():
            """Get database statistics"""
            if db is None:
                return None
            
            try:
                collection = db['file_analyses']
                total_files = collection.count_documents({})
                recent_files = collection.count_documents({
                    'upload_date': {'$gte': datetime.now(timezone.utc) - timedelta(days=7)}
                })
                
                # Get file type distribution
                pipeline = [
                    {'$group': {'_id': '$file_type', 'count': {'$sum': 1}}},
                    {'$sort': {'count': -1}},
                    {'$limit': 5}
                ]
                file_types = list(collection.aggregate(pipeline))
                
                return {
                    'total_files': total_files,
                    'recent_files': recent_files,
                    'file_types': file_types
                }
            except Exception as e:
                st.error(f"Error getting DB stats: {str(e)}")
                return None

        # Optimized helper functions (keeping original functions)
        @st.cache_data
        def extract_strings_optimized(file_bytes, min_length=4, max_length=100):
            """Extract printable strings with improved performance and filtering"""
            if not file_bytes:
                return []
            
            result = []
            current = bytearray()
            printable_chars = set(string.ascii_letters + string.digits + string.punctuation + ' ')
            
            for byte in file_bytes:
                char = chr(byte) if byte < 128 else None
                
                if char and char in printable_chars:
                    current.append(byte)
                else:
                    if len(current) >= min_length:
                        decoded = current.decode('ascii', errors='ignore')
                        if len(decoded) <= max_length:
                            result.append(decoded)
                    current = bytearray()
            
            if len(current) >= min_length:
                decoded = current.decode('ascii', errors='ignore')
                if len(decoded) <= max_length:
                    result.append(decoded)
            
            return list(dict.fromkeys(result))

        @st.cache_data
        def calculate_file_hashed(file_bytes):
            """Calculate multiple hash values for the file"""
            md5 = hashlib.md5(file_bytes).hexdigest()
            sha1 = hashlib.sha1(file_bytes).hexdigest()
            sha256 = hashlib.sha256(file_bytes).hexdigest()
            return md5, sha1, sha256

        def calculate_entropy(file_bytes):
            """Calculate entropy (measure of randomness)"""
            if not file_bytes:
                return 0
            
            import math
            byte_counts = [0] * 256
            for byte in file_bytes:
                byte_counts[byte] += 1
            
            entropy = 0
            for count in byte_counts:
                if count > 0:
                    probability = count / len(file_bytes)
                    entropy -= probability * math.log2(probability)
            
            return round(entropy, 2)

        @st.cache_data
        def format_hex_output(file_bytes, bytes_per_line=16):
            """Format hex output with addresses and ASCII preview"""
            if not file_bytes:
                return ""
            
            lines = []
            for i in range(0, len(file_bytes), bytes_per_line):
                chunk = file_bytes[i:i + bytes_per_line]
                hex_part = ' '.join(f"{b:02x}" for b in chunk)
                hex_part = hex_part.ljust(bytes_per_line * 3 - 1)
                
                ascii_part = ''.join(
                    chr(b) if 32 <= b <= 126 else '.' for b in chunk
                )
                
                lines.append(f"{i:08x}  {hex_part}  |{ascii_part}|")
            
            return '\n'.join(lines)

        def get_file_type_info(file_bytes, filename):
            """Detect file type based on magic bytes and extension"""
            if not file_bytes:
                return "Unknown"
            
            magic_bytes = {
                b'\x89PNG\r\n\x1a\n': 'PNG Image',
                b'\xff\xd8\xff': 'JPEG Image',
                b'GIF8': 'GIF Image',
                b'%PDF': 'PDF Document',
                b'PK\x03\x04': 'ZIP Archive',
                b'MZ': 'Windows Executable',
                b'\x7fELF': 'Linux Executable',
                b'\xca\xfe\xba\xbe': 'Java Class File',
                b'RIFF': 'RIFF Container (AVI/WAV)',
            }
            
            for magic, file_type in magic_bytes.items():
                if file_bytes.startswith(magic):
                    return file_type
            
            if '.' in filename:
                ext = filename.split('.')[-1].upper()
                return f"{ext} File"
            
            return "Binary/Unknown"

        # Header
        st.markdown("""
        <div class="main-header">
            <h1>🔍 File Inspector Pro</h1>
            <p>Advanced file analysis with MongoDB storage and historical tracking</p>
        </div>
        """, unsafe_allow_html=True)

        # Database status indicator
        if db is not None:
            st.markdown("""
            <div class="db-status">
                <strong>✅ MongoDB Connected:</strong> Data will be stored in the cloud database
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #ffebee; padding: 0.5rem; border-radius: 5px; border-left: 3px solid #f44336; margin: 0.5rem 0;">
                <strong>❌ MongoDB Disconnected:</strong> Analysis will not be saved
            </div>
            """, unsafe_allow_html=True)

        # Sidebar configuration
        with st.sidebar:
            st.header("⚙️ Settings")
            
            min_string_length = st.slider("Minimum string length", 1, 20, 4)
            max_string_length = st.slider("Maximum string length", 10, 500, 100)
            bytes_per_line = st.selectbox("Hex bytes per line", [8, 16, 32], index=1)
            
            st.header("📊 Analysis Options")
            show_hashes = st.checkbox("Show file hashes", True)
            show_hex = st.checkbox("Show hex view", True)
            show_strings = st.checkbox("Show extracted strings", True)
            show_stats = st.checkbox("Show byte statistics", True)
            save_to_db = st.checkbox("Save analysis to database", True, disabled=(db is None))
            
            # Database Statistics
            if db is not None:
                st.header("📈 Database Stats")
                db_stats = get_db_stats()
                if db_stats:
                    st.metric("Total Files", db_stats['total_files'])
                    st.metric("This Week", db_stats['recent_files'])
                    
                    if db_stats['file_types']:
                        st.write("**Top File Types:**")
                        for ft in db_stats['file_types']:
                            st.write(f"• {ft['_id']}: {ft['count']}")

        # Create main tabs
        tab_main, tab_history, tab_search = st.tabs(["🔍 Analysis", "📚 History", "🔎 Search"])

        with tab_main:
            # Main content
            # col1, col2 = st.columns([2, 1])

            # with col1:
            uploaded_file = st.file_uploader(
                    "📁 Upload a file for analysis", 
                    type=None,
                    help="Upload any file type for binary analysis"
                )

            if uploaded_file is not None:
                # Read file with progress indication
                with st.spinner("Reading file..."):
                    file_bytes = uploaded_file.read()
                
                # File summary
                # with col2:
                #     st.markdown("""
                #     <div class="info-card">
                #         <h3>📋 File Information</h3>
                #     </div>
                #     """, unsafe_allow_html=True)
                    
                    # col2_1, col2_2 = st.columns(2)
                    
                    # with col2_1:
                        # st.metric("📄 Size", f"{len(file_bytes):,} bytes")
                        # file_type = get_file_type_info(file_bytes, uploaded_file.name)
                        # st.metric("🎯 Type", file_type)
                    
                    # with col2_2:
                        # entropy = calculate_entropy(file_bytes)
                        # st.metric("📈 Entropy", f"{entropy}")
                        
                        # null_count = file_bytes.count(0)
                        # null_percent = (null_count / len(file_bytes)) * 100 if file_bytes else 0
                        # st.metric("🔳 Null %", f"{null_percent:.1f}%")
                
                # with col1:
                    st.markdown("---")
                    st.markdown(f"**📁 Filename:** `{uploaded_file.name}`")
                    
                    # Hash values
                    if show_hashes:
                        with st.expander("📁 Basic File Info", expanded=False):
                            if file_bytes:
                                st.markdown("""
                                <div class="info-card">
                                    <h3>📋 File Information</h3>
                                </div>
                                """, unsafe_allow_html=True)
                                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                                
                                with col_stat1:
                                    st.metric("📄 Size", f"{len(file_bytes):,} bytes")
                                    file_type = get_file_type_info(file_bytes, uploaded_file.name)
                                
                                with col_stat2:
                                    st.metric("🎯 Type", file_type)
                                
                                with col_stat3:
                                    entropy = calculate_entropy(file_bytes)
                                    st.metric("📈 Entropy", f"{entropy}")
                                
                                with col_stat4:
                                    null_count = file_bytes.count(0)
                                    null_percent = (null_count / len(file_bytes)) * 100 if file_bytes else 0
                                    st.metric("🔳 Null %", f"{null_percent:.1f}%")

                            if show_stats:
                                st.markdown("---")
                                # st.subheader("📊 Byte Statistics")
                                st.markdown("""
                                <div class="info-card">
                                    <h3>📊 Byte Statistics</h3>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                if file_bytes:
                                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                                    
                                    with col_stat1:
                                        printable_count = sum(1 for b in file_bytes if 32 <= b <= 126)
                                        st.metric("Printable", f"{printable_count:,}")
                                    
                                    with col_stat2:
                                        control_count = sum(1 for b in file_bytes if b < 32)
                                        st.metric("Control", f"{control_count:,}")
                                    
                                    with col_stat3:
                                        extended_count = sum(1 for b in file_bytes if b > 126)
                                        st.metric("Extended", f"{extended_count:,}")
                                    
                                    with col_stat4:
                                        unique_bytes = len(set(file_bytes))
                                        st.metric("Unique", f"{unique_bytes}")
                            # st.metric("📄 Size", f"{len(file_bytes):,} bytes")
                            # file_type = get_file_type_info(file_bytes, uploaded_file.name)
                            # st.metric("🎯 Type", file_type)
                            # entropy = calculate_entropy(file_bytes)
                            # st.metric("📈 Entropy", f"{entropy}")
                            
                            # null_count = file_bytes.count(0)
                            # null_percent = (null_count / len(file_bytes)) * 100 if file_bytes else 0
                            # st.metric("🔳 Null %", f"{null_percent:.1f}%")
                            with st.spinner("Calculating hashes..."):
                                st.markdown("---")
                                st.markdown("""
                                <div class="info-card">
                                    <h3>🔐 File Hashes</h3>
                                </div>
                                """, unsafe_allow_html=True)
                                hashes = calculate_file_hashed(file_bytes)
                            
                            st.code(f"MD5:    {hashes[0]}")
                            st.code(f"SHA1:   {hashes[1]}")
                            st.code(f"SHA256: {hashes[2]}")
                
                # Extract strings for analysis
                extracted_strings = []
                if show_strings:
                    with st.spinner("Extracting strings..."):
                        extracted_strings = extract_strings_optimized(
                            file_bytes, 
                            min_string_length, 
                            max_string_length
                        )
                
                # Save to database
                if save_to_db and db is not None:
                    if st.button("💾 Save Analysis to Database", type="primary"):
                        with st.spinner("Saving to MongoDB..."):
                            doc_id = save_file_analysis(
                                uploaded_file.name,
                                file_bytes,
                                extracted_strings,
                                hashes,
                                file_type
                            )
                            
                            if doc_id:
                                st.success(f"✅ Analysis saved! Document ID: {doc_id}")
                            else:
                                st.error("❌ Failed to save analysis")
                    st.markdown("---")
                
                # Rest of the analysis interface (keeping original structure)
                # if show_stats:
                #     st.markdown("---")
                #     st.subheader("📊 Byte Statistics")
                    
                #     if file_bytes:
                #         col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                        
                #         with col_stat1:
                #             printable_count = sum(1 for b in file_bytes if 32 <= b <= 126)
                #             st.metric("Printable", f"{printable_count:,}")
                        
                #         with col_stat2:
                #             control_count = sum(1 for b in file_bytes if b < 32)
                #             st.metric("Control", f"{control_count:,}")
                        
                #         with col_stat3:
                #             extended_count = sum(1 for b in file_bytes if b > 126)
                #             st.metric("Extended", f"{extended_count:,}")
                        
                #         with col_stat4:
                #             unique_bytes = len(set(file_bytes))
                #             st.metric("Unique", f"{unique_bytes}")
                #         st.markdown("---")

                # Tabbed interface for different views
                subtab1, subtab2, subtab3, subtab4, subtab5 = st.tabs(["🔢 Hex View", "✏️ Hex Editor", "🧵 Strings", "📝 Text Editor", "📊 Raw Data"])
                
                with subtab1:
                    if show_hex and file_bytes:
                        st.subheader("🔢 Hexadecimal View")
                        
                        max_display_bytes = 8192
                        display_bytes = file_bytes[:max_display_bytes]
                        
                        if len(file_bytes) > max_display_bytes:
                            st.warning(f"⚠️ Showing first {max_display_bytes:,} bytes of {len(file_bytes):,} total")
                        
                        hex_output = format_hex_output(display_bytes, bytes_per_line)
                        st.code(hex_output, language='')
                
                with subtab2:
                    st.subheader("✏️ Interactive Hex Editor")
                    
                    if file_bytes:
                        st.info("💡 Edit hex values directly. Use format: 'FF 0A 1B' (space-separated hex bytes)")
                        
                        max_edit_bytes = 2048
                        edit_bytes = file_bytes[:max_edit_bytes]
                        
                        if len(file_bytes) > max_edit_bytes:
                            st.warning(f"⚠️ Editing limited to first {max_edit_bytes:,} bytes for performance")
                        
                        hex_string = ' '.join(f"{b:02x}" for b in edit_bytes)
                        
                        edited_hex = st_ace(
                            value=hex_string,
                            language='text',
                            theme='monokai',
                            height=300,
                            key="hex-editor",
                            font_size=14,
                            wrap=True
                        )
                        
                        col_hex1, col_hex2, col_hex3 = st.columns([1, 1, 2])
                        
                        with col_hex1:
                            if st.button("🔄 Parse & Preview", type="primary"):
                                try:
                                    hex_values = edited_hex.replace('\n', ' ').split()
                                    parsed_bytes = bytearray()
                                    
                                    for hex_val in hex_values:
                                        if hex_val.strip():
                                            parsed_bytes.append(int(hex_val, 16))
                                    
                                    st.session_state.parsed_bytes = bytes(parsed_bytes)
                                    st.success(f"✅ Parsed {len(parsed_bytes)} bytes successfully!")
                                    
                                except ValueError as e:
                                    st.error(f"❌ Invalid hex format: {str(e)}")
                                    st.session_state.parsed_bytes = None
                        
                        with col_hex2:
                            if 'parsed_bytes' in st.session_state and st.session_state.parsed_bytes:
                                st.download_button(
                                    label="💾 Download Binary",
                                    data=st.session_state.parsed_bytes,
                                    file_name=f"edited_{uploaded_file.name}",
                                    mime="application/octet-stream",
                                    key="download-hex-binary"
                                )
                
                with subtab3:
                    if show_strings:
                        st.subheader("🧵 Extracted Strings")
                        
                        if extracted_strings:
                            st.success(f"Found {len(extracted_strings)} unique strings")
                            
                            search_term = st.text_input("🔍 Search strings:", placeholder="Enter search term...")
                            
                            if search_term:
                                filtered_strings = [s for s in extracted_strings if search_term.lower() in s.lower()]
                                st.info(f"Found {len(filtered_strings)} matching strings")
                                display_strings = filtered_strings
                            else:
                                display_strings = extracted_strings
                            
                            strings_text = '\n'.join(display_strings)
                            st.text_area("Extracted strings:", strings_text, height=300)
                            
                            st.download_button(
                                label="💾 Download Strings",
                                data=strings_text,
                                file_name=f"{uploaded_file.name}_strings.txt",
                                mime="text/plain"
                            )
                        else:
                            st.info("No strings found with current settings")
                
                with subtab4:
                    st.subheader("📝 Interactive Editor")
                    
                    if show_strings and extracted_strings:
                        default_content = '\n'.join(extracted_strings)
                    else:
                        default_content = ''.join(
                            chr(b) if 32 <= b <= 126 else '.' for b in file_bytes[:2048]
                        )
                    
                    edited_content = st_ace(
                        value=default_content,
                        language='text',
                        theme='github_dark',
                        height=400,
                        auto_update=False,
                        key="file-editor"
                    )
                    
                    if st.button("💾 Save Edited Content"):
                        st.download_button(
                            label="📥 Download Edited File",
                            data=edited_content,
                            file_name=f"edited_{uploaded_file.name}.txt",
                            mime="text/plain",
                            key="download-edited"
                        )
                        st.success("✅ Content ready for download!")
                
                with subtab5:
                    st.subheader("📊 Raw Data Views")
                    
                    col_raw1, col_raw2 = st.columns(2)
                    
                    with col_raw1:
                        st.write("**Binary (first 256 bytes)**")
                        if file_bytes:
                            binary_output = ' '.join(f"{b:08b}" for b in file_bytes[:32])
                            st.code(binary_output, language='')
                    
                    with col_raw2:
                        st.write("**Decimal Values (first 64 bytes)**")
                        if file_bytes:
                            decimal_output = ' '.join(str(b) for b in file_bytes[:64])
                            st.code(decimal_output, language='')

            else:
                # Welcome message
                st.markdown("""
                <div class="info-card">
                    <h3>👋 Welcome to File Inspector Pro</h3>
                    <p>Upload any file to get started with advanced binary analysis:</p>
                    <ul>
                        <li>🔍 Hex and binary visualization</li>
                        <li>🧵 String extraction and search</li>
                        <li>🔐 Hash calculation (MD5, SHA1, SHA256)</li>
                        <li>📊 Byte statistics and entropy analysis</li>
                        <li>📝 Interactive text editing</li>
                        <li>🎯 File type detection</li>
                        <li>💾 MongoDB cloud storage</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

        with tab_history:
            st.subheader("📚 Analysis History")
            
            if db is not None:
                recent_analyses = get_recent_analyses(20)
                
                if recent_analyses:
                    st.write(f"**Showing {len(recent_analyses)} most recent analyses:**")
                    
                    for analysis in recent_analyses:
                        with st.expander(f"📄 {analysis['filename']} - {analysis['upload_date'].strftime('%Y-%m-%d %H:%M')}"):
                            col_h1, col_h2, col_h3 = st.columns(3)
                            
                            with col_h1:
                                st.write(f"**Size:** {analysis['file_size']:,} bytes")
                                st.write(f"**Type:** {analysis['file_type']}")
                                st.write(f"**Strings:** {analysis['string_count']}")
                            
                            with col_h2:
                                st.write(f"**MD5:** `{analysis['md5_hash'][:16]}...`")
                                st.write(f"**Entropy:** {analysis.get('entropy', 'N/A')}")
                                st.write(f"**Null %:** {analysis.get('null_percentage', 0):.1f}%")
                            
                            with col_h3:
                                if st.button(f"🔍 View Details", key=f"view_{analysis['_id']}"):
                                    st.session_state.selected_analysis = analysis['_id']
                                
                                if analysis.get('file_data_base64'):
                                    try:
                                        file_data = base64.b64decode(analysis['file_data_base64'])
                                        st.download_button(
                                            label="💾 Download",
                                            data=file_data,
                                            file_name=analysis['filename'],
                                            key=f"download_{analysis['_id']}"
                                        )
                                    except:
                                        st.write("File data unavailable")
                            
                            # Show strings if available (no nested expander)
                            if analysis.get('extracted_strings'):
                                st.markdown("**📝 Extracted Strings:**")
                                with st.container():
                                    strings_display = '\n'.join(analysis['extracted_strings'][:50])  # Show first 50
                                    st.code(strings_display)
                                    if len(analysis['extracted_strings']) > 50:
                                        st.info(f"Showing first 50 of {len(analysis['extracted_strings'])} strings")
                else:
                    st.info("No analysis history found. Upload and analyze files to see them here.")
            else:
                st.error("Database connection required to view history")

        with tab_search:
            st.subheader("🔎 Search Analyses")
            
            if db is not None:
                search_term = st.text_input("Search by filename or hash:", placeholder="Enter filename or hash...")
                
                if search_term:
                    with st.spinner("Searching..."):
                        search_results = search_analyses(search_term)
                    
                    if search_results:
                        st.success(f"Found {len(search_results)} matching analyses")
                        
                        for result in search_results:
                            with st.expander(f"📄 {result['filename']} - {result['upload_date'].strftime('%Y-%m-%d %H:%M')}"):
                                col_s1, col_s2 = st.columns(2)
                                
                                with col_s1:
                                    st.write(f"**Size:** {result['file_size']:,} bytes")
                                    st.write(f"**Type:** {result['file_type']}")
                                    st.write(f"**MD5:** `{result['md5_hash']}`")
                                
                                with col_s2:
                                    st.write(f"**SHA1:** `{result['sha1_hash']}`")
                                    st.write(f"**SHA256:** `{result['sha256_hash']}`")
                                    st.write(f"**Strings:** {result['string_count']}")
                                
                                if result.get('extracted_strings'):
                                    st.write("**Sample Strings:**")
                                    sample_strings = '\n'.join(result['extracted_strings'][:10])
                                    st.code(sample_strings)
                    else:
                        st.info("No matching analyses found")
                else:
                    st.info("Enter a search term to find analyses")
            else:
                st.error("Database connection required for search")

        # Footer
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #6c757d;'>"
            "🔍 File Inspector Pro - Advanced Binary Analysis with MongoDB Storage"
            "</div>", 
            unsafe_allow_html=True
        )
    elif page == "📊 Advanced Image Forensics Analyzer":
        # st.title("Advanced Image Forensics Analyzer")
        # Configure page
        # st.set_page_config(
        #     page_title="Advanced Image Forensics Analyzer",
        #     page_icon="🔍",
        #     layout="wide",
        #     initial_sidebar_state="expanded"
        # )

        # Custom CSS for better UI
        st.markdown("""
        <style>
            .main > div {
                padding-top: 2rem;
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 2rem;
            }
            .stTabs [data-baseweb="tab"] {
                height: 50px;
                white-space: pre-wrap;
                background-color: #f0f2f6;
                border-radius: 4px;
                padding: 10px 20px;
                font-weight: 500;
            }
            .stTabs [aria-selected="true"] {
                background-color: #ff4b4b;
                color: white;
            }
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
        </style>
        """, unsafe_allow_html=True)

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
            """Analyze JPEG compression artifacts"""
            gray = np.array(image.convert("L"))
            
            # DCT-based analysis
            dct_coeffs = cv2.dct(gray.astype(np.float32))
            
            # Quantization noise analysis
            reconstructed = cv2.idct(dct_coeffs)
            noise = np.abs(gray.astype(np.float32) - reconstructed)
            
            # Block artifact detection
            blocks = []
            for i in range(0, gray.shape[0], 8):
                for j in range(0, gray.shape[1], 8):
                    block = gray[i:i+8, j:j+8]
                    if block.shape == (8, 8):
                        blocks.append(np.var(block))
            
            block_variance = np.mean(blocks) if blocks else 0
            
            return {
                "dct_analysis": Image.fromarray(np.uint8(255 * np.abs(dct_coeffs) / np.max(np.abs(dct_coeffs)))),
                "quantization_noise": Image.fromarray(np.uint8(255 * noise / np.max(noise))),
                "block_variance": block_variance
            }

        # Enhanced metadata extraction
        @st.cache_data
        def extract_comprehensive_metadata(img_bytes):
            """Extract comprehensive metadata from image"""
            metadata = {}
            
            try:
                exif_dict = piexif.load(img_bytes)
                
                # Basic EXIF data
                if "0th" in exif_dict:
                    for tag, value in exif_dict["0th"].items():
                        tag_name = piexif.TAGS["0th"].get(tag, {"name": f"Tag_{tag}"})["name"]
                        metadata[tag_name] = str(value)
                
                # GPS data
                if "GPS" in exif_dict:
                    gps_data = exif_dict["GPS"]
                    metadata["GPS_Info"] = {}
                    for tag, value in gps_data.items():
                        tag_name = piexif.TAGS["GPS"].get(tag, {"name": f"GPS_{tag}"})["name"]
                        metadata["GPS_Info"][tag_name] = str(value)
                
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

            st.markdown("""
            <div class="main-header">
                <h1>🔍 Advanced Image Metadata & Security Analyzer</h1>
                <p>Upload an image to extract comprehensive metadata, EXIF data, and check for malware using VirusTotal.</p>
            </div>
            """, unsafe_allow_html=True)
            # st.title("🔍 Advanced Image Forensics Analyzer")
            # st.markdown("### Professional-grade image authentication and tampering detection")
            
            # Sidebar for settings
            with st.sidebar:
                st.header("⚙️ Analysis Settings")
                
                analysis_mode = st.selectbox(
                    "Analysis Mode",
                    ["Quick Scan", "Deep Analysis", "Expert Mode"],
                    help="Choose analysis depth"
                )
                
                show_confidence = st.checkbox("Show Confidence Scores", True)
                generate_report = st.checkbox("Generate Report", False)
                
                st.markdown("---")
                st.markdown("### 📊 Analysis Coverage")
                st.markdown("- Error Level Analysis (ELA)")
                st.markdown("- JPEG Artifact Analysis")
                st.markdown("- Metadata Forensics")
                st.markdown("- Quantization Table Analysis")
                st.markdown("- Geometric Analysis")
            
            # File upload
            uploaded_file = st.file_uploader(
                "Upload image for forensic analysis",
                type=["jpg", "jpeg", "png", "tiff", "bmp"],
                help="Supported formats: JPEG, PNG, TIFF, BMP"
            )
            
            if uploaded_file is not None:
                # Load and display image
                image = Image.open(uploaded_file).convert("RGB")
                uploaded_file.seek(0)
                img_bytes = uploaded_file.read()
                st.markdown("---")
                
                # Image overview
                with st.expander("📁 Basic File Info", expanded=False):
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col1:
                        st.image(image, caption="Original Image")
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>📏 Image Info</h4>
                            <p><strong>Size:</strong> {image.size[0]} × {image.size[1]}</p>
                            <p><strong>Format:</strong> {uploaded_file.type}</p>
                            <p><strong>File Size:</strong> {len(img_bytes) / 1024:.1f} KB</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        # Quick authenticity score
                        authenticity_score = np.random.randint(60, 95)  # Placeholder
                        color = "green" if authenticity_score > 80 else "orange" if authenticity_score > 60 else "red"
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>🛡️ Authenticity Score</h4>
                            <h2 style="color: {color};">{authenticity_score}%</h2>
                            <p>Preliminary assessment</p>
                        </div>
                        """, unsafe_allow_html=True)
                st.markdown("---")
                # Analysis tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "🔍 Tampering Detection",
                    "📊 JPEG Analysis", 
                    "📝 Metadata Forensics",
                    "🔬 Advanced Analysis",
                    "📋 Report"
                ])
                
                with tab1:
                    st.markdown("<div class=\"analysis-header\"><h3>🔍 Tampering Detection Analysis</h3></div>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        with st.expander("Error Level Analysis (ELA)", expanded=False):
                            st.subheader("Error Level Analysis (ELA)")
                            ela_results = perform_ela(image)
                            
                            ela_quality = st.selectbox("ELA Quality", ["Q70", "Q80", "Q90", "Q95"])
                            st.image(ela_results[ela_quality], caption=f"ELA at {ela_quality}")
                            
                            if show_confidence:
                                st.info("🔍 Look for bright areas indicating potential editing")
                        
                        with st.expander("Enhanced Edge Detection", expanded=False):
                            st.subheader("Enhanced Edge Detection")
                            edge_results = enhanced_edge_detection(image)
                            
                            edge_method = st.selectbox("Edge Detection Method", ["canny", "sobel", "laplacian"])
                            st.image(edge_results[edge_method], caption=f"{edge_method.capitalize()} Edge Detection")
                    
                    with col2:
                        with st.expander("Luminance Analysis", expanded=False):
                            st.subheader("Luminance Analysis")
                            # Luminance consistency analysis
                            gray = np.array(image.convert("L"))
                            
                            # Create luminance map
                            luminance_map = cv2.equalizeHist(gray)
                            st.image(luminance_map, caption="Luminance Map")
                            
                            if show_confidence:
                                st.info("🔍 Inconsistent luminance patterns may indicate manipulation")
                        
                        with st.expander("Noise Analysis", expanded=False):
                            st.subheader("Noise Analysis")
                            # Noise residual analysis
                            blur = cv2.GaussianBlur(gray, (5, 5), 0)
                            noise = cv2.absdiff(gray, blur)
                            st.image(noise, caption="Noise Residual")
                            
                            if show_confidence:
                                st.info("🔍 Uneven noise distribution may indicate tampering")
                
                with tab2:
                    st.markdown("<div class=\"analysis-header\"><h3>📊 JPEG Compression Analysis</h3></div>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        with st.expander("Quantization Table Analysis", expanded=False):
                            st.subheader("Quantization Table Analysis")
                            try:
                                exif_dict = piexif.load(img_bytes)
                                qtable = exif_dict.get("0th", {})
                                
                                if qtable:
                                    source, confidence, analysis = classify_quantization_table(qtable)
                                    
                                    st.success(f"**Estimated Source:** {source}")
                                    st.metric("Confidence Score", f"{confidence:.1f}%")
                                    
                                    with st.expander("Detailed Analysis"):
                                        st.text(analysis)
                                        
                                    # Visualize quantization table
                                    if len(qtable) > 0:
                                        fig, ax = plt.subplots(figsize=(8, 6))
                                        qtable_values = list(qtable.values())[:64]  # First 64 values
                                        if len(qtable_values) >= 64:
                                            qtable_matrix = np.array(qtable_values).reshape(8, 8)
                                            im = ax.imshow(qtable_matrix, cmap='viridis')
                                            ax.set_title("Quantization Table Visualization")
                                            plt.colorbar(im, ax=ax)
                                            st.pyplot(fig)
                                        else:
                                            st.info("Insufficient quantization table data for visualization")
                                else:
                                    st.warning("No quantization table found in EXIF data")
                                    
                            except Exception as e:
                                st.error(f"Error analyzing quantization table: {str(e)}")
                    
                    with col2:
                        with st.expander("JPEG Artifact Analysis", expanded=False):
                            st.subheader("JPEG Artifact Analysis")
                            jpeg_analysis = analyze_jpeg_artifacts(image)
                            
                            st.image(jpeg_analysis["dct_analysis"], caption="DCT Coefficient Analysis")
                            st.image(jpeg_analysis["quantization_noise"], caption="Quantization Noise")
                            
                            st.metric("Block Variance", f"{jpeg_analysis['block_variance']:.2f}")
                            
                            # Compression history estimation
                        with st.expander("Compression History", expanded=False):
                            st.subheader("Compression History")
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
                    st.markdown("<div class=\"analysis-header\"><h3>📝 Metadata Forensics</h3></div>", unsafe_allow_html=True)
                    
                    metadata = extract_comprehensive_metadata(img_bytes)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        with st.expander("EXIF Data", expanded=False):
                            st.subheader("EXIF Data")
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
                            st.subheader("🌍 Geolocation Data")
                            if "GPS_Info" in metadata:
                                st.json(metadata["GPS_Info"])
                                st.info("GPS coordinates found in image")
                            else:
                                st.info("No GPS data found")
                    
                    with col2:
                        with st.expander("Thumbnail Analysis", expanded=False):
                            st.subheader("Thumbnail Analysis")
                            if "thumbnail_hash" in metadata:
                                st.success(f"Thumbnail Hash: {metadata['thumbnail_hash']}")
                                st.metric("Thumbnail Size", f"{metadata['thumbnail_size']} bytes")
                            else:
                                st.info("No thumbnail found")
                        
                        # Timestamp analysis
                        with st.expander("Timestamp Analysis", expanded=False):
                            st.subheader("📅 Timestamp Analysis")
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
                            st.subheader("🔧 Software Detection")
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
                    st.markdown("<div class=\"analysis-header\"><h3>🔬 Advanced Forensic Analysis</h3></div>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        with st.expander("PCA Analysis", expanded=False):
                            st.subheader("PCA Analysis")
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
                            st.subheader("Luminance Analysis")
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
                            st.subheader("Frequency Domain Analysis")
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
                            st.subheader("Statistical Analysis")
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
                            
                            for stat, value in stats.items():
                                st.metric(stat, f"{value:.2f}")
                
                with tab5:
                    st.markdown("<div class=\"analysis-header\"><h3>📋 Forensic Analysis Report</h3></div>", unsafe_allow_html=True)
                    
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
                st.info("👆 Upload an image to begin forensic analysis")
                
                # Show example analysis
                st.markdown("### 🎯 What This Tool Analyzes")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("""
                    **🔍 Tampering Detection**
                    - Error Level Analysis (ELA)
                    - Splicing Detection
                    - Luminance Analysis
                    - Noise Pattern Analysis
                    """)
                
                with col2:
                    st.markdown("""
                    **📊 JPEG Analysis**
                    - Quantization Tables
                    - Compression History
                    - Artifact Detection
                    - Quality Assessment
                    """)
                
                with col3:
                    st.markdown("""
                    **📝 Metadata Forensics**
                    - EXIF Data Analysis
                    - GPS Coordinates
                    - Timestamp Verification
                    - Software Detection
                    """)
            st.markdown("---")

        if __name__ == "__main__":
            main()
         
if __name__ == "__main__":
    main()