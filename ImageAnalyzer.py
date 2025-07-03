import streamlit as st
import hashlib
import binascii
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import io
import struct
import json
import os
from datetime import datetime
import requests
import time
import piexif

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
    
    st.title("🔍 Advanced Image Metadata & Security Analyzer")
    st.markdown("Upload an image to extract comprehensive metadata, EXIF data, and check for malware using VirusTotal.")
    
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
        
        # Calculate hashes early for VirusTotal
        md5_hash = calculate_file_hash(file_bytes, 'md5')
        sha256_hash = calculate_file_hash(file_bytes, 'sha256')
        
        # Display image
        col1, col2 = st.columns([1, 2])
        
        with col1:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption=uploaded_file.name, use_column_width=True)
            except Exception as e:
                st.error(f"Could not display image: {e}")
                image = None
        
        with col2:
            # Basic file info
            st.subheader("📁 Basic File Info")
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
        st.subheader("🛡️ Malware & Security Analysis")
        
        if st.button("🔍 Scan with VirusTotal", type="primary"):
            with st.spinner("Scanning file with VirusTotal..."):
                vt_result = process_virustotal_scan(file_bytes, uploaded_file.name, sha256_hash, api_key)
                st.session_state['vt_result'] = vt_result
        
        # Display cached results if available
        if 'vt_result' in st.session_state:
            display_virustotal_results(st.session_state['vt_result'])
        
        # Image properties
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
        st.subheader("🏷️ JFIF Metadata")
        jfif_info = extract_jfif_info(file_bytes)
        if jfif_info:
            for key, value in jfif_info.items():
                st.text(f"{key}: {value}")
        else:
            st.text("No JFIF metadata found")
        
        # ICC Color Profile
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

if __name__ == "__main__":
    main()