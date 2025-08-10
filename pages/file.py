import streamlit as st
from streamlit_ace import st_ace
import string
import hashlib
from pymongo import MongoClient
from dotenv import load_dotenv
import base64
import os
from datetime import datetime, timedelta, timezone

load_dotenv()

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
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 2rem 0;
        border-top: 1px solid #e0e0e0;
        color: #666;
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
def to_rational(value, precision=100):
    return (int(value * precision), precision)

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
    
st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    """,
    unsafe_allow_html=True,
)

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

st.markdown("""
<div class="main-header">
    <h1><i class="fa-solid fa-magnifying-glass"></i> File Inspector</h1>
    <p>Advanced file analysis with MongoDB storage and historical tracking</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.sidebar.markdown("---")
    with st.expander("Settings", expanded=False):
        st.markdown('### <i class="fa-solid fa-sliders"></i> Analysis Options', unsafe_allow_html=True)
        # st.header("📊 Analysis Options")
        show_hashes = st.checkbox("Show file hashes", True)
        show_hex = st.checkbox("Show hex view", True)
        show_strings = st.checkbox("Show extracted strings", True)
        show_stats = st.checkbox("Show byte statistics", True)
        save_to_db = st.checkbox("Save analysis to database", True, disabled=(db is None))
        
        min_string_length = st.slider("Minimum string length", 1, 20, 4)
        max_string_length = st.slider("Maximum string length", 10, 500, 100)
        bytes_per_line = st.selectbox("Hex bytes per line", [8, 16, 32], index=1)
        
    # Database Statistics
    with st.expander("Database Stats", expanded=False):
        if db is not None:
            st.markdown('## <i class="fa-solid fa-chart-line"></i> Database Stats', unsafe_allow_html=True)
            # st.header("📈 Database Stats")
            db_stats = get_db_stats()
            if db_stats:
                st.metric("Total Files", db_stats['total_files'])
                st.metric("This Week", db_stats['recent_files'])
                
                if db_stats['file_types']:
                    st.write("**Top File Types:**")
                    for ft in db_stats['file_types']:
                        st.write(f"• {ft['_id']}: {ft['count']}")
    st.sidebar.markdown("---")

# Create main tabs
tab_main, tab_history, tab_search = st.tabs(["🔍 Analysis", "📚 History", "🔎 Search"])

with tab_main:
    # Main content
    uploaded_file = st.file_uploader(
            "📁 Upload a file for analysis", 
            type=None,
            help="Upload any file type for binary analysis"
        )

    if uploaded_file is not None:
        # Read file with progress indication
        with st.spinner("Reading file..."):
            file_bytes = uploaded_file.read()
        
            st.markdown("---")
            
            # Hash values
            if show_hashes:
                st.markdown("<h4><i class='fa-solid fa-file'></i> Basic File Info</h4>", unsafe_allow_html=True)
                with st.expander("Details", expanded=False):
                    if file_bytes:
                        st.markdown("""
                        <div class="info-card">
                            <h3>File Information</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                        
                        with col_stat1:
                            st.metric("Size", f"{len(file_bytes):,} bytes")
                            file_type = get_file_type_info(file_bytes, uploaded_file.name)
                        
                        with col_stat2:
                            st.metric("Type", file_type)
                        
                        with col_stat3:
                            entropy = calculate_entropy(file_bytes)
                            st.metric("Entropy", f"{entropy}")
                        
                        with col_stat4:
                            null_count = file_bytes.count(0)
                            null_percent = (null_count / len(file_bytes)) * 100 if file_bytes else 0
                            st.metric("Null %", f"{null_percent:.1f}%")

                    if show_stats:
                        st.markdown("---")
                        st.markdown("""
                        <div class="info-card">
                            <h3>Byte Statistics</h3>
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
                    with st.spinner("Calculating hashes..."):
                        st.markdown("---")
                        st.markdown("""
                        <div class="info-card">
                            <h3>File Hashes</h3>
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

        # Tabbed interface for different views
        st.markdown("<h4><i class='fa-solid fas fa-print'></i> Hex Editor & Tools</h4>", unsafe_allow_html=True)
        with st.expander("Details", expanded=False):
            subtab1, subtab2, subtab3, subtab4, subtab5 = st.tabs(["Hex View", "Hex Editor", "Strings", "Text Editor", "Raw Data"])
        
        with subtab1:
            if show_hex and file_bytes:
                st.subheader("Hexadecimal View")
                
                max_display_bytes = 8192
                display_bytes = file_bytes[:max_display_bytes]
                
                if len(file_bytes) > max_display_bytes:
                    st.warning(f"⚠️ Showing first {max_display_bytes:,} bytes of {len(file_bytes):,} total")
                
                hex_output = format_hex_output(display_bytes, bytes_per_line)
                st.code(hex_output, language='')
        
        with subtab2:
            st.subheader("Interactive Hex Editor")
            
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
                    if st.button("Parse & Preview", type="primary"):
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
                            label="Download Binary",
                            data=st.session_state.parsed_bytes,
                            file_name=f"edited_{uploaded_file.name}",
                            mime="application/octet-stream",
                            key="download-hex-binary"
                        )
        
        with subtab3:
            if show_strings:
                st.subheader("Extracted Strings")
                
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
                        label="Download Strings",
                        data=strings_text,
                        file_name=f"{uploaded_file.name}_strings.txt",
                        mime="text/plain"
                    )
                else:
                    st.info("No strings found with current settings")
        
        with subtab4:
            st.subheader("Interactive Editor")
            
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
            
            if st.button("Save Edited Content"):
                st.download_button(
                    label="Download Edited File",
                    data=edited_content,
                    file_name=f"edited_{uploaded_file.name}.txt",
                    mime="text/plain",
                    key="download-edited"
                )
                st.success("✅ Content ready for download!")
        
        with subtab5:
            st.subheader("Raw Data Views")
            
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
        
        # Save to database
        if save_to_db and db is not None:
            if st.button("Save Analysis to Database", type="primary"):
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
            # st.markdown("---")
    else:
        # Welcome message
        st.markdown("""
        <div class="info-card">
            <h3> Welcome to File Inspector</h3>
            <p>Upload any file to get started with advanced binary analysis:</p>
            <ul>
                <li><i class="fa-solid fa-magnifying-glass"></i> Hex and binary visualization</li>
                <li><i class="fa-solid fa-spaghetti-monster-flying"></i> String extraction and search</li>
                <li><i class="fa-solid fa-lock"></i> Hash calculation (MD5, SHA1, SHA256)</li>
                <li><i class="fa-solid fa-chart-bar"></i> Byte statistics and entropy analysis</li>
                <li><i class="fa-solid fa-pen-to-square"></i> Interactive text editing</li>
                <li><i class="fa-solid fa-bullseye"></i> File type detection</li>
                <li><i class="fa-solid fa-database"></i> MongoDB cloud storage</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

with tab_history:
    st.markdown('### <i class="fa-solid fa-book"></i> Analysis History', unsafe_allow_html=True)
    # st.subheader("📚 Analysis History")
    
    if db is not None:
        recent_analyses = get_recent_analyses(20)
        
        if recent_analyses:
            st.write(f"**Showing {len(recent_analyses)} most recent analyses:**")
            
            for analysis in recent_analyses:
                with st.expander(f"{analysis['filename']} - {analysis['upload_date'].strftime('%Y-%m-%d %H:%M')}"):
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
                        if st.button(f"View Details", key=f"view_{analysis['_id']}"):
                            st.session_state.selected_analysis = analysis['_id']
                        
                        if analysis.get('file_data_base64'):
                            try:
                                file_data = base64.b64decode(analysis['file_data_base64'])
                                st.download_button(
                                    label="Download",
                                    data=file_data,
                                    file_name=analysis['filename'],
                                    key=f"download_{analysis['_id']}"
                                )
                            except:
                                st.write("File data unavailable")
                    
                    # Show strings if available (no nested expander)
                    if analysis.get('extracted_strings'):
                        st.markdown('<strong><i class="fa-solid fa-file-lines"></i> Extracted Strings:</strong>', unsafe_allow_html=True)
                        # st.markdown("**📝 Extracted Strings:**")
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
    st.markdown('<h3><i class="fa-solid fa-magnifying-glass"></i> Search Analyses</h3>', unsafe_allow_html=True)
    # st.subheader("🔎 Search Analyses")
    
    if db is not None:
        search_term = st.text_input("Search by filename or hash:", placeholder="Enter filename or hash...")
        
        if search_term:
            with st.spinner("Searching..."):
                search_results = search_analyses(search_term)
            
            if search_results:
                st.success(f"Found {len(search_results)} matching analyses")
                
                for result in search_results:
                    with st.expander(f"{result['filename']} - {result['upload_date'].strftime('%Y-%m-%d %H:%M')}"):
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
# st.markdown("---")
st.markdown("""
    <div class="footer">
        <h4>Image Analyzer Suite</h4>
        <p><strong>© 2025 Radheshyam Janwa</strong> | All Rights Reserved</p>

    </div>
    """, unsafe_allow_html=True)
