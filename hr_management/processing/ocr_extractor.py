"""
OCR Extraction Module for Video Timestamps and Location
Extracts date/time from top-left and location from bottom-right of video frames
"""

import cv2
import numpy as np
from datetime import datetime
import re
try:
    import pytesseract
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: OCR libraries not available. Install pytesseract or easyocr for OCR functionality")

class VideoOCRExtractor:
    """Extract timestamp and location information from video frames using OCR"""
    
    def __init__(self, ocr_engine='easyocr', date_format='DD-MM-YYYY'):
        """
        Initialize OCR extractor
        
        Args:
            ocr_engine: 'easyocr' or 'tesseract'
            date_format: Expected date format (DD-MM-YYYY, MM-DD-YYYY, YYYY-MM-DD)
        """
        self.ocr_engine = ocr_engine
        self.reader = None
        self.date_format = date_format
        
        if OCR_AVAILABLE:
            if ocr_engine == 'easyocr':
                try:
                    # Check if CUDA is available for OCR
                    import torch
                    gpu_available = torch.cuda.is_available()
                    if gpu_available:
                        print("🎮 GPU detected for OCR, using GPU acceleration")
                        self.reader = easyocr.Reader(['en'], gpu=True, verbose=False)
                    else:
                        print("⚠️ No GPU detected for OCR, using CPU")
                        self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                except Exception as e:
                    print(f"⚠️ Failed to initialize EasyOCR with GPU: {e}")
                    self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            elif ocr_engine == 'tesseract':
                # Configure tesseract path for Windows
                import platform
                if platform.system() == 'Windows':
                    # Common tesseract installation paths on Windows
                    tesseract_paths = [
                        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                        r'C:\tesseract\tesseract.exe'
                    ]
                    for path in tesseract_paths:
                        import os
                        if os.path.exists(path):
                            pytesseract.pytesseract.tesseract_cmd = path
                            break
    
    def extract_frame_info(self, frame):
        """
        Extract timestamp and location from a single frame
        
        Args:
            frame: OpenCV frame (numpy array)
            
        Returns:
            dict: {
                'timestamp': datetime object or None,
                'timestamp_text': raw OCR text,
                'location': location string or None,
                'location_text': raw OCR text
            }
        """
        if not OCR_AVAILABLE:
            return {
                'timestamp': None,
                'timestamp_text': 'OCR not available',
                'location': None,
                'location_text': 'OCR not available'
            }
        
        height, width = frame.shape[:2]
        
        # Extract regions of interest
        # Top-left for timestamp (e.g., "05-12-2025 Mon 08:55:13")
        timestamp_roi = frame[0:int(height*0.1), 0:int(width*0.4)]
        
        # Bottom-right for location (e.g., "TANG TRET")
        location_roi = frame[int(height*0.85):height, int(width*0.6):width]
        
        # Extract text from ROIs
        timestamp_text = self._extract_text(timestamp_roi)
        location_text = self._extract_text(location_roi)
        
        # Parse timestamp
        timestamp = self._parse_timestamp(timestamp_text)
        
        # Clean location text
        location = self._clean_location(location_text)
        
        return {
            'timestamp': timestamp,
            'timestamp_text': timestamp_text,
            'location': location,
            'location_text': location_text
        }
    
    def _extract_text(self, roi):
        """Extract text from region of interest"""
        try:
            # Preprocess image for better OCR
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to get black text on white background
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Invert if text is white on dark background
            if np.mean(thresh) < 127:
                thresh = cv2.bitwise_not(thresh)
            
            # Apply dilation to connect text
            kernel = np.ones((2,2), np.uint8)
            processed = cv2.dilate(thresh, kernel, iterations=1)
            
            # Extract text
            if self.ocr_engine == 'easyocr' and self.reader:
                results = self.reader.readtext(processed, detail=0)
                text = ' '.join(results) if results else ''
            else:
                text = pytesseract.image_to_string(processed).strip()
            
            return text
        except Exception as e:
            print(f"OCR extraction error: {e}")
            return ''
    
    def _parse_timestamp(self, text):
        """
        Parse timestamp from OCR text
        Expected format: "05-12-2025 Mon 08:55:13" or similar
        """
        if not text:
            return None
        
        # Common date patterns
        patterns = [
            # DD-MM-YYYY HH:MM:SS
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{4})\s*\w*\s*(\d{1,2}:\d{2}:\d{2})',
            # MM-DD-YYYY HH:MM:SS
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{4})\s*\w*\s*(\d{1,2}:\d{2}:\d{2})',
            # YYYY-MM-DD HH:MM:SS
            r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})\s*\w*\s*(\d{1,2}:\d{2}:\d{2})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                date_str = match.group(1)
                time_str = match.group(2)
                
                # Map display format to strptime format
                format_map = {
                    'DD-MM-YYYY': '%d-%m-%Y',
                    'MM-DD-YYYY': '%m-%d-%Y',
                    'YYYY-MM-DD': '%Y-%m-%d'
                }
                
                # Get primary format based on configuration
                primary_format = format_map.get(self.date_format, '%d-%m-%Y')
                
                # Try configured format first, then fallback to others
                date_formats = [primary_format]
                # Add other formats as fallback (excluding the primary)
                for fmt in ['%d-%m-%Y', '%m-%d-%Y', '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y']:
                    if fmt != primary_format and fmt not in date_formats:
                        date_formats.append(fmt)
                
                for date_format in date_formats:
                    try:
                        datetime_str = f"{date_str} {time_str}"
                        timestamp = datetime.strptime(datetime_str, f"{date_format} %H:%M:%S")
                        return timestamp
                    except ValueError:
                        continue
        
        return None
    
    def _clean_location(self, text):
        """Clean and standardize location text"""
        if not text:
            return None
        
        # Remove extra whitespace and special characters
        location = re.sub(r'[^\w\s-]', '', text)
        location = ' '.join(location.split())
        
        # Convert to uppercase for consistency
        location = location.upper()
        
        # Return None if empty after cleaning
        return location if location else None
    
    def extract_video_info(self, video_path, sample_interval=30):
        """
        Extract timestamp and location info from video
        
        Args:
            video_path: Path to video file
            sample_interval: Extract info every N frames
            
        Returns:
            list: List of extracted info dictionaries
        """
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        extracted_info = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract info at intervals
            if frame_count % sample_interval == 0:
                info = self.extract_frame_info(frame)
                info['frame_number'] = frame_count
                info['video_time'] = frame_count / fps if fps > 0 else 0
                extracted_info.append(info)
            
            frame_count += 1
        
        cap.release()
        
        # Consolidate results
        return self._consolidate_results(extracted_info)
    
    def _consolidate_results(self, results):
        """Consolidate OCR results to get most common/reliable values"""
        if not results:
            return {
                'location': None,
                'timestamps': [],
                'video_date': None
            }
        
        # Get most common location
        locations = [r['location'] for r in results if r['location']]
        location_counts = {}
        for loc in locations:
            location_counts[loc] = location_counts.get(loc, 0) + 1
        
        most_common_location = max(location_counts.items(), key=lambda x: x[1])[0] if location_counts else None
        
        # Extract all valid timestamps
        timestamps = []
        for r in results:
            if r['timestamp']:
                timestamps.append({
                    'timestamp': r['timestamp'],
                    'frame_number': r['frame_number'],
                    'video_time': r['video_time']
                })
        
        # Get video date (most common date from timestamps)
        if timestamps:
            dates = [t['timestamp'].date() for t in timestamps]
            date_counts = {}
            for date in dates:
                date_counts[date] = date_counts.get(date, 0) + 1
            video_date = max(date_counts.items(), key=lambda x: x[1])[0]
        else:
            video_date = None
        
        return {
            'location': most_common_location,
            'timestamps': timestamps,
            'video_date': video_date,
            'extraction_summary': {
                'total_frames_analyzed': len(results),
                'successful_timestamp_extractions': len(timestamps),
                'successful_location_extractions': len(locations),
                'confidence': len(timestamps) / len(results) if results else 0
            }
        }


def install_ocr_dependencies():
    """Install OCR dependencies"""
    import subprocess
    import sys
    
    print("Installing OCR dependencies...")
    
    # Install Python packages
    packages = ['easyocr', 'pytesseract']
    for package in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
    
    print("\nFor Windows users:")
    print("1. Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
    print("2. Install it (remember the installation path)")
    print("3. The code will auto-detect common installation paths")
    
    print("\nOCR dependencies installed successfully!")


if __name__ == '__main__':
    # Test OCR extraction
    print("OCR Extractor Module")
    if not OCR_AVAILABLE:
        print("Installing OCR dependencies...")
        install_ocr_dependencies()