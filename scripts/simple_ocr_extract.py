#!/usr/bin/env python3
"""
Simple OCR extraction for existing videos (Windows-compatible)
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app

def extract_ocr_simple(video_id):
    """Extract OCR data for a single video with minimal output"""
    try:
        Video = globals()['Video']
        DetectedPerson = globals()['DetectedPerson'] 
        db = globals()['db']
        
        video = Video.query.get(video_id)
        if not video:
            return False, "Video not found"
        
        # Find video file
        uploads_dir = Path('static/uploads')
        base_name = Path(video.filename).stem
        video_path = None
        
        # First try to find exact match (original file)
        exact_match = uploads_dir / video.filename
        if exact_match.exists():
            video_path = str(exact_match)
        else:
            # Look for original files (non-annotated) first
            for video_file in uploads_dir.glob('*.mp4'):
                if base_name in video_file.name and 'annotated' not in video_file.name:
                    video_path = str(video_file)
                    break
            
            # If no original found, use any matching file (including annotated)
            if not video_path:
                for video_file in uploads_dir.glob('*.mp4'):
                    if base_name in video_file.name:
                        video_path = str(video_file)
                        break
        
        if not video_path:
            return False, "Video file not found"
        
        # Extract OCR with simple OpenCV + EasyOCR
        import cv2
        import easyocr
        import re
        
        # Initialize reader
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "Could not open video"
        
        # Sample a few frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_frames = min(5, total_frames // 10)  # Sample 5 frames
        
        timestamps = []
        times = []
        locations = []
        
        for i in range(sample_frames):
            frame_num = i * (total_frames // sample_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            height, width = frame.shape[:2]
            
            # Extract timestamp and time (top region) - use full width
            timestamp_region = frame[0:int(height*0.1), 0:width]  # Full width, top 10%
            timestamp_results = reader.readtext(timestamp_region)
            
            for result in timestamp_results:
                text = result[1].strip()
                # Look for date patterns (DD-MM-YYYY, MM-DD-YYYY, etc.)
                if re.search(r'\d{2}[-/]\d{2}[-/]\d{4}', text):
                    timestamps.append(text)
                # Look for time patterns (HH:MM:SS, HH:MM) - extract time from mixed text
                time_match = re.search(r'\d{1,2}:\d{2}(:\d{2})?', text)
                if time_match:
                    times.append(time_match.group())
                
            # Extract location from center-bottom to right-bottom
            # Best region based on testing: bottom 15%, from 30% to right edge
            location_region = frame[int(height*0.85):height, int(width*0.3):width]
            location_results = reader.readtext(location_region)
            
            location_texts = []
            for result in location_results:
                text = result[1].strip()
                confidence = result[2] if len(result) > 2 else 0
                
                # Filter out numbers, times, and very short text
                if (len(text) > 1 and 
                    confidence > 0.5 and  # Only high confidence text
                    not re.match(r'^\d+$', text) and  # Not just numbers
                    not re.search(r'\d{1,2}:\d{2}', text) and  # Not time
                    not re.search(r'\d{2}[-/]\d{2}', text)):  # Not date
                    
                    # Common OCR corrections for this specific case
                    if text.upper() == 'IRET':
                        text = 'TRET'
                    
                    location_texts.append(text)
            
            # Combine location texts if multiple parts detected
            if location_texts:
                combined_location = ' '.join(location_texts)
                locations.append(combined_location)
        
        cap.release()
        
        # Process results
        video_date = None
        video_time = None
        location = None
        confidence = 0.0
        
        # Process timestamps (dates)
        if timestamps:
            # Use most common timestamp
            timestamp_text = max(set(timestamps), key=timestamps.count)
            try:
                # Try different date formats
                for fmt in ['%d-%m-%Y', '%m-%d-%Y', '%Y-%m-%d']:
                    try:
                        date_part = re.search(r'\d{2}[-/]\d{2}[-/]\d{4}', timestamp_text).group()
                        date_part = date_part.replace('/', '-')
                        video_date = datetime.strptime(date_part, fmt).date()
                        break
                    except:
                        continue
            except:
                pass
        
        # Process times
        if times:
            # Use most common time
            time_text = max(set(times), key=times.count)
            try:
                # Clean up time text (remove extra spaces)
                time_text = time_text.replace(' ', '')
                
                # Try different time formats
                if ':' in time_text:
                    parts = time_text.split(':')
                    if len(parts) == 3:  # HH:MM:SS
                        hour, minute, second = parts
                        video_time = datetime.strptime(f"{hour.zfill(2)}:{minute.zfill(2)}:{second.zfill(2)}", '%H:%M:%S').time()
                    elif len(parts) == 2:  # HH:MM
                        hour, minute = parts
                        video_time = datetime.strptime(f"{hour.zfill(2)}:{minute.zfill(2)}", '%H:%M').time()
            except Exception as e:
                print(f"Time parsing error: {e} for '{time_text}'")
        
        # Process locations
        if locations:
            # Use most common location
            location = max(set(locations), key=locations.count)
            confidence = 0.8  # Basic confidence
        
        # Update database
        video.ocr_location = location
        video.ocr_video_date = video_date
        video.ocr_video_time = video_time
        video.ocr_extraction_done = True
        video.ocr_extraction_confidence = confidence
        
        # Update person detections
        detections = DetectedPerson.query.filter_by(video_id=video.id).all()
        updated_count = 0
        
        for detection in detections:
            if not detection.attendance_location and location:
                detection.attendance_location = location
            
            if not detection.attendance_date and video_date:
                detection.attendance_date = video_date
                
                # Calculate attendance time from video timestamp + OCR time
                if detection.timestamp is not None:
                    if video_time:
                        # Combine OCR time with video offset
                        base_time = datetime.combine(datetime.today(), video_time)
                        time_offset = timedelta(seconds=float(detection.timestamp))
                        final_time = base_time + time_offset
                        detection.attendance_time = final_time.time()
                    else:
                        # Fallback to just video offset
                        time_in_video = timedelta(seconds=float(detection.timestamp))
                        detection.attendance_time = (datetime.min + time_in_video).time()
                
                updated_count += 1
        
        db.session.commit()
        
        # Create result message
        result_parts = []
        if location:
            result_parts.append(f"Location: {location}")
        if video_date:
            result_parts.append(f"Date: {video_date}")
        if video_time:
            result_parts.append(f"Time: {video_time}")
        result_parts.append(f"Updated {updated_count} detections")
        
        return True, f"OCR successful - {', '.join(result_parts)}"
        
    except Exception as e:
        return False, f"Error: {str(e)}"

def main():
    app = create_app()
    
    with app.app_context():
        # Get models
        Video = app.Video
        DetectedPerson = app.DetectedPerson
        db = app.db
        
        # Make available globally
        globals()['Video'] = Video
        globals()['DetectedPerson'] = DetectedPerson
        globals()['db'] = db
        
        # Find videos needing OCR
        videos = Video.query.filter(
            (Video.ocr_extraction_done == False) | (Video.ocr_extraction_done == None),
            Video.status == 'completed'
        ).all()
        
        if not videos:
            print("All videos already have OCR data")
            return
        
        print(f"Processing {len(videos)} videos...")
        
        for video in videos:
            print(f"\nProcessing: {video.filename}")
            success, message = extract_ocr_simple(video.id)
            if success:
                print(f"  SUCCESS: {message}")
            else:
                print(f"  FAILED: {message}")

if __name__ == '__main__':
    print("Simple OCR Extraction for Videos")
    print("=" * 40)
    main()
    print("\nDone!")