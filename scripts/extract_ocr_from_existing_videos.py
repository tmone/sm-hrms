#!/usr/bin/env python3
"""
Extract OCR data from existing processed videos without affecting person detections
This script will:
1. Find videos that don't have OCR data extracted
2. Extract timestamp and location using OCR
3. Update video records with OCR data
4. Update existing person detections with attendance data
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from hr_management.processing.ocr_extractor import VideoOCRExtractor

def extract_ocr_for_video(video_id):
    """Extract OCR data for a single video"""
    video = Video.query.get(video_id)
    if not video:
        print(f"[ERROR] Video {video_id} not found")
        return False
    
    print(f"\nProcessing video: {video.filename}")
    
    # Check if OCR already extracted
    if video.ocr_extraction_done:
        print(f"   OCR already extracted - Location: {video.ocr_location}, Date: {video.ocr_video_date}")
        return True
    
    # Get the video file path
    video_path = None
    
    # Try the stored file_path first
    if hasattr(video, 'file_path') and video.file_path and os.path.exists(video.file_path):
        video_path = video.file_path
    else:
        # Search for the video file in uploads directory
        uploads_dir = Path('static/uploads')
        if uploads_dir.exists():
            # Look for files that contain the video filename (without extension)
            base_name = Path(video.filename).stem
            
            # Try exact match first
            exact_match = uploads_dir / video.filename
            if exact_match.exists():
                video_path = str(exact_match)
            else:
                # Search for files containing the base filename
                for video_file in uploads_dir.glob('*.mp4'):
                    if base_name in video_file.name:
                        # Prefer original files over annotated ones
                        if 'annotated' not in video_file.name:
                            video_path = str(video_file)
                            break
                
                # If no original found, use any matching file (including annotated)
                if not video_path:
                    for video_file in uploads_dir.glob('*.mp4'):
                        if base_name in video_file.name:
                            video_path = str(video_file)
                            break
    
    if not video_path:
        print(f"   ERROR: Video file not found")
        return False
    
    try:
        # Initialize OCR extractor
        print("   Extracting OCR data...")
        
        # Suppress EasyOCR verbose output
        import sys
        from contextlib import redirect_stdout, redirect_stderr
        
        with open(os.devnull, 'w') as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                ocr_extractor = VideoOCRExtractor(ocr_engine='easyocr')
                
                # Sample every 10 seconds for OCR
                sample_interval = 300  # 10 seconds at 30fps
                ocr_data = ocr_extractor.extract_video_info(video_path, sample_interval=sample_interval)
        
        if ocr_data:
            # Update video record
            video.ocr_location = ocr_data.get('location')
            video.ocr_video_date = ocr_data.get('video_date')
            video.ocr_extraction_done = True
            video.ocr_extraction_confidence = ocr_data.get('confidence', 0.0)
            
            print(f"   SUCCESS: OCR extraction complete:")
            print(f"      - Location: {video.ocr_location}")
            print(f"      - Video Date: {video.ocr_video_date}")
            print(f"      - Confidence: {ocr_data.get('confidence', 0):.2%}")
            
            # Update existing person detections with attendance data
            detections = DetectedPerson.query.filter_by(video_id=video.id).all()
            updated_count = 0
            
            for detection in detections:
                # Only update if attendance data is not already set
                if not detection.attendance_location:
                    detection.attendance_location = video.ocr_location
                    
                if not detection.attendance_date and video.ocr_video_date:
                    detection.attendance_date = video.ocr_video_date
                    
                    # Calculate attendance time from video timestamp
                    if detection.timestamp is not None:
                        time_in_video = timedelta(seconds=float(detection.timestamp))
                        detection.attendance_time = (datetime.min + time_in_video).time()
                    
                    updated_count += 1
            
            db.session.commit()
            print(f"   SUCCESS: Updated {updated_count} person detections with attendance data")
            
            return True
        else:
            print("   WARNING: No OCR data could be extracted")
            # Still mark as attempted
            video.ocr_extraction_done = True
            video.ocr_extraction_confidence = 0.0
            db.session.commit()
            return False
            
    except Exception as e:
        print(f"   ERROR: OCR extraction failed: {e}")
        db.session.rollback()
        return False

def main():
    """Main function to process all videos"""
    app = create_app()
    
    with app.app_context():
        # Get models from app
        Video = app.Video
        DetectedPerson = app.DetectedPerson
        db = app.db
        
        # Make models available globally for other functions
        globals()['Video'] = Video
        globals()['DetectedPerson'] = DetectedPerson
        globals()['db'] = db
        
        # Find videos that need OCR extraction
        videos_to_process = Video.query.filter(
            (Video.ocr_extraction_done == False) | (Video.ocr_extraction_done == None),
            Video.status == 'completed'  # Only process completed videos
        ).all()
        
        if not videos_to_process:
            print("All videos already have OCR data extracted")
            return
        
        print(f"Found {len(videos_to_process)} videos to process")
        
        success_count = 0
        failed_count = 0
        
        for video in videos_to_process:
            if extract_ocr_for_video(video.id):
                success_count += 1
            else:
                failed_count += 1
        
        print(f"\nSummary:")
        print(f"   Successfully processed: {success_count}")
        print(f"   Failed: {failed_count}")
        print(f"   Total: {len(videos_to_process)}")

if __name__ == '__main__':
    print("OCR Extraction Script for Existing Videos")
    print("=" * 50)
    
    # Check if OCR dependencies are available
    try:
        import easyocr
        print("EasyOCR is available")
    except ImportError:
        print("ERROR: EasyOCR not found. Please install: pip install easyocr")
        sys.exit(1)
    
    main()