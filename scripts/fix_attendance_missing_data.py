#!/usr/bin/env python3
"""
Fix missing attendance data by:
1. Running the migration to add ocr_video_time field
2. Re-extracting OCR data from existing videos
3. Updating detections with attendance information
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from datetime import datetime, timedelta
import subprocess

def run_migration():
    """Run the migration to add ocr_video_time field"""
    print("="*80)
    print("STEP 1: Running database migration")
    print("="*80)
    
    try:
        # Run the migration script
        result = subprocess.run([sys.executable, 'scripts/migrate_add_ocr_video_time.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Migration completed successfully")
            print(result.stdout)
        else:
            print("✗ Migration failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"✗ Error running migration: {e}")
        return False
    
    return True

def extract_and_save_ocr_data(app):
    """Extract OCR data from videos and save to database"""
    print("\n" + "="*80)
    print("STEP 2: Extracting OCR data from videos")
    print("="*80)
    
    with app.app_context():
        db = app.db
        Video = app.Video
        
        # Get all completed videos without OCR data
        videos = Video.query.filter(
            Video.status == 'completed',
            Video.ocr_extraction_done == False
        ).all()
        
        print(f"Found {len(videos)} videos needing OCR extraction")
        
        if not videos:
            print("No videos need OCR extraction")
            return True
        
        try:
            from hr_management.processing.ocr_extractor import VideoOCRExtractor
            ocr_extractor = VideoOCRExtractor(ocr_engine='easyocr')
            
            for video in videos:
                print(f"\n[{video.id}] Processing: {video.filename}")
                
                # Find video file
                video_path = None
                from pathlib import Path
                
                uploads_dir = Path('static/uploads')
                if video.file_path:
                    # Try the stored path first
                    if os.path.exists(os.path.join(uploads_dir, video.file_path)):
                        video_path = os.path.join(uploads_dir, video.file_path)
                    elif os.path.exists(video.file_path):
                        video_path = video.file_path
                
                if not video_path and uploads_dir.exists():
                    # Search for the video file
                    base_name = Path(video.filename).stem
                    for vfile in uploads_dir.glob('*.mp4'):
                        if base_name in vfile.name and 'annotated' not in vfile.name:
                            video_path = str(vfile)
                            break
                
                if not video_path:
                    print(f"  ✗ Video file not found")
                    continue
                
                # Extract OCR data
                try:
                    ocr_results = ocr_extractor.extract_video_info(video_path, sample_interval=300)
                    
                    if ocr_results:
                        # Update video with OCR data
                        video.ocr_location = ocr_results.get('location')
                        video.ocr_video_date = ocr_results.get('video_date')
                        
                        # Extract time from the timestamps
                        if ocr_results.get('timestamps'):
                            # Get the first valid timestamp
                            first_timestamp = ocr_results['timestamps'][0]['timestamp']
                            video.ocr_video_time = first_timestamp.time()
                        
                        video.ocr_extraction_done = True
                        video.ocr_extraction_confidence = ocr_results.get('extraction_summary', {}).get('confidence', 0)
                        
                        db.session.commit()
                        
                        print(f"  ✓ OCR extracted:")
                        print(f"    - Location: {video.ocr_location}")
                        print(f"    - Date: {video.ocr_video_date}")
                        print(f"    - Time: {video.ocr_video_time}")
                        print(f"    - Confidence: {video.ocr_extraction_confidence:.2%}")
                    else:
                        print(f"  ✗ No OCR data could be extracted")
                        
                except Exception as e:
                    print(f"  ✗ OCR extraction error: {e}")
                    continue
                    
        except ImportError:
            print("✗ OCR extractor module not available")
            print("Please ensure easyocr is installed: pip install easyocr")
            return False
        except Exception as e:
            print(f"✗ Error during OCR extraction: {e}")
            db.session.rollback()
            return False
    
    return True

def update_detection_attendance_data(app):
    """Update detections with attendance data based on OCR"""
    print("\n" + "="*80)
    print("STEP 3: Updating detections with attendance data")
    print("="*80)
    
    with app.app_context():
        db = app.db
        Video = app.Video
        DetectedPerson = app.DetectedPerson
        
        # Get videos with OCR data
        videos_with_ocr = Video.query.filter(
            Video.ocr_extraction_done == True,
            Video.ocr_video_date.isnot(None)
        ).all()
        
        print(f"Found {len(videos_with_ocr)} videos with OCR data")
        
        updated_count = 0
        
        for video in videos_with_ocr:
            # Get all detections for this video without attendance data
            detections = DetectedPerson.query.filter(
                DetectedPerson.video_id == video.id,
                DetectedPerson.attendance_date.is_(None)
            ).all()
            
            if not detections:
                continue
                
            print(f"\n[{video.id}] Updating {len(detections)} detections for: {video.filename}")
            
            for detection in detections:
                # Update attendance fields
                detection.attendance_location = video.ocr_location
                detection.attendance_date = video.ocr_video_date
                
                # Calculate attendance time based on detection timestamp
                if detection.timestamp is not None and video.ocr_video_time:
                    # Combine date and time
                    base_datetime = datetime.combine(video.ocr_video_date, video.ocr_video_time)
                    detection_offset = timedelta(seconds=float(detection.timestamp))
                    attendance_datetime = base_datetime + detection_offset
                    
                    detection.attendance_time = attendance_datetime.time()
                    detection.check_in_time = attendance_datetime
                
                updated_count += 1
            
            try:
                db.session.commit()
                print(f"  ✓ Updated {len(detections)} detections")
            except Exception as e:
                print(f"  ✗ Error updating detections: {e}")
                db.session.rollback()
        
        print(f"\n✓ Total detections updated: {updated_count}")
    
    return True

def verify_attendance_data(app):
    """Verify that attendance data is now available"""
    print("\n" + "="*80)
    print("STEP 4: Verifying attendance data")
    print("="*80)
    
    with app.app_context():
        db = app.db
        Video = app.Video
        DetectedPerson = app.DetectedPerson
        
        # Count videos with OCR data
        ocr_videos = Video.query.filter(
            Video.ocr_extraction_done == True
        ).count()
        
        # Count detections with attendance data
        attendance_detections = DetectedPerson.query.filter(
            DetectedPerson.attendance_date.isnot(None)
        ).count()
        
        # Get unique dates with attendance
        unique_dates = db.session.query(
            DetectedPerson.attendance_date
        ).filter(
            DetectedPerson.attendance_date.isnot(None)
        ).distinct().count()
        
        # Get unique locations
        unique_locations = db.session.query(
            Video.ocr_location
        ).filter(
            Video.ocr_location.isnot(None)
        ).distinct().all()
        
        print(f"✓ Videos with OCR data: {ocr_videos}")
        print(f"✓ Detections with attendance data: {attendance_detections}")
        print(f"✓ Unique attendance dates: {unique_dates}")
        print(f"✓ Locations found: {[loc[0] for loc in unique_locations]}")
        
        if attendance_detections > 0:
            print("\n✅ Attendance data is now available! You should see data in the attendance report page.")
        else:
            print("\n⚠️  No attendance data found. Please check that:")
            print("   1. Videos have been processed successfully")
            print("   2. Videos contain OCR text (timestamp and location)")
            print("   3. OCR extraction is working properly")

def main():
    """Main function to fix missing attendance data"""
    print("FIXING MISSING ATTENDANCE DATA")
    print("This script will:")
    print("1. Run database migration to add missing fields")
    print("2. Extract OCR data from existing videos")
    print("3. Update detections with attendance information")
    print("4. Verify the data is available")
    print("")
    
    # Create app context
    app = create_app()
    
    # Step 1: Run migration
    if not run_migration():
        print("\n❌ Migration failed. Please fix the issue and try again.")
        return
    
    # Step 2: Extract OCR data
    if not extract_and_save_ocr_data(app):
        print("\n❌ OCR extraction failed. Please fix the issue and try again.")
        return
    
    # Step 3: Update detections
    if not update_detection_attendance_data(app):
        print("\n❌ Detection update failed. Please fix the issue and try again.")
        return
    
    # Step 4: Verify
    verify_attendance_data(app)
    
    print("\n" + "="*80)
    print("✅ Process completed! Check the attendance report page now.")
    print("="*80)

if __name__ == "__main__":
    main()