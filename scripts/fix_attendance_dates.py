#!/usr/bin/env python3
"""
Fix attendance dates that were parsed incorrectly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from datetime import datetime

def fix_attendance_dates():
    """Fix incorrectly parsed attendance dates"""
    app = create_app()
    
    with app.app_context():
        db = app.db
        Video = app.Video
        DetectedPerson = app.DetectedPerson
        
        print("="*80)
        print("FIXING ATTENDANCE DATES")
        print("="*80)
        
        # Get all videos with OCR data
        videos_with_ocr = Video.query.filter(
            Video.ocr_extraction_done == True,
            Video.ocr_video_date.isnot(None)
        ).all()
        
        fixed_count = 0
        
        for video in videos_with_ocr:
            print(f"\nProcessing video {video.id}: {video.filename}")
            print(f"  Video OCR date: {video.ocr_video_date}")
            
            # Get all detections for this video
            detections = DetectedPerson.query.filter_by(video_id=video.id).all()
            
            if detections:
                # Check first detection's date
                first_detection = detections[0]
                print(f"  First detection date: {first_detection.attendance_date}")
                
                # If dates don't match, fix them
                if first_detection.attendance_date != video.ocr_video_date:
                    print(f"  ⚠️ Date mismatch detected!")
                    print(f"  Fixing {len(detections)} detections...")
                    
                    for detection in detections:
                        # Update attendance date to match video OCR date
                        detection.attendance_date = video.ocr_video_date
                        
                        # Recalculate attendance time if needed
                        if video.ocr_video_time and detection.timestamp is not None:
                            from datetime import timedelta
                            base_datetime = datetime.combine(video.ocr_video_date, video.ocr_video_time)
                            detection_offset = timedelta(seconds=float(detection.timestamp))
                            attendance_datetime = base_datetime + detection_offset
                            detection.attendance_time = attendance_datetime.time()
                            detection.check_in_time = attendance_datetime
                        
                        fixed_count += 1
                    
                    db.session.commit()
                    print(f"  ✅ Fixed {len(detections)} detections")
                else:
                    print(f"  ✅ Dates match correctly")
        
        print(f"\n{'='*80}")
        print(f"Total detections fixed: {fixed_count}")
        print("="*80)
        
        # Verify fix
        print("\nVerifying fix...")
        sample_detections = DetectedPerson.query.limit(5).all()
        for detection in sample_detections:
            video = Video.query.get(detection.video_id)
            print(f"Detection {detection.id}: Video date={video.ocr_video_date}, Attendance date={detection.attendance_date}")

if __name__ == "__main__":
    fix_attendance_dates()