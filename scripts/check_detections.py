#!/usr/bin/env python3
"""
Check DetectedPerson records and their fields
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_detections():
    """Check what DetectedPerson records we have"""
    # Import here to avoid module errors
    try:
        from app import create_app
        app = create_app()
    except Exception as e:
        print(f"Error creating app: {e}")
        return
    
    with app.app_context():
        Video = app.Video
        DetectedPerson = app.DetectedPerson
        db = app.db
        
        print("="*80)
        print("CHECKING DETECTEDPERSON RECORDS")
        print("="*80)
        
        # 1. Count total detections
        total_detections = DetectedPerson.query.count()
        print(f"\nTotal DetectedPerson records: {total_detections}")
        
        if total_detections == 0:
            print("\n⚠️  No DetectedPerson records found!")
            print("This is why attendance reports are empty.")
            print("\nTo fix this:")
            print("1. Make sure videos have been processed for person detection")
            print("2. Go to Videos page and click 'Process' on each video")
            return
        
        # 2. Check sample records
        print("\nSample DetectedPerson records:")
        samples = DetectedPerson.query.limit(5).all()
        
        for i, det in enumerate(samples, 1):
            print(f"\n--- Record {i} ---")
            print(f"ID: {det.id}")
            print(f"Video ID: {det.video_id}")
            
            # Check what fields exist
            fields = ['person_id', 'person_code', 'timestamp', 'start_time', 'end_time', 
                     'confidence', 'attendance_date', 'attendance_time', 'attendance_location']
            
            for field in fields:
                if hasattr(det, field):
                    value = getattr(det, field)
                    print(f"{field}: {value}")
                else:
                    print(f"{field}: [FIELD DOES NOT EXIST]")
        
        # 3. Check videos with detections
        print("\n\nVideos with detections:")
        videos_with_detections = db.session.query(
            Video.id, 
            Video.filename,
            Video.ocr_extraction_done,
            Video.ocr_video_date,
            Video.ocr_location,
            db.func.count(DetectedPerson.id).label('detection_count')
        ).join(DetectedPerson).group_by(Video.id).limit(5).all()
        
        for video in videos_with_detections:
            print(f"\nVideo {video.id}: {video.filename}")
            print(f"  OCR Done: {video.ocr_extraction_done}")
            print(f"  OCR Date: {video.ocr_video_date}")
            print(f"  OCR Location: {video.ocr_location}")
            print(f"  Detection Count: {video.detection_count}")
        
        # 4. Check if any records have person_id
        with_person_id = DetectedPerson.query.filter(
            DetectedPerson.person_id.isnot(None)
        ).count()
        print(f"\n\nDetections with person_id: {with_person_id}/{total_detections}")
        
        # 5. Check unique person_ids
        if with_person_id > 0:
            unique_persons = db.session.query(
                DetectedPerson.person_id
            ).filter(
                DetectedPerson.person_id.isnot(None)
            ).distinct().all()
            print(f"Unique person IDs: {len(unique_persons)}")
            for person in unique_persons[:10]:  # Show first 10
                print(f"  - {person[0]}")

if __name__ == '__main__':
    check_detections()