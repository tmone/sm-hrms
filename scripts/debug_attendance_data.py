#!/usr/bin/env python3
"""
Debug attendance data to see why records aren't showing
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from datetime import datetime

def debug_attendance_data():
    """Debug attendance data flow"""
    app = create_app()
    
    with app.app_context():
        db = app.db
        Video = app.Video
        DetectedPerson = app.DetectedPerson
        
        print("="*80)
        print("DEBUGGING ATTENDANCE DATA")
        print("="*80)
        
        # 1. Check videos with OCR data
        print("\n1. Videos with OCR data:")
        videos_with_ocr = Video.query.filter(
            Video.ocr_extraction_done == True,
            Video.ocr_video_date.isnot(None)
        ).all()
        
        print(f"   Found {len(videos_with_ocr)} videos with OCR data")
        for video in videos_with_ocr[:5]:  # Show first 5
            print(f"   - Video {video.id}: {video.filename}")
            print(f"     Date: {video.ocr_video_date}, Time: {video.ocr_video_time}, Location: {video.ocr_location}")
        
        # 2. Check DetectedPerson records
        print("\n2. DetectedPerson records:")
        total_detections = DetectedPerson.query.count()
        print(f"   Total detections: {total_detections}")
        
        # Check if attendance fields are populated
        with_attendance = DetectedPerson.query.filter(
            DetectedPerson.attendance_date.isnot(None)
        ).count()
        print(f"   Detections with attendance_date: {with_attendance}")
        
        # 3. Check sample DetectedPerson records
        print("\n3. Sample DetectedPerson records:")
        sample_detections = DetectedPerson.query.limit(5).all()
        for det in sample_detections:
            print(f"   - Detection {det.id}:")
            print(f"     Video ID: {det.video_id}")
            print(f"     Person ID: {det.person_id}")
            print(f"     Timestamp: {det.timestamp}")
            print(f"     Attendance Date: {det.attendance_date}")
            print(f"     Attendance Time: {det.attendance_time}")
            print(f"     Attendance Location: {det.attendance_location}")
        
        # 4. Detailed analysis of attendance data
        print("\n4. Attendance Data Analysis:")
        
        # Check videos without OCR data
        videos_without_ocr = Video.query.filter(
            db.or_(
                Video.ocr_extraction_done == False,
                Video.ocr_video_date.is_(None)
            )
        ).count()
        print(f"   Videos without OCR data: {videos_without_ocr}")
        
        # Check detections with partial attendance data
        with_time = DetectedPerson.query.filter(
            DetectedPerson.attendance_time.isnot(None)
        ).count()
        with_location = DetectedPerson.query.filter(
            DetectedPerson.attendance_location.isnot(None)
        ).count()
        
        print(f"   Detections with attendance_time: {with_time}")
        print(f"   Detections with attendance_location: {with_location}")
        
        # Show summary
        print("\n5. Summary:")
        print(f"   Total videos: {Video.query.count()}")
        print(f"   Videos with OCR: {len(videos_with_ocr)} ({len(videos_with_ocr)/Video.query.count()*100:.1f}%)")
        print(f"   Total detections: {total_detections}")
        print(f"   Detections with attendance data: {with_attendance} ({with_attendance/total_detections*100:.1f}%)" if total_detections > 0 else "   No detections found")
        
        # Check if we need to populate attendance fields
        if with_attendance == 0 and len(videos_with_ocr) > 0:
            print("\n[WARNING]  ISSUE FOUND: Videos have OCR data but DetectedPerson records don't have attendance fields populated!")
            print("\nThis is why attendance reports are empty.")
            print("\nTo fix this, we need to populate the attendance fields in DetectedPerson records")
            print("based on the OCR data from their associated videos.")
            
            # Show what needs to be done
            print("\n6. What needs to be populated:")
            for video in videos_with_ocr[:3]:
                detections_count = DetectedPerson.query.filter_by(video_id=video.id).count()
                if detections_count > 0:
                    print(f"\n   Video: {video.filename}")
                    print(f"   OCR Date: {video.ocr_video_date}")
                    print(f"   OCR Time: {video.ocr_video_time}")
                    print(f"   OCR Location: {video.ocr_location}")
                    print(f"   Has {detections_count} detections that need attendance fields updated")
        elif with_attendance > 0:
            print("\n[OK] Attendance data is already populated for some detections.")
            print("   Attendance reports should be showing data.")

def populate_attendance_fields():
    """Populate attendance fields in DetectedPerson records from video OCR data"""
    app = create_app()
    
    with app.app_context():
        db = app.db
        Video = app.Video
        DetectedPerson = app.DetectedPerson
        
        print("\n" + "="*80)
        print("POPULATING ATTENDANCE FIELDS")
        print("="*80)
        
        # Get videos with OCR data
        videos_with_ocr = Video.query.filter(
            Video.ocr_extraction_done == True,
            Video.ocr_video_date.isnot(None)
        ).all()
        
        updated_count = 0
        
        for video in videos_with_ocr:
            print(f"\nProcessing video: {video.filename}")
            
            # Get all detections for this video
            detections = DetectedPerson.query.filter_by(video_id=video.id).all()
            
            if not detections:
                print(f"  No detections found")
                continue
            
            print(f"  Found {len(detections)} detections")
            print(f"  OCR Data - Date: {video.ocr_video_date}, Time: {video.ocr_video_time}, Location: {video.ocr_location}")
            
            # Update each detection with attendance info
            for detection in detections:
                # Set attendance date from video
                detection.attendance_date = video.ocr_video_date
                detection.attendance_location = video.ocr_location
                
                # Calculate attendance time from video OCR time + detection timestamp
                if video.ocr_video_time and detection.timestamp is not None:
                    # Create datetime from OCR base time
                    base_datetime = datetime.combine(video.ocr_video_date, video.ocr_video_time)
                    # Add detection timestamp (seconds from start)
                    actual_datetime = base_datetime.timestamp() + detection.timestamp
                    actual_datetime = datetime.fromtimestamp(actual_datetime)
                    detection.attendance_time = actual_datetime.time()
                    
                    # Set check-in time (using the detection timestamp)
                    check_in_timestamp = base_datetime.timestamp() + detection.timestamp
                    detection.check_in_time = datetime.fromtimestamp(check_in_timestamp)
                    
                    # Note: check_out_time would typically be set from a different detection
                    # or calculated differently based on business logic
                
                updated_count += 1
            
            print(f"  [CHECK] Updated {len(detections)} detections")
        
        # Commit all changes
        try:
            db.session.commit()
            print(f"\n[OK] Successfully updated {updated_count} detection records with attendance data!")
            print("\nAttendance reports should now show data.")
        except Exception as e:
            db.session.rollback()
            print(f"\n[ERROR] Error updating records: {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Debug and fix attendance data')
    parser.add_argument('--fix', action='store_true', help='Fix by populating attendance fields')
    
    args = parser.parse_args()
    
    # Always run debug first
    debug_attendance_data()
    
    if args.fix:
        response = input("\nDo you want to populate attendance fields? (yes/no): ")
        if response.lower() == 'yes':
            populate_attendance_fields()
        else:
            print("\nTo fix the issue, run: python scripts/debug_attendance_data.py --fix")

if __name__ == '__main__':
    main()