#!/usr/bin/env python3
"""
Test attendance query logic
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from datetime import datetime, date, timedelta
from sqlalchemy import func

def test_attendance_query():
    """Test the attendance query logic"""
    app = create_app()
    
    with app.app_context():
        db = app.db
        Video = app.Video
        DetectedPerson = app.DetectedPerson
        
        print("="*80)
        print("TESTING ATTENDANCE QUERY")
        print("="*80)
        
        # Test the same query used in attendance blueprint
        print("\n1. Testing attendance summary query:")
        
        # Get videos with OCR data
        total_videos = Video.query.filter(
            Video.ocr_extraction_done == True
        ).count()
        print(f"   Total videos with OCR: {total_videos}")
        
        # Get unique locations
        locations_query = db.session.query(
            Video.ocr_location,
            func.count(Video.id).label('count')
        ).filter(
            Video.ocr_location.isnot(None)
        ).group_by(Video.ocr_location).all()
        
        print(f"   Locations found: {locations_query}")
        
        # Test daily report query for specific date
        print("\n2. Testing daily report query:")
        
        # Check available dates
        dates_query = db.session.query(
            func.distinct(DetectedPerson.attendance_date)
        ).filter(
            DetectedPerson.attendance_date.isnot(None)
        ).all()
        
        print(f"   Available attendance dates: {[d[0] for d in dates_query]}")
        
        # Test for May 12, 2025 (the fixed date)
        report_date = date(2025, 5, 12)
        print(f"\n3. Testing query for date: {report_date}")
        
        # Query videos for this date
        videos_query = Video.query.filter(
            Video.ocr_video_date == report_date,
            Video.ocr_extraction_done == True
        ).all()
        
        print(f"   Videos found for {report_date}: {len(videos_query)}")
        
        if videos_query:
            for video in videos_query:
                print(f"   - Video {video.id}: {video.filename}")
                print(f"     OCR Date: {video.ocr_video_date}")
                print(f"     OCR Time: {video.ocr_video_time}")
                print(f"     OCR Location: {video.ocr_location}")
                
                # Get detections for this video
                detections = DetectedPerson.query.filter_by(video_id=video.id).limit(5).all()
                print(f"     Sample detections ({len(detections)} shown):")
                for det in detections:
                    print(f"       - Person {det.person_id}: Date={det.attendance_date}, Time={det.attendance_time}")
        
        # Test person count query
        print("\n4. Testing person count query:")
        
        # Count unique persons for the date
        person_count = db.session.query(
            func.count(func.distinct(DetectedPerson.person_id))
        ).filter(
            DetectedPerson.attendance_date == report_date
        ).scalar() or 0
        
        print(f"   Unique persons on {report_date}: {person_count}")
        
        # Get actual person IDs
        person_ids = db.session.query(
            func.distinct(DetectedPerson.person_id)
        ).filter(
            DetectedPerson.attendance_date == report_date
        ).limit(10).all()
        
        print(f"   Sample person IDs: {[p[0] for p in person_ids]}")

if __name__ == "__main__":
    test_attendance_query()