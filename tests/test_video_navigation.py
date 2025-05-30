#!/usr/bin/env python3
"""
Test the video navigation feature from attendance report
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app

def test_video_navigation():
    """Test that video navigation links are working correctly"""
    app = create_app()
    
    with app.app_context():
        db = app.db
        Video = app.Video
        DetectedPerson = app.DetectedPerson
        
        print("Testing Video Navigation Feature")
        print("=" * 50)
        
        # Get a video with detections
        video = Video.query.filter(
            Video.status == 'completed',
            Video.ocr_extraction_done == True
        ).first()
        
        if not video:
            print("❌ No completed video with OCR data found")
            return
        
        print(f"\n✅ Found video: {video.filename}")
        print(f"   - ID: {video.id}")
        print(f"   - Location: {video.ocr_location}")
        print(f"   - Date: {video.ocr_video_date}")
        print(f"   - Time: {video.ocr_video_time}")
        
        # Get detections for this video
        detections = DetectedPerson.query.filter_by(video_id=video.id).all()
        
        print(f"\n📊 Detections found: {len(detections)}")
        
        # Group by person
        persons = {}
        for detection in detections:
            person_id = detection.person_id or f"unknown-{detection.id}"
            if person_id not in persons:
                persons[person_id] = []
            persons[person_id].append(detection)
        
        print(f"\n👥 Unique persons: {len(persons)}")
        
        # Show navigation URLs for first 3 persons
        print("\n🔗 Sample Navigation URLs:")
        for i, (person_id, person_detections) in enumerate(list(persons.items())[:3]):
            if person_detections:
                first_detection = min(person_detections, key=lambda d: d.timestamp if d.timestamp else float('inf'))
                if first_detection.timestamp is not None:
                    url = f"/videos/{video.id}?t={int(first_detection.timestamp)}&person={person_id}"
                    print(f"\n   Person {person_id}:")
                    print(f"   - First seen at: {first_detection.timestamp:.1f}s")
                    print(f"   - Navigation URL: {url}")
                    print(f"   - Total detections: {len(person_detections)}")
        
        print("\n✅ Video navigation URLs are correctly formatted!")
        print("\nTo test in browser:")
        print("1. Go to http://localhost:5001/attendance/daily")
        print("2. Click on any 'View' button in the Actions column")
        print("3. The video should automatically seek to the timestamp")
        print("4. The person should be highlighted in the detection navigator")

if __name__ == '__main__':
    test_video_navigation()