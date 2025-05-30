#!/usr/bin/env python3
"""
Comprehensive diagnostic for attendance system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set environment to use correct database
os.environ['DATABASE_URL'] = f'sqlite:///{os.path.join(os.path.dirname(os.path.abspath(__file__)), "instance", "stepmedia_hrm.db")}'

print("="*80)
print("ATTENDANCE SYSTEM DIAGNOSTIC")
print("="*80)

try:
    from app import create_app
    from datetime import datetime, date
    
    app = create_app()
    
    with app.app_context():
        Video = app.Video
        DetectedPerson = app.DetectedPerson
        db = app.db
        
        print("\n1. DATABASE CONNECTION")
        print("-" * 40)
        print(f"Database URI: {app.config['SQLALCHEMY_DATABASE_URI']}")
        
        # Check videos with OCR
        print("\n2. VIDEOS WITH OCR DATA")
        print("-" * 40)
        videos_with_ocr = Video.query.filter(
            Video.ocr_extraction_done == True
        ).all()
        
        print(f"Total videos with OCR: {len(videos_with_ocr)}")
        
        for video in videos_with_ocr:
            print(f"\nVideo ID {video.id}: {video.filename}")
            print(f"  - OCR Date: {video.ocr_video_date}")
            print(f"  - OCR Time: {video.ocr_video_time}")
            print(f"  - OCR Location: {video.ocr_location}")
            print(f"  - Status: {video.status}")
            
            # Count detections for this video
            detection_count = DetectedPerson.query.filter_by(video_id=video.id).count()
            print(f"  - Detections: {detection_count}")
        
        # Check detections
        print("\n3. DETECTION RECORDS")
        print("-" * 40)
        total_detections = DetectedPerson.query.count()
        print(f"Total DetectedPerson records: {total_detections}")
        
        # Check attendance fields
        with_attendance_date = DetectedPerson.query.filter(
            DetectedPerson.attendance_date.isnot(None)
        ).count()
        print(f"Detections with attendance_date: {with_attendance_date}")
        
        # Sample detection
        sample = DetectedPerson.query.first()
        if sample:
            print(f"\nSample detection fields:")
            for field in ['id', 'video_id', 'person_id', 'timestamp', 'attendance_date', 
                         'attendance_time', 'attendance_location']:
                if hasattr(sample, field):
                    print(f"  - {field}: {getattr(sample, field)}")
        
        # Test attendance logic
        print("\n4. TESTING ATTENDANCE REPORT LOGIC")
        print("-" * 40)
        
        # Use today's date or the OCR date
        if videos_with_ocr:
            test_date = videos_with_ocr[0].ocr_video_date
        else:
            test_date = date.today()
        
        print(f"Testing with date: {test_date}")
        
        # Query videos for this date
        videos_for_date = Video.query.filter(
            Video.ocr_video_date == test_date,
            Video.ocr_extraction_done == True
        ).all()
        
        print(f"Videos for this date: {len(videos_for_date)}")
        
        attendance_data = []
        
        for video in videos_for_date:
            print(f"\n  Processing video: {video.filename}")
            detections = DetectedPerson.query.filter_by(video_id=video.id).all()
            print(f"  Found {len(detections)} detections")
            
            # Group by person
            person_groups = {}
            for det in detections:
                pid = det.person_id or f"unknown-{det.id}"
                if pid not in person_groups:
                    person_groups[pid] = []
                person_groups[pid].append(det)
            
            print(f"  Unique persons: {len(person_groups)}")
            
            # Calculate attendance for each person
            for person_id, dets in person_groups.items():
                timestamps = [d.timestamp for d in dets if d.timestamp is not None]
                if timestamps:
                    first_ts = min(timestamps)
                    last_ts = max(timestamps)
                    
                    # Calculate clock times
                    if video.ocr_video_time:
                        base_dt = datetime.combine(video.ocr_video_date, video.ocr_video_time)
                        clock_in = (base_dt.timestamp() + first_ts)
                        clock_out = (base_dt.timestamp() + last_ts)
                        clock_in_time = datetime.fromtimestamp(clock_in).strftime("%H:%M:%S")
                        clock_out_time = datetime.fromtimestamp(clock_out).strftime("%H:%M:%S")
                    else:
                        clock_in_time = f"{int(first_ts//60)}:{int(first_ts%60):02d}"
                        clock_out_time = f"{int(last_ts//60)}:{int(last_ts%60):02d}"
                    
                    duration = last_ts - first_ts
                    
                    print(f"    Person {person_id}: {clock_in_time} to {clock_out_time} ({duration:.1f}s)")
                    
                    attendance_data.append({
                        'person_id': person_id,
                        'location': video.ocr_location,
                        'date': video.ocr_video_date,
                        'clock_in': clock_in_time,
                        'clock_out': clock_out_time,
                        'duration_seconds': duration
                    })
        
        # Test the attendance endpoints
        print("\n5. TESTING ATTENDANCE ENDPOINTS")
        print("-" * 40)
        
        with app.test_client() as client:
            # Test summary endpoint
            print("\nTesting /attendance/summary")
            response = client.get('/attendance/summary?days=7')
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json
                print(f"Response keys: {list(data.keys())}")
                print(f"Total videos: {data.get('total_videos', 0)}")
                print(f"Locations: {data.get('locations', {})}")
            
            # Test daily endpoint
            print("\nTesting /attendance/daily?format=json")
            response = client.get(f'/attendance/daily?format=json&date={test_date}')
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json
                print(f"Response keys: {list(data.keys())}")
                print(f"Attendance records: {len(data.get('attendance_data', []))}")
                
                # Show first record
                if data.get('attendance_data'):
                    first = data['attendance_data'][0]
                    print(f"\nFirst record:")
                    for key in ['person_id', 'location', 'clock_in', 'clock_out', 'duration_seconds']:
                        print(f"  - {key}: {first.get(key)}")
        
        print("\n6. HOW TO ACCESS ATTENDANCE DATA")
        print("-" * 40)
        print("\n1. Start the Flask server:")
        print("   cd /mnt/d/sm-hrm")
        print("   python app.py")
        print("\n2. Navigate to:")
        print("   http://localhost:5001/attendance/")
        print("\n3. If redirected to login:")
        print("   - Email: admin@stepmedia.com")
        print("   - Password: (leave blank)")
        print("\n4. Direct API access (no login):")
        print(f"   http://localhost:5001/attendance/test")
        
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n\nTrying direct database check...")
    import sqlite3
    
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "instance", "stepmedia_hrm.db")
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check videos
        cursor.execute("SELECT COUNT(*) FROM videos WHERE ocr_extraction_done = 1")
        count = cursor.fetchone()[0]
        print(f"\nVideos with OCR: {count}")
        
        # Check detections
        cursor.execute("SELECT COUNT(*) FROM detected_persons")
        count = cursor.fetchone()[0]
        print(f"Total detections: {count}")
        
        conn.close()