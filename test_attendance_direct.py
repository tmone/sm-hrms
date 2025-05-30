#!/usr/bin/env python3
"""
Direct test of attendance data without Flask dependencies
"""

import sqlite3
import os
from datetime import datetime, timedelta

db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "instance", "stepmedia_hrm.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("="*80)
print("ATTENDANCE DATA TEST - DIRECT DATABASE ACCESS")
print("="*80)

# 1. Check video with OCR
print("\n1. VIDEO WITH OCR DATA:")
cursor.execute("""
    SELECT id, filename, ocr_video_date, ocr_video_time, ocr_location, status
    FROM videos 
    WHERE ocr_extraction_done = 1
""")
video = cursor.fetchone()

if video:
    vid_id, filename, ocr_date, ocr_time, location, status = video
    print(f"Video ID: {vid_id}")
    print(f"Filename: {filename}")
    print(f"OCR Date: {ocr_date}")
    print(f"OCR Time: {ocr_time}")
    print(f"Location: {location}")
    print(f"Status: {status}")
    
    # 2. Get detections grouped by person
    print(f"\n2. PERSON DETECTIONS IN THIS VIDEO:")
    cursor.execute("""
        SELECT person_id, 
               COUNT(*) as detection_count,
               MIN(timestamp) as first_seen_seconds,
               MAX(timestamp) as last_seen_seconds
        FROM detected_persons
        WHERE video_id = ?
        GROUP BY person_id
        ORDER BY CAST(person_id AS INTEGER)
    """, (vid_id,))
    
    persons = cursor.fetchall()
    print(f"Found {len(persons)} unique persons\n")
    
    # Parse OCR time
    if ocr_time:
        # Parse time string (format: HH:MM:SS.ffffff)
        time_parts = ocr_time.split(':')
        base_hour = int(time_parts[0])
        base_minute = int(time_parts[1])
        base_second = float(time_parts[2])
        
        print("ATTENDANCE RECORDS (Narrative Format):")
        print(f"From the {location} video on {ocr_date}:")
        
        for person_id, count, first_ts, last_ts in persons:
            # Calculate actual clock times
            first_seconds = base_hour * 3600 + base_minute * 60 + base_second + first_ts
            last_seconds = base_hour * 3600 + base_minute * 60 + base_second + last_ts
            
            # Convert to HH:MM:SS format
            first_time = f"{int(first_seconds//3600):02d}:{int((first_seconds%3600)//60):02d}:{int(first_seconds%60):02d}"
            last_time = f"{int(last_seconds//3600):02d}:{int((last_seconds%3600)//60):02d}:{int(last_seconds%60):02d}"
            
            duration = last_ts - first_ts
            
            # Format duration
            if duration < 60:
                duration_str = f"{int(duration)} seconds"
            else:
                duration_str = f"{int(duration//60)} minutes {int(duration%60)} seconds"
            
            print(f"  - Person {person_id} was present from {first_time} to {last_time} ({duration_str})")
        
        print(f"  - Location: {location}")
        print(f"  - Date: {ocr_date}")

# 3. Check what the attendance page would query
print("\n\n3. ATTENDANCE PAGE QUERY SIMULATION:")
print("The attendance page queries videos by date, then gets detections for each video.")

# Test the query that attendance page uses
if video:
    cursor.execute("""
        SELECT COUNT(*) 
        FROM videos 
        WHERE ocr_video_date = ? AND ocr_extraction_done = 1
    """, (ocr_date,))
    
    count = cursor.fetchone()[0]
    print(f"\nVideos for date {ocr_date}: {count}")
    
    # Get all detections for attendance calculation
    cursor.execute("""
        SELECT dp.id, dp.person_id, dp.timestamp, dp.confidence
        FROM detected_persons dp
        JOIN videos v ON dp.video_id = v.id
        WHERE v.ocr_video_date = ? AND v.ocr_extraction_done = 1
        ORDER BY dp.person_id, dp.timestamp
    """, (ocr_date,))
    
    all_detections = cursor.fetchall()
    print(f"Total detections for this date: {len(all_detections)}")

print("\n\n4. INSTRUCTIONS TO SEE ATTENDANCE DATA:")
print("-" * 50)
print("1. Make sure Flask server is running:")
print("   cd /mnt/d/sm-hrm")
print("   python app.py")
print("")
print("2. Go to: http://localhost:5001/auth/login")
print("   - Email: admin@stepmedia.com")
print("   - Password: (leave blank - just click Login)")
print("")
print("3. After login, go to: http://localhost:5001/attendance/")
print("")
print("4. You should see:")
print("   - Dashboard with statistics")
print("   - 'Recent Attendance Records' section showing the data above")
print("")
print("5. Click 'Daily Attendance Report' to see more details")
print(f"   - Select date: {ocr_date if video else 'today'}")
print("")
print("6. Alternative - Direct API test (after login):")
print(f"   http://localhost:5001/attendance/daily?format=json&date={ocr_date if video else '2025-12-05'}")

conn.close()