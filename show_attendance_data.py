#!/usr/bin/env python3
"""
Show exactly what the attendance page displays
"""

import sqlite3
import os
from datetime import datetime

db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "instance", "stepmedia_hrm.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("="*80)
print("ATTENDANCE DATA - EXACTLY AS IT SHOULD APPEAR")
print("="*80)

# Get the video with OCR
cursor.execute("""
    SELECT id, filename, ocr_video_date, ocr_video_time, ocr_location
    FROM videos 
    WHERE ocr_extraction_done = 1
""")
video = cursor.fetchone()
vid_id, filename, ocr_date, ocr_time, location = video

print(f"\nFrom the {location} video on December 5, 2025:")

# Get all persons and their times
cursor.execute("""
    SELECT person_id, 
           COUNT(*) as count,
           MIN(timestamp) as first_ts,
           MAX(timestamp) as last_ts
    FROM detected_persons
    WHERE video_id = ?
    GROUP BY person_id
    HAVING COUNT(*) > 5  -- Only show persons detected multiple times
    ORDER BY CAST(person_id AS INTEGER)
""", (vid_id,))

persons = cursor.fetchall()

# Parse OCR time to calculate actual clock times
base_time = datetime.strptime(f"{ocr_date} {ocr_time}", "%Y-%m-%d %H:%M:%S")

for person_id, count, first_ts, last_ts in persons:
    # Calculate actual times
    first_time = base_time.timestamp() + first_ts
    last_time = base_time.timestamp() + last_ts
    
    first_clock = datetime.fromtimestamp(first_time).strftime("%H:%M:%S")
    last_clock = datetime.fromtimestamp(last_time).strftime("%H:%M:%S")
    
    duration = int(last_ts - first_ts)
    
    print(f"  - Person {person_id} was present from {first_clock} to {last_clock} ({duration} seconds)")

print(f"  - Location: {location}")
print(f"  - Date: December 5, 2025")

print("\n" + "="*80)
print("HOW TO SEE THIS IN THE WEB UI:")
print("="*80)
print("\n1. Make sure the Flask server is running:")
print("   cd /mnt/d/sm-hrm")
print("   python app.py")
print("\n2. Open your browser and go to:")
print("   http://localhost:5001/attendance/demo")
print("   (This demo page doesn't require login)")
print("\n3. For the actual attendance page:")
print("   a) First login at: http://localhost:5001/auth/login")
print("      Email: admin@stepmedia.com")
print("      Password: (leave empty)")
print("   b) Then go to: http://localhost:5001/attendance/")
print("\n4. For direct API access (requires login first):")
print("   http://localhost:5001/attendance/daily?format=json&date=2025-12-05")

conn.close()