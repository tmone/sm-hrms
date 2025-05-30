#!/usr/bin/env python3
import sqlite3
import os

# Try different database locations
db_paths = [
    'stepmedia_hrm.db',
    'instance/stepmedia_hrm.db',
    'instance/hr_management.db',
    'instance/hrm_database.db'
]

for db_path in db_paths:
    if os.path.exists(db_path):
        print(f"\n=== Checking {db_path} ===")
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [t[0] for t in cursor.fetchall()]
            
            if 'videos' in tables:
                print("Found videos table!")
                
                # Count videos with OCR
                cursor.execute("SELECT COUNT(*) FROM videos WHERE ocr_extraction_done = 1")
                ocr_count = cursor.fetchone()[0]
                print(f"Videos with OCR: {ocr_count}")
                
                # Sample video with OCR
                cursor.execute("SELECT id, filename, ocr_video_date, ocr_location FROM videos WHERE ocr_extraction_done = 1 LIMIT 3")
                videos = cursor.fetchall()
                for v in videos:
                    print(f"  Video {v[0]}: {v[1]}, Date: {v[2]}, Location: {v[3]}")
                
                # Check detected_persons
                if 'detected_persons' in tables:
                    cursor.execute("SELECT COUNT(*) FROM detected_persons")
                    det_count = cursor.fetchone()[0]
                    print(f"\nTotal detections: {det_count}")
                    
                    # Get columns
                    cursor.execute("PRAGMA table_info(detected_persons)")
                    columns = [col[1] for col in cursor.fetchall()]
                    print(f"DetectedPerson columns: {', '.join(columns[:10])}...")
                    
                    # Sample detections
                    if det_count > 0:
                        cursor.execute("SELECT * FROM detected_persons LIMIT 2")
                        detections = cursor.fetchall()
                        for d in detections:
                            print(f"\nDetection sample: {dict(zip(columns[:10], d[:10]))}")
                
            conn.close()
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"\n{db_path} does not exist")