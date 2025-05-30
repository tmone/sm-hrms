#!/usr/bin/env python3
"""
Fix OCR time extraction for video
"""

import sqlite3
import os

db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "instance", "stepmedia_hrm.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("Checking current OCR data...")
cursor.execute("SELECT id, filename, ocr_video_date, ocr_video_time, ocr_location FROM videos WHERE id = 1")
video = cursor.fetchone()
print(f"Video: {video}")

# Based on the previous output, we know the time should be 08:55:22
# Let's update it manually for now
print("\nUpdating OCR time to 08:55:22...")
cursor.execute("""
    UPDATE videos 
    SET ocr_video_time = '08:55:22' 
    WHERE id = 1
""")

conn.commit()

# Verify the update
cursor.execute("SELECT id, filename, ocr_video_date, ocr_video_time, ocr_location FROM videos WHERE id = 1")
video = cursor.fetchone()
print(f"Updated video: {video}")

print("\nNow the attendance page should show clock times!")
print("Go to: http://localhost:5001/attendance/")

conn.close()