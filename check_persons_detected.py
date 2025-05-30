#!/usr/bin/env python3
import sqlite3

conn = sqlite3.connect('instance/stepmedia_hrm.db')
cursor = conn.cursor()

print("=== Person Detection Summary ===\n")

# Get unique person_ids
cursor.execute("""
    SELECT DISTINCT person_id, COUNT(*) as detection_count, 
           MIN(timestamp) as first_seen, MAX(timestamp) as last_seen
    FROM detected_persons 
    WHERE video_id = 1
    GROUP BY person_id
    ORDER BY CAST(person_id AS INTEGER)
""")

persons = cursor.fetchall()
print(f"Found {len(persons)} unique persons in the video\n")

for person in persons:
    person_id, count, first_seen, last_seen = person
    duration = last_seen - first_seen
    print(f"Person {person_id}:")
    print(f"  - Detected {count} times")
    print(f"  - First seen at: {first_seen:.2f} seconds")
    print(f"  - Last seen at: {last_seen:.2f} seconds")
    print(f"  - Duration: {duration:.2f} seconds")
    print()

# Get video OCR info
cursor.execute("SELECT ocr_video_date, ocr_video_time, ocr_location FROM videos WHERE id = 1")
video_info = cursor.fetchone()
print(f"\nVideo OCR Info:")
print(f"  Date: {video_info[0]}")
print(f"  Time: {video_info[1]}")
print(f"  Location: {video_info[2]}")

conn.close()