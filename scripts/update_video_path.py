"""
Quick script to update the annotated video path for video ID 1
"""
import sqlite3
import os

# Connect to the database
db_path = 'instance/hr_management.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# The annotated video filename we found
annotated_filename = '3c63c24a-a120-43c3-a21a-a7fa6c84d9e9_TANG_TRET_84A_Tret_84A_Tret_20250512085459_20250512091458_472401_annotated_20250524_210010.mp4'

# Update the video record
cursor.execute("""
    UPDATE videos 
    SET annotated_video_path = ?, 
        processed_path = ?
    WHERE id = 1
""", (annotated_filename, annotated_filename))

# Commit the changes
conn.commit()

# Verify the update
cursor.execute("SELECT id, filename, annotated_video_path, processed_path FROM videos WHERE id = 1")
result = cursor.fetchone()
print(f"Updated video ID {result[0]}:")
print(f"  Filename: {result[1]}")
print(f"  Annotated path: {result[2]}")
print(f"  Processed path: {result[3]}")

conn.close()
print("\n[OK] Database updated successfully!")