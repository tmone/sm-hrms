#!/usr/bin/env python3
"""
Simple test to check attendance data
"""

# Set up minimal Flask app context
import os
os.environ['FLASK_APP'] = 'app.py'

try:
    from flask import Flask
    from flask_sqlalchemy import SQLAlchemy
    
    # Create minimal app
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///instance/hrm_database.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db = SQLAlchemy(app)
    
    with app.app_context():
        # Check videos table
        videos = db.session.execute("SELECT id, filename, ocr_extraction_done, ocr_video_date, ocr_location FROM videos WHERE ocr_extraction_done = 1").fetchall()
        print(f"Videos with OCR: {len(videos)}")
        for v in videos[:3]:
            print(f"  Video {v[0]}: {v[1]}, Date: {v[3]}, Location: {v[4]}")
        
        # Check detected_persons table
        detections = db.session.execute("SELECT COUNT(*) FROM detected_persons").fetchone()
        print(f"\nTotal detections: {detections[0]}")
        
        # Check sample detections
        sample_detections = db.session.execute("SELECT id, video_id, person_id, timestamp FROM detected_persons LIMIT 5").fetchall()
        print("\nSample detections:")
        for d in sample_detections:
            print(f"  Detection {d[0]}: Video {d[1]}, Person {d[2]}, Timestamp {d[3]}")
            
except Exception as e:
    print(f"Error: {e}")
    print("\nTrying direct SQLite access...")
    
    import sqlite3
    try:
        conn = sqlite3.connect('instance/hrm_database.db')
        cursor = conn.cursor()
        
        # List all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print("\nTables in database:")
        for table in tables:
            print(f"  - {table[0]}")
        
        # Check videos
        cursor.execute("SELECT COUNT(*) FROM videos WHERE ocr_extraction_done = 1")
        ocr_count = cursor.fetchone()[0]
        print(f"\nVideos with OCR: {ocr_count}")
        
        # Check detections
        cursor.execute("SELECT COUNT(*) FROM detected_persons")
        det_count = cursor.fetchone()[0]
        print(f"Total detections: {det_count}")
        
        conn.close()
    except Exception as e2:
        print(f"SQLite error: {e2}")