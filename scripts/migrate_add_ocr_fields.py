#!/usr/bin/env python3
"""
Add OCR extracted fields to database models
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from sqlalchemy import text

def add_ocr_fields():
    """Add OCR fields to videos and detected_persons tables"""
    app = create_app()
    
    with app.app_context():
        db = app.db
        
        print("Adding OCR fields to database...")
        
        try:
            # Add OCR fields to videos table
            video_fields = [
                "ALTER TABLE videos ADD COLUMN IF NOT EXISTS ocr_location VARCHAR(100)",
                "ALTER TABLE videos ADD COLUMN IF NOT EXISTS ocr_video_date DATE",
                "ALTER TABLE videos ADD COLUMN IF NOT EXISTS ocr_extraction_done BOOLEAN DEFAULT FALSE",
                "ALTER TABLE videos ADD COLUMN IF NOT EXISTS ocr_extraction_confidence FLOAT"
            ]
            
            for sql in video_fields:
                try:
                    db.session.execute(text(sql))
                    print(f"[CHECK] {sql}")
                except Exception as e:
                    print(f"✗ {sql}: {e}")
            
            # Add attendance fields to detected_persons table
            person_fields = [
                "ALTER TABLE detected_persons ADD COLUMN IF NOT EXISTS attendance_date DATE",
                "ALTER TABLE detected_persons ADD COLUMN IF NOT EXISTS attendance_time TIME",
                "ALTER TABLE detected_persons ADD COLUMN IF NOT EXISTS attendance_location VARCHAR(100)",
                "ALTER TABLE detected_persons ADD COLUMN IF NOT EXISTS check_in_time TIMESTAMP",
                "ALTER TABLE detected_persons ADD COLUMN IF NOT EXISTS check_out_time TIMESTAMP"
            ]
            
            for sql in person_fields:
                try:
                    db.session.execute(text(sql))
                    print(f"[CHECK] {sql}")
                except Exception as e:
                    print(f"✗ {sql}: {e}")
            
            # Create attendance summary table
            create_attendance_sql = """
            CREATE TABLE IF NOT EXISTS attendance_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id VARCHAR(20) NOT NULL,
                employee_id INTEGER,
                attendance_date DATE NOT NULL,
                location VARCHAR(100),
                first_seen_time TIME,
                last_seen_time TIME,
                total_duration_minutes INTEGER,
                detection_count INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (employee_id) REFERENCES employees(id),
                UNIQUE(person_id, attendance_date, location)
            )
            """
            
            try:
                db.session.execute(text(create_attendance_sql))
                print("[CHECK] Created attendance_summary table")
            except Exception as e:
                print(f"✗ Error creating attendance_summary table: {e}")
            
            # Add index for faster queries
            index_sql = [
                "CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance_summary(attendance_date)",
                "CREATE INDEX IF NOT EXISTS idx_person_attendance ON attendance_summary(person_id, attendance_date)",
                "CREATE INDEX IF NOT EXISTS idx_location_date ON attendance_summary(location, attendance_date)"
            ]
            
            for sql in index_sql:
                try:
                    db.session.execute(text(sql))
                    print(f"[CHECK] {sql}")
                except Exception as e:
                    print(f"✗ {sql}: {e}")
            
            db.session.commit()
            print("\nOCR fields added successfully!")
            
        except Exception as e:
            print(f"\nError adding OCR fields: {e}")
            db.session.rollback()

if __name__ == '__main__':
    add_ocr_fields()