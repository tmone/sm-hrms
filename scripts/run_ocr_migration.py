#!/usr/bin/env python3
"""
Simple OCR database migration for SQLite
This script directly updates the SQLite database with OCR fields
"""

import sqlite3
import os
from pathlib import Path

def find_database_file():
    """Find the SQLite database file"""
    possible_paths = [
        'instance/database.db',
        'instance/app.db', 
        'database.db',
        'app.db',
        'hr_management.db'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # Look for any .db files in instance folder
    instance_dir = Path('instance')
    if instance_dir.exists():
        db_files = list(instance_dir.glob('*.db'))
        if db_files:
            return str(db_files[0])
    
    return None

def run_migration():
    """Run the OCR migration on SQLite database"""
    
    # Find database file
    db_path = find_database_file()
    if not db_path:
        print("ERROR: No SQLite database file found!")
        print("   Looking for files like: instance/database.db, app.db, etc.")
        return False
    
    print(f"Found database: {db_path}")
    
    try:
        # Connect to SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print("Adding OCR fields to database...")
        
        # Add OCR fields to videos table
        video_commands = [
            "ALTER TABLE videos ADD COLUMN ocr_location TEXT",
            "ALTER TABLE videos ADD COLUMN ocr_video_date DATE", 
            "ALTER TABLE videos ADD COLUMN ocr_extraction_done BOOLEAN DEFAULT 0",
            "ALTER TABLE videos ADD COLUMN ocr_extraction_confidence REAL"
        ]
        
        for cmd in video_commands:
            try:
                cursor.execute(cmd)
                print(f"SUCCESS: {cmd}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e).lower():
                    print(f"WARNING: Column already exists: {cmd}")
                else:
                    print(f"ERROR: {cmd}: {e}")
        
        # Add attendance fields to detected_persons table
        person_commands = [
            "ALTER TABLE detected_persons ADD COLUMN attendance_date DATE",
            "ALTER TABLE detected_persons ADD COLUMN attendance_time TIME", 
            "ALTER TABLE detected_persons ADD COLUMN attendance_location TEXT",
            "ALTER TABLE detected_persons ADD COLUMN check_in_time TIMESTAMP",
            "ALTER TABLE detected_persons ADD COLUMN check_out_time TIMESTAMP"
        ]
        
        for cmd in person_commands:
            try:
                cursor.execute(cmd)
                print(f"SUCCESS: {cmd}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e).lower():
                    print(f"WARNING: Column already exists: {cmd}")
                else:
                    print(f"ERROR: {cmd}: {e}")
        
        # Update existing videos
        try:
            cursor.execute("UPDATE videos SET ocr_extraction_done = 0 WHERE ocr_extraction_done IS NULL")
            print("SUCCESS: Updated existing videos to mark OCR as not done")
        except Exception as e:
            print(f"WARNING: Update command: {e}")
        
        # Commit changes
        conn.commit()
        conn.close()
        
        print("\nDatabase migration completed successfully!")
        print("OCR fields have been added to your database")
        print("You can now use OCR extraction features")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Migration failed: {e}")
        return False

if __name__ == '__main__':
    print("OCR Database Migration for SQLite")
    print("=" * 50)
    
    if run_migration():
        print("\nNext steps:")
        print("1. Restart your Flask application")
        print("2. Go to any completed video and click 'Extract OCR Data'")
        print("3. Or run: python scripts/batch_extract_ocr.py")
    else:
        print("\nMigration failed. Please check the error messages above.")