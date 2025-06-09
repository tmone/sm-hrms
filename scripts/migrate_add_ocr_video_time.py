#!/usr/bin/env python3
"""
Add missing ocr_video_time field to videos table
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from sqlalchemy import text

def add_ocr_video_time_field():
    """Add ocr_video_time field to videos table"""
    app = create_app()
    
    with app.app_context():
        db = app.db
        
        print("Adding ocr_video_time field to videos table...")
        
        try:
            # Add ocr_video_time field to videos table
            sql = "ALTER TABLE videos ADD COLUMN IF NOT EXISTS ocr_video_time TIME"
            
            db.session.execute(text(sql))
            print(f"✓ Added ocr_video_time field successfully")
            
            db.session.commit()
            print("\nMigration completed successfully!")
            
            # Verify the field was added
            result = db.session.execute(text("PRAGMA table_info(videos)"))
            columns = [row[1] for row in result]
            
            if 'ocr_video_time' in columns:
                print("✓ Verified: ocr_video_time column exists in videos table")
            else:
                print("✗ Warning: ocr_video_time column not found in videos table")
            
        except Exception as e:
            print(f"\nError adding ocr_video_time field: {e}")
            db.session.rollback()

if __name__ == '__main__':
    add_ocr_video_time_field()