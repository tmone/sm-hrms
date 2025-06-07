#!/usr/bin/env python3
"""
Add video chunks support for processing large videos
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from models.base import db
from sqlalchemy import text

def add_video_chunks_support():
    """Add fields to support video chunking"""
    app = create_app()
    
    with app.app_context():
        try:
            # Add parent_video_id field to videos table
            print("Adding parent_video_id to videos table...")
            result = db.session.execute(text("PRAGMA table_info(videos)"))
            columns = [row[1] for row in result.fetchall()]
            
            if 'parent_video_id' not in columns:
                db.session.execute(text("""
                    ALTER TABLE videos 
                    ADD COLUMN parent_video_id INTEGER REFERENCES videos(id)
                """))
                print("✅ Added parent_video_id column")
            
            # Add chunk_index field
            if 'chunk_index' not in columns:
                db.session.execute(text("""
                    ALTER TABLE videos 
                    ADD COLUMN chunk_index INTEGER
                """))
                print("✅ Added chunk_index column")
            
            # Add total_chunks field
            if 'total_chunks' not in columns:
                db.session.execute(text("""
                    ALTER TABLE videos 
                    ADD COLUMN total_chunks INTEGER
                """))
                print("✅ Added total_chunks column")
            
            # Add is_chunk field
            if 'is_chunk' not in columns:
                db.session.execute(text("""
                    ALTER TABLE videos 
                    ADD COLUMN is_chunk BOOLEAN DEFAULT 0
                """))
                print("✅ Added is_chunk column")
            
            db.session.commit()
            print("✅ Video chunks support added successfully")
            
        except Exception as e:
            print(f"❌ Error adding video chunks support: {e}")
            db.session.rollback()
            raise

if __name__ == "__main__":
    add_video_chunks_support()