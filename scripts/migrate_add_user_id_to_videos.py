#!/usr/bin/env python3
"""
Add user_id field to videos table
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from models.base import db
from sqlalchemy import text

def add_user_id_field():
    """Add user_id field to videos table"""
    app = create_app()
    
    with app.app_context():
        try:
            # Check if column already exists
            result = db.session.execute(text("PRAGMA table_info(videos)"))
            columns = [row[1] for row in result.fetchall()]
            
            if 'user_id' not in columns:
                print("Adding user_id column to videos table...")
                db.session.execute(text("""
                    ALTER TABLE videos 
                    ADD COLUMN user_id INTEGER
                """))
                db.session.commit()
                print("[OK] Added user_id column")
            else:
                print("[OK] user_id column already exists")
                
        except Exception as e:
            print(f"[ERROR] Error adding user_id column: {e}")
            db.session.rollback()
            raise

if __name__ == "__main__":
    add_user_id_field()