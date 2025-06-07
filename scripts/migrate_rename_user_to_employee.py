#!/usr/bin/env python3
"""
Rename user_id to employee_id in videos table
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from models.base import db
from sqlalchemy import text

def rename_user_to_employee():
    """Rename user_id to employee_id"""
    app = create_app()
    
    with app.app_context():
        try:
            # Check if user_id exists and employee_id doesn't
            result = db.session.execute(text("PRAGMA table_info(videos)"))
            columns = [row[1] for row in result.fetchall()]
            
            if 'user_id' in columns and 'employee_id' not in columns:
                print("Renaming user_id to employee_id...")
                # SQLite doesn't support ALTER COLUMN, so we need to:
                # 1. Add new column
                db.session.execute(text("""
                    ALTER TABLE videos 
                    ADD COLUMN employee_id INTEGER
                """))
                
                # 2. Copy data
                db.session.execute(text("""
                    UPDATE videos 
                    SET employee_id = user_id
                """))
                
                # 3. Note: In SQLite we can't drop columns easily
                # So we'll just leave user_id and use employee_id going forward
                
                db.session.commit()
                print("✅ Created employee_id column and copied data")
            elif 'employee_id' in columns:
                print("✅ employee_id column already exists")
            else:
                # Neither exists, just add employee_id
                db.session.execute(text("""
                    ALTER TABLE videos 
                    ADD COLUMN employee_id INTEGER
                """))
                db.session.commit()
                print("✅ Added employee_id column")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            db.session.rollback()
            raise

if __name__ == "__main__":
    rename_user_to_employee()