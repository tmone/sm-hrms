"""
Migration script to add missing fields to videos table
"""
import sqlite3
import sys

def migrate():
    try:
        # Connect to database
        conn = sqlite3.connect('instance/stepmedia_hrm.db')
        cursor = conn.cursor()
        
        # Check if columns already exist
        cursor.execute("PRAGMA table_info(videos)")
        columns = [column[1] for column in cursor.fetchall()]
        
        # Add task_id if missing
        if 'task_id' not in columns:
            print("Adding task_id column to videos table...")
            cursor.execute("ALTER TABLE videos ADD COLUMN task_id VARCHAR(100)")
            conn.commit()
            print("[OK] Added task_id column!")
        else:
            print("[OK] Column task_id already exists.")
        
        # Add processing_progress if missing  
        if 'processing_progress' not in columns:
            print("Adding processing_progress column to videos table...")
            cursor.execute("ALTER TABLE videos ADD COLUMN processing_progress INTEGER DEFAULT 0")
            conn.commit()
            print("[OK] Added processing_progress column!")
        else:
            print("[OK] Column processing_progress already exists.")
            
        # Add annotated_video_path if missing
        if 'annotated_video_path' not in columns:
            print("Adding annotated_video_path column to videos table...")
            cursor.execute("ALTER TABLE videos ADD COLUMN annotated_video_path VARCHAR(500)")
            conn.commit()
            print("[OK] Added annotated_video_path column!")
        else:
            print("[OK] Column annotated_video_path already exists.")
        
        conn.close()
        print("\n[OK] Migration completed successfully!")
        
    except Exception as e:
        print(f"[ERROR] Migration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    migrate()