"""
Migration script to add annotated_video_path field to videos table
"""
import sqlite3
import sys

def migrate():
    try:
        # Connect to database
        conn = sqlite3.connect('instance/stepmedia_hrm.db')
        cursor = conn.cursor()
        
        # Check if column already exists
        cursor.execute("PRAGMA table_info(videos)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'annotated_video_path' not in columns:
            print("Adding annotated_video_path column to videos table...")
            cursor.execute("ALTER TABLE videos ADD COLUMN annotated_video_path VARCHAR(500)")
            conn.commit()
            print("[OK] Migration completed successfully!")
        else:
            print("[OK] Column annotated_video_path already exists, skipping migration.")
        
        # Also add processing_progress if missing
        if 'processing_progress' not in columns:
            print("Adding processing_progress column to videos table...")
            cursor.execute("ALTER TABLE videos ADD COLUMN processing_progress INTEGER DEFAULT 0")
            conn.commit()
            print("[OK] Added processing_progress column!")
        
        conn.close()
        
    except Exception as e:
        print(f"[ERROR] Migration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    migrate()