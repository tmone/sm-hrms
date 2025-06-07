#!/usr/bin/env python3
"""
Database migration script to add annotated_video_path column
Adds annotated_video_path column to videos table for enhanced detection
"""

import os
import sys
import sqlite3
from datetime import datetime

def backup_database(db_path):
    """Create a backup of the database before migration"""
    backup_path = f"{db_path}.backup_annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        import shutil
        shutil.copy2(db_path, backup_path)
        print(f"[OK] Database backed up to: {backup_path}")
        return backup_path
    except Exception as e:
        print(f"[WARNING] Warning: Could not create backup: {e}")
        return None

def check_column_exists(db_path):
    """Check if annotated_video_path column already exists"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get table schema
        cursor.execute("PRAGMA table_info(videos)")
        columns = [row[1] for row in cursor.fetchall()]
        
        conn.close()
        
        has_annotated_path = 'annotated_video_path' in columns
        
        print(f"[INFO] Current columns in videos: {len(columns)} columns")
        print(f"   - annotated_video_path exists: {has_annotated_path}")
        
        return has_annotated_path
        
    except Exception as e:
        print(f"[ERROR] Error checking columns: {e}")
        return False

def add_annotated_video_path_column(db_path):
    """Add annotated_video_path column to videos table"""
    try:
        print("[CONFIG] Adding annotated_video_path column to videos table...")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if column exists first
        has_annotated_path = check_column_exists(db_path)
        
        if not has_annotated_path:
            print("   Adding annotated_video_path column...")
            cursor.execute("ALTER TABLE videos ADD COLUMN annotated_video_path VARCHAR(500)")
            print("   [OK] annotated_video_path column added")
        else:
            print("   ℹ️ annotated_video_path column already exists")
        
        # Commit changes
        conn.commit()
        conn.close()
        
        print("[OK] Database migration completed successfully!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Migration failed: {e}")
        return False

def verify_migration(db_path):
    """Verify that the migration was successful"""
    try:
        print("[SEARCH] Verifying migration...")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check table structure
        cursor.execute("PRAGMA table_info(videos)")
        columns = cursor.fetchall()
        
        # Check for our new column
        column_names = [col[1] for col in columns]
        
        if 'annotated_video_path' in column_names:
            print("[OK] Migration verification successful!")
            print("   [INFO] Table structure updated:")
            for col in columns:
                if col[1] == 'annotated_video_path':
                    print(f"      [OK] {col[1]} ({col[2]})")
            return True
        else:
            print("[ERROR] Migration verification failed!")
            return False
            
    except Exception as e:
        print(f"[ERROR] Verification error: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def main():
    """Main migration function"""
    print("🗃️ Database Migration: Adding Annotated Video Path Column")
    print("=" * 60)
    
    # Find database file
    possible_db_paths = [
        'stepmedia_hrm.db',
        'instance/stepmedia_hrm.db',
        'hr_management.db'
    ]
    
    db_path = None
    for path in possible_db_paths:
        if os.path.exists(path):
            db_path = path
            break
    
    if not db_path:
        print("[ERROR] Database file not found!")
        print("   Checked paths:")
        for path in possible_db_paths:
            print(f"   - {path}")
        print("\nPlease run this script from the project root directory.")
        return False
    
    print(f"[FILE] Found database: {db_path}")
    
    # Check current schema
    has_annotated_path = check_column_exists(db_path)
    
    if has_annotated_path:
        print("[OK] Database already has annotated_video_path column!")
        return True
    
    # Create backup
    backup_path = backup_database(db_path)
    
    # Perform migration
    success = add_annotated_video_path_column(db_path)
    
    if success:
        # Verify migration
        if verify_migration(db_path):
            print("\n🎉 Migration completed successfully!")
            print("\nEnhanced detection can now store annotated video paths!")
            print("The video previewer will automatically show annotated videos when available.")
            
            if backup_path:
                print(f"\n[SAVE] Backup saved at: {backup_path}")
            
            return True
        else:
            print("\n[ERROR] Migration verification failed!")
            return False
    else:
        print("\n[ERROR] Migration failed!")
        if backup_path:
            print(f"[SAVE] You can restore from backup: {backup_path}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)