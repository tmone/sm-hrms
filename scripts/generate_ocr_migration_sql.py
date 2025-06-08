#!/usr/bin/env python3
"""
Generate SQL commands to add OCR fields to existing database
Run the output SQL commands in your database to add OCR support
"""

def generate_migration_sql():
    """Generate SQL commands for adding OCR fields"""
    
    sql_commands = [
        "-- Add OCR fields to videos table",
        "ALTER TABLE videos ADD COLUMN IF NOT EXISTS ocr_location VARCHAR(100);",
        "ALTER TABLE videos ADD COLUMN IF NOT EXISTS ocr_video_date DATE;",
        "ALTER TABLE videos ADD COLUMN IF NOT EXISTS ocr_extraction_done BOOLEAN DEFAULT FALSE;",
        "ALTER TABLE videos ADD COLUMN IF NOT EXISTS ocr_extraction_confidence FLOAT;",
        "",
        "-- Add attendance fields to detected_persons table", 
        "ALTER TABLE detected_persons ADD COLUMN IF NOT EXISTS attendance_date DATE;",
        "ALTER TABLE detected_persons ADD COLUMN IF NOT EXISTS attendance_time TIME;",
        "ALTER TABLE detected_persons ADD COLUMN IF NOT EXISTS attendance_location VARCHAR(100);",
        "ALTER TABLE detected_persons ADD COLUMN IF NOT EXISTS check_in_time TIMESTAMP;",
        "ALTER TABLE detected_persons ADD COLUMN IF NOT EXISTS check_out_time TIMESTAMP;",
        "",
        "-- Update existing videos to mark OCR as not done",
        "UPDATE videos SET ocr_extraction_done = FALSE WHERE ocr_extraction_done IS NULL;",
        "",
        "-- Commit changes",
        "COMMIT;"
    ]
    
    return "\n".join(sql_commands)

if __name__ == '__main__':
    print("[TEXT] OCR Database Migration SQL Commands")
    print("=" * 50)
    print("Copy and run these SQL commands in your database:")
    print()
    
    sql = generate_migration_sql()
    print(sql)
    
    print("\n" + "=" * 50)
    print("[OK] After running these commands, you can use OCR extraction features!")