"""
Script to reset all person codes and start from PERSON-0001
This will:
1. Clear all person folders in processing/outputs/persons
2. Reset the person ID counter to 0 
3. Clear all DetectedPerson records from database
4. Update all videos to have 0 person count
"""
import os
import sys
import shutil
from pathlib import Path
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app


def reset_person_codes():
    """Reset all person codes and start from PERSON-0001"""
    print("[PROCESSING] Starting person code reset...")
    
    # 1. Clear all person folders
    persons_dir = Path('processing/outputs/persons')
    if persons_dir.exists():
        print(f"[DELETE]  Removing all person folders in {persons_dir}")
        # Remove all PERSON-* directories
        person_folders = list(persons_dir.glob('PERSON-*'))
        for folder in person_folders:
            try:
                shutil.rmtree(folder)
                print(f"   Removed: {folder.name}")
            except Exception as e:
                print(f"   [WARNING]  Error removing {folder.name}: {e}")
        
        # Remove the counter file
        counter_file = persons_dir / 'person_id_counter.json'
        if counter_file.exists():
            counter_file.unlink()
            print("   [OK] Removed person ID counter file")
    else:
        persons_dir.mkdir(parents=True, exist_ok=True)
        print(f"   [OK] Created persons directory: {persons_dir}")
    
    # 2. Reset the person ID counter to 0
    counter_file = persons_dir / 'person_id_counter.json'
    counter_data = {
        'last_person_id': 0,
        'updated_at': None,
        'total_persons': 0
    }
    with open(counter_file, 'w') as f:
        json.dump(counter_data, f, indent=2)
    print("   [OK] Reset person ID counter to 0 (next ID will be PERSON-0001)")
    
    # 3. Clear database records
    app = create_app()
    with app.app_context():
        db = app.db
        DetectedPerson = app.DetectedPerson
        Video = app.Video
        
        # Count existing records
        person_count = DetectedPerson.query.count()
        print(f"\n[INFO] Found {person_count} person detection records in database")
        
        if person_count > 0:
            # Clear all DetectedPerson records
            DetectedPerson.query.delete()
            print("   [OK] Deleted all DetectedPerson records")
            
            # Update all videos to have 0 person count
            videos = Video.query.all()
            for video in videos:
                video.person_count = 0
            print(f"   [OK] Reset person_count to 0 for {len(videos)} videos")
            
            # Commit changes
            db.session.commit()
            print("   [OK] Database changes committed")
        else:
            print("   ℹ️  No person detection records to clear")
    
    print("\n[OK] Person code reset complete!")
    print("   - All person folders removed")
    print("   - Person ID counter reset to 0")
    print("   - All DetectedPerson records cleared")
    print("   - All video person counts reset to 0")
    print("   - Next person will be: PERSON-0001")


if __name__ == '__main__':
    # Ask for confirmation
    print("[WARNING]  WARNING: This will delete ALL person data!")
    print("   - All person folders and images")
    print("   - All person detection records in database")
    print("   - All video person counts will be reset to 0")
    print()
    
    confirmation = input("Are you sure you want to continue? Type 'YES' to confirm: ")
    
    if confirmation == 'YES':
        reset_person_codes()
    else:
        print("[ERROR] Reset cancelled")