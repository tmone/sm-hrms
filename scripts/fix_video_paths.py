#!/usr/bin/env python
"""
Fix video paths in database to ensure they are in the correct format
"""
import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app, db

def fix_video_paths():
    """Fix annotated_video_path entries in the database"""
    app = create_app()
    
    with app.app_context():
        # Import Video model from app context
        Video = app.Video
        
        print("=" * 70)
        print("Fixing Video Paths in Database")
        print("=" * 70)
        print()
        
        # Get all videos
        videos = Video.query.all()
        fixed_count = 0
        
        for video in videos:
            if video.annotated_video_path:
                original_path = video.annotated_video_path
                
                # Check if path needs fixing
                if 'processing/outputs/' in original_path or '/' in original_path:
                    # Extract just the filename
                    fixed_path = os.path.basename(original_path)
                    video.annotated_video_path = fixed_path
                    
                    # Also update processed_path if it has the same issue
                    if video.processed_path and ('/' in video.processed_path):
                        video.processed_path = os.path.basename(video.processed_path)
                    
                    fixed_count += 1
                    print(f"[OK] Fixed Video {video.id}:")
                    print(f"   Original: {original_path}")
                    print(f"   Fixed:    {fixed_path}")
                    
                    # Verify the file exists
                    file_path = os.path.join('processing', 'outputs', fixed_path)
                    if os.path.exists(file_path):
                        print(f"   [OK] File exists: {file_path}")
                    else:
                        print(f"   [WARNING]  File not found: {file_path}")
                    print()
        
        if fixed_count > 0:
            # Commit changes
            db.session.commit()
            print(f"[OK] Fixed {fixed_count} video paths in database")
        else:
            print("[OK] All video paths are already in correct format")
        
        # Show current status
        print("\n" + "-" * 70)
        print("Current Video Status:")
        print("-" * 70)
        
        videos_with_annotated = Video.query.filter(Video.annotated_video_path.isnot(None)).count()
        videos_without_annotated = Video.query.filter(Video.annotated_video_path.is_(None)).count()
        
        print(f"Videos with annotated path: {videos_with_annotated}")
        print(f"Videos without annotated path: {videos_without_annotated}")
        
        # Check outputs directory
        outputs_dir = Path('processing/outputs')
        if outputs_dir.exists():
            video_files = list(outputs_dir.glob('*_annotated_*.mp4')) + list(outputs_dir.glob('*_annotated_*.avi'))
            print(f"\nAnnotated video files in outputs: {len(video_files)}")
            
            # Show unmatched files
            db_filenames = {v.annotated_video_path for v in videos if v.annotated_video_path}
            disk_filenames = {f.name for f in video_files}
            unmatched = disk_filenames - db_filenames
            
            if unmatched:
                print(f"\n[WARNING]  Found {len(unmatched)} annotated videos not linked in database:")
                for filename in list(unmatched)[:5]:
                    print(f"   - {filename}")
                if len(unmatched) > 5:
                    print(f"   ... and {len(unmatched) - 5} more")

if __name__ == "__main__":
    fix_video_paths()