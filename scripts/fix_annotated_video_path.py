#!/usr/bin/env python3
"""
Fix annotated video path in database
"""
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app

def fix_annotated_video_paths():
    """Find and fix missing annotated video paths"""
    app = create_app()
    
    with app.app_context():
        db = app.db
        Video = app.Video
        
        # Find videos with missing annotated_video_path
        videos_to_fix = Video.query.filter(
            Video.status == 'completed',
            Video.annotated_video_path == None
        ).all()
        
        print(f"Found {len(videos_to_fix)} completed videos with missing annotated_video_path")
        
        outputs_dir = os.path.join('processing', 'outputs')
        
        for video in videos_to_fix:
            print(f"\nProcessing video ID {video.id}: {video.filename}")
            print(f"  File path: {video.file_path}")
            
            # Look for matching annotated video in outputs directory
            if os.path.exists(outputs_dir):
                # Get the base filename without extension
                base_name = video.file_path.rsplit('.', 1)[0] if video.file_path else ''
                
                # Look for annotated video files
                found_files = []
                for file in os.listdir(outputs_dir):
                    if file.endswith('.mp4') and 'annotated' in file:
                        # Check if this file matches our video
                        if base_name and base_name in file:
                            found_files.append(file)
                        elif video.file_path and video.file_path.replace('.mp4', '') in file:
                            found_files.append(file)
                
                if found_files:
                    # Use the most recent annotated file
                    annotated_file = sorted(found_files)[-1]
                    print(f"  [OK] Found annotated video: {annotated_file}")
                    
                    # Update the database
                    # Store just the filename (without processing/outputs/ prefix)
                    video.annotated_video_path = annotated_file.replace('processing/outputs/', '')
                    
                    # Also update processed_path if not set
                    if not video.processed_path:
                        video.processed_path = video.annotated_video_path
                    
                    db.session.commit()
                    print(f"  [OK] Updated annotated_video_path to: {video.annotated_video_path}")
                else:
                    print(f"  [ERROR] No annotated video found for this video")
            else:
                print(f"  [ERROR] Outputs directory not found: {outputs_dir}")
        
        print("\n[OK] Finished updating annotated video paths")

if __name__ == '__main__':
    fix_annotated_video_paths()