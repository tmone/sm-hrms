#!/usr/bin/env python3
"""
Fix chunk video paths in database
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from pathlib import Path

def fix_chunk_paths():
    app = create_app()
    
    with app.app_context():
        db = app.db
        Video = app.Video
        
        # Find all chunk videos
        chunks = Video.query.filter_by(is_chunk=True).all()
        
        print(f"Found {len(chunks)} chunk videos to fix")
        
        fixed_count = 0
        for chunk in chunks:
            old_path = chunk.file_path
            
            # If path contains 'static/uploads', it's probably full path
            if 'static/uploads' in old_path or old_path.startswith('/'):
                # Extract just the relative path from chunks directory
                path_obj = Path(old_path)
                
                # Try to find the actual file
                possible_paths = [
                    f"chunks/{path_obj.parent.name}/{path_obj.name}",
                    f"chunks/{chunk.filename.split('_')[0]}_{chunk.filename.split('_')[1]}_{path_obj.parent.name}/{path_obj.name}",
                    path_obj.name  # Just filename
                ]
                
                # Check which path exists
                upload_dir = 'static/uploads'
                for possible_path in possible_paths:
                    full_path = os.path.join(upload_dir, possible_path)
                    if os.path.exists(full_path):
                        chunk.file_path = possible_path
                        chunk.status = 'uploaded'  # Reset status
                        fixed_count += 1
                        print(f"Fixed chunk {chunk.id}: {old_path} -> {possible_path}")
                        break
                else:
                    # Try to construct from parent name pattern
                    parent = Video.query.get(chunk.parent_video_id)
                    if parent:
                        # Look for directories matching parent filename
                        chunks_dir = os.path.join(upload_dir, 'chunks')
                        if os.path.exists(chunks_dir):
                            for dir_name in os.listdir(chunks_dir):
                                if parent.filename.split('.')[0] in dir_name:
                                    possible_path = f"chunks/{dir_name}/{chunk.filename}"
                                    full_path = os.path.join(upload_dir, possible_path)
                                    if os.path.exists(full_path):
                                        chunk.file_path = possible_path
                                        chunk.status = 'uploaded'
                                        fixed_count += 1
                                        print(f"Fixed chunk {chunk.id}: {old_path} -> {possible_path}")
                                        break
        
        db.session.commit()
        print(f"\nFixed {fixed_count} chunk paths")

if __name__ == "__main__":
    fix_chunk_paths()