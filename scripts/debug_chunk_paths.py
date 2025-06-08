#!/usr/bin/env python3
"""
Debug chunk paths
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app

def debug_paths():
    app = create_app()
    
    with app.app_context():
        db = app.db
        Video = app.Video
        
        # Get one chunk example
        chunk = Video.query.filter_by(is_chunk=True).first()
        if chunk:
            print(f"Example chunk:")
            print(f"  ID: {chunk.id}")
            print(f"  Filename: {chunk.filename}")
            print(f"  File_path in DB: {chunk.file_path}")
            print(f"  Parent ID: {chunk.parent_video_id}")
            
            # List actual files
            print(f"\nActual chunk files in uploads/chunks:")
            chunks_dir = 'static/uploads/chunks'
            if os.path.exists(chunks_dir):
                for subdir in os.listdir(chunks_dir):
                    subdir_path = os.path.join(chunks_dir, subdir)
                    if os.path.isdir(subdir_path):
                        print(f"\n  Directory: {subdir}")
                        for file in os.listdir(subdir_path):
                            print(f"    - {file}")
                            if file == chunk.filename:
                                correct_path = f"chunks/{subdir}/{file}"
                                print(f"      [CHECK] MATCH! Correct path should be: {correct_path}")

if __name__ == "__main__":
    debug_paths()