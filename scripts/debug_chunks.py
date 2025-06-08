#!/usr/bin/env python3
"""
Debug video chunks status
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from datetime import datetime

def debug_chunks():
    app = create_app()
    
    with app.app_context():
        db = app.db
        Video = app.Video
        
        # Find all parent videos
        parent_videos = Video.query.filter_by(is_chunk=False).filter(
            Video.total_chunks.isnot(None)
        ).all()
        
        print(f"\n=== PARENT VIDEOS WITH CHUNKS ===")
        for parent in parent_videos:
            print(f"\nParent Video ID: {parent.id}")
            print(f"  Filename: {parent.filename}")
            print(f"  Status: {parent.status}")
            print(f"  Total Chunks: {parent.total_chunks}")
            print(f"  Created: {parent.created_at}")
            
            # Find chunks
            chunks = Video.query.filter_by(
                parent_video_id=parent.id,
                is_chunk=True
            ).order_by(Video.chunk_index).all()
            
            print(f"  Found {len(chunks)} chunks:")
            for chunk in chunks:
                print(f"    - Chunk {chunk.chunk_index}: {chunk.filename}")
                print(f"      Status: {chunk.status}")
                print(f"      Progress: {chunk.processing_progress}%")
                print(f"      File exists: {os.path.exists(chunk.file_path)}")
                
        # Find orphan chunks
        orphan_chunks = Video.query.filter_by(is_chunk=True).filter(
            Video.parent_video_id.is_(None)
        ).all()
        
        if orphan_chunks:
            print(f"\n=== ORPHAN CHUNKS (no parent) ===")
            for chunk in orphan_chunks:
                print(f"  - {chunk.filename} (ID: {chunk.id})")

if __name__ == "__main__":
    debug_chunks()