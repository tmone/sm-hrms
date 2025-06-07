#!/usr/bin/env python3
"""
Fix all chunk paths - remove static/uploads prefix and fix slashes
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app

def fix_all_chunks():
    app = create_app()
    
    with app.app_context():
        db = app.db
        Video = app.Video
        
        # Get all chunks
        chunks = Video.query.filter_by(is_chunk=True).all()
        print(f"Found {len(chunks)} chunks to fix")
        
        for chunk in chunks:
            old_path = chunk.file_path
            
            # Remove static/uploads/ or static\uploads\ prefix
            new_path = old_path
            if new_path.startswith('static\\uploads\\'):
                new_path = new_path[16:]  # Remove 'static\uploads\'
            elif new_path.startswith('static/uploads/'):
                new_path = new_path[15:]  # Remove 'static/uploads/'
            
            # Convert backslashes to forward slashes
            new_path = new_path.replace('\\', '/')
            
            # Update in database
            chunk.file_path = new_path
            chunk.status = 'uploaded'  # Reset to uploaded so they can be processed
            
            print(f"Fixed chunk {chunk.id}:")
            print(f"  Old: {old_path}")
            print(f"  New: {new_path}")
            
        db.session.commit()
        print(f"\nAll chunks fixed!")

if __name__ == "__main__":
    fix_all_chunks()