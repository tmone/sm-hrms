#!/usr/bin/env python3
"""
Restart processing for uploaded chunks
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from datetime import datetime

def restart_chunks():
    app = create_app()
    
    with app.app_context():
        db = app.db
        Video = app.Video
        
        # Import the processing function
        from hr_management.blueprints.videos import start_enhanced_gpu_processing
        
        # Find chunks that need processing
        chunks = Video.query.filter_by(
            is_chunk=True,
            status='uploaded'
        ).all()
        
        print(f"Found {len(chunks)} chunks to process")
        
        for chunk in chunks:
            print(f"\nProcessing chunk {chunk.chunk_index + 1}/{chunk.total_chunks}: {chunk.filename}")
            
            # Update status
            chunk.status = 'processing'
            chunk.processing_started_at = datetime.utcnow()
            db.session.commit()
            
            # Processing options
            processing_options = {
                'extract_persons': True,
                'face_recognition': False,
                'extract_frames': False,
                'use_enhanced_detection': True,
                'use_gpu': True
            }
            
            # Start processing
            try:
                start_enhanced_gpu_processing(chunk, processing_options, app)
                print(f"[OK] Started processing for chunk {chunk.id}")
            except Exception as e:
                print(f"[ERROR] Error starting chunk {chunk.id}: {e}")
                chunk.status = 'failed'
                chunk.error_message = str(e)
                db.session.commit()

if __name__ == "__main__":
    restart_chunks()