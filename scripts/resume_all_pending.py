#!/usr/bin/env python3
"""
Resume all pending video processing jobs that have checkpoints
"""
import os
import sys
import time
import threading
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processing.checkpoint_manager import get_checkpoint_manager
from config_database import db, Video
from processing.gpu_enhanced_detection import gpu_person_detection_task

def resume_video_processing(video, checkpoint, app):
    """Resume processing for a single video"""
    try:
        print(f"\n[PROCESSING] Resuming video {video.id}: {video.filename}")
        print(f"   Starting from frame {checkpoint.get('last_processed_frame', 0)}")
        
        video_path = os.path.join('static', 'uploads', video.filename)
        if not os.path.exists(video_path):
            print(f"[ERROR] Video file not found: {video_path}")
            video.processing_status = 'failed'
            video.processing_log = f"{video.processing_log or ''}\nFile not found during resume at {datetime.now()}"
            with app.app_context():
                db.session.commit()
            return False
        
        # Update status
        with app.app_context():
            video_obj = Video.query.get(video.id)
            video_obj.processing_status = 'processing'
            video_obj.processing_log = f"{video_obj.processing_log or ''}\nResumed from checkpoint at {datetime.now()}"
            db.session.commit()
        
        # Process video with app context
        result = gpu_person_detection_task(
            video_path=video_path,
            video_id=video.id,
            app=app
        )
        
        if 'error' in result:
            print(f"[ERROR] Error resuming video {video.id}: {result['error']}")
            with app.app_context():
                video_obj = Video.query.get(video.id)
                video_obj.processing_status = 'failed'
                video_obj.processing_log = f"{video_obj.processing_log or ''}\nResume failed: {result['error']}"
                db.session.commit()
            return False
        else:
            print(f"[OK] Video {video.id} processing completed successfully")
            with app.app_context():
                video_obj = Video.query.get(video.id)
                video_obj.processing_status = 'completed'
                video_obj.processing_progress = 100
                db.session.commit()
            return True
            
    except Exception as e:
        print(f"[ERROR] Exception resuming video {video.id}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Resume all pending videos with checkpoints"""
    from app import app
    
    print("\n" + "="*60)
    print("[START] RESUMING ALL PENDING VIDEO PROCESSING")
    print("="*60)
    
    with app.app_context():
        # Find videos that are still marked as processing
        processing_videos = Video.query.filter(
            Video.processing_status == 'processing'
        ).all()
        
        if not processing_videos:
            print("\n[OK] No videos in processing state")
            return
        
        manager = get_checkpoint_manager()
        videos_to_resume = []
        
        # Check which videos have checkpoints
        for video in processing_videos:
            checkpoint = manager.load_checkpoint(video.id)
            if checkpoint:
                videos_to_resume.append((video, checkpoint))
            else:
                print(f"\n[WARNING]  Video {video.id} ({video.filename}) has no checkpoint - skipping")
        
        if not videos_to_resume:
            print("\n[OK] No videos with checkpoints to resume")
            return
        
        print(f"\n[INFO] Found {len(videos_to_resume)} video(s) to resume")
        
        # Process videos sequentially to avoid overload
        successful = 0
        failed = 0
        
        for i, (video, checkpoint) in enumerate(videos_to_resume, 1):
            print(f"\n{'='*60}")
            print(f"Processing {i}/{len(videos_to_resume)}")
            print(f"{'='*60}")
            
            if resume_video_processing(video, checkpoint, app):
                successful += 1
            else:
                failed += 1
            
            # Small delay between videos
            if i < len(videos_to_resume):
                print("\n[WAIT] Waiting 5 seconds before next video...")
                time.sleep(5)
        
        print("\n" + "="*60)
        print("[INFO] RESUME SUMMARY")
        print(f"   - Total videos: {len(videos_to_resume)}")
        print(f"   - [OK] Successful: {successful}")
        print(f"   - [ERROR] Failed: {failed}")
        print("="*60 + "\n")

if __name__ == '__main__':
    main()