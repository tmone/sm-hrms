#!/usr/bin/env python3
"""
Check for pending video processing on startup and optionally resume
This script should be run when the server starts
"""
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processing.checkpoint_manager import get_checkpoint_manager
from config_database import db, Video

def check_and_report_pending():
    """Check for pending videos and report status"""
    from app import app
    
    print("\n" + "="*60)
    print("[START] SERVER STARTUP - CHECKING PENDING VIDEO PROCESSING")
    print("="*60)
    
    with app.app_context():
        # Find videos that are still marked as processing
        processing_videos = Video.query.filter(
            Video.processing_status == 'processing'
        ).all()
        
        if not processing_videos:
            print("\n[OK] All clear - no videos stuck in processing state")
            return []
        
        print(f"\n[WARNING]  Found {len(processing_videos)} video(s) that may need attention:")
        
        manager = get_checkpoint_manager()
        resumable_videos = []
        
        for video in processing_videos:
            print(f"\n[VIDEO] Video ID: {video.id} - {video.filename}")
            print(f"   Upload date: {video.upload_date}")
            print(f"   Last progress: {video.processing_progress}%")
            
            # Check if checkpoint exists
            checkpoint = manager.load_checkpoint(video.id)
            if checkpoint:
                frames = checkpoint.get('last_processed_frame', 0)
                total = checkpoint.get('total_frames', 0)
                status = checkpoint.get('status', 'unknown')
                
                print(f"   [OK] CHECKPOINT FOUND")
                print(f"      - Status: {status}")
                print(f"      - Progress: {frames}/{total} frames", end="")
                if total > 0:
                    print(f" ({frames/total*100:.1f}%)")
                else:
                    print()
                    
                if 'error' in checkpoint:
                    print(f"      - Last error: {checkpoint['error']}")
                    print(f"      - Error time: {checkpoint.get('error_time', 'Unknown')}")
                
                resumable_videos.append({
                    'video': video,
                    'checkpoint': checkpoint
                })
            else:
                print(f"   [ERROR] NO CHECKPOINT - would need to restart from beginning")
                # Mark as failed if no checkpoint
                video.processing_status = 'failed'
                video.processing_log = f"{video.processing_log or ''}\nNo checkpoint found on server restart at {datetime.now()}"
                db.session.commit()
        
        print("\n" + "-"*60)
        
        if resumable_videos:
            print(f"\n[INFO] SUMMARY: {len(resumable_videos)} video(s) can be resumed")
            print("\nTo resume processing:")
            for item in resumable_videos:
                video = item['video']
                print(f"   python scripts/manage_checkpoints.py resume {video.id}")
            
            print("\nOr to resume all at once:")
            print("   python scripts/resume_all_pending.py")
        
        return resumable_videos

def main():
    """Main entry point"""
    try:
        resumable = check_and_report_pending()
        
        # Also check checkpoint directory
        manager = get_checkpoint_manager()
        all_checkpoints = manager.list_checkpoints()
        
        if len(all_checkpoints) > len(resumable):
            print(f"\n[WARNING]  Note: Found {len(all_checkpoints)} total checkpoint(s) but only {len(resumable)} match processing videos")
            print("   Run 'python scripts/manage_checkpoints.py cleanup' to remove orphaned checkpoints")
        
        print("\n" + "="*60)
        print("[OK] Startup check complete")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n[ERROR] Error during startup check: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()