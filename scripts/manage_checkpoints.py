#!/usr/bin/env python3
"""
Manage video processing checkpoints - view, resume, or clean up checkpoints
"""
import os
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processing.checkpoint_manager import get_checkpoint_manager
from config_database import db, Video

def list_checkpoints():
    """List all existing checkpoints"""
    manager = get_checkpoint_manager()
    checkpoints = manager.list_checkpoints()
    
    if not checkpoints:
        print("[OK] No checkpoints found")
        return
    
    print(f"\n[TRACE] Found {len(checkpoints)} checkpoint(s):\n")
    
    for cp in checkpoints:
        print(f"Video ID: {cp['video_id']}")
        print(f"  - Checkpoint ID: {cp['checkpoint_id']}")
        print(f"  - Created: {cp['created_at']}")
        print(f"  - Updated: {cp.get('updated_at', 'N/A')}")
        print(f"  - Status: {cp.get('status', 'unknown')}")
        print(f"  - Progress: {cp['last_processed_frame']}/{cp['total_frames']} frames")
        progress_pct = (cp['last_processed_frame'] / cp['total_frames'] * 100) if cp['total_frames'] > 0 else 0
        print(f"  - Progress %: {progress_pct:.1f}%")
        print(f"  - File: {cp['file_path']}")
        print()

def check_pending_videos():
    """Check for videos that might need resuming"""
    from app import app
    
    with app.app_context():
        # Find videos that are still marked as processing
        processing_videos = Video.query.filter(
            Video.processing_status == 'processing'
        ).all()
        
        if not processing_videos:
            print("[OK] No videos stuck in processing state")
            return
        
        print(f"\n[WARNING]  Found {len(processing_videos)} video(s) in processing state:")
        
        manager = get_checkpoint_manager()
        
        for video in processing_videos:
            print(f"\n[VIDEO] Video ID: {video.id}")
            print(f"   - Filename: {video.filename}")
            print(f"   - Upload date: {video.upload_date}")
            print(f"   - Progress: {video.processing_progress}%")
            
            # Check if checkpoint exists
            checkpoint = manager.load_checkpoint(video.id)
            if checkpoint:
                print(f"   [OK] Checkpoint found - can be resumed")
                frames = checkpoint.get('last_processed_frame', 0)
                total = checkpoint.get('total_frames', 0)
                if total > 0:
                    print(f"   - Checkpoint progress: {frames}/{total} frames ({frames/total*100:.1f}%)")
            else:
                print(f"   [ERROR] No checkpoint found - needs to restart from beginning")

def resume_video(video_id):
    """Resume processing a specific video"""
    from app import app
    
    manager = get_checkpoint_manager()
    checkpoint = manager.load_checkpoint(video_id)
    
    if not checkpoint:
        print(f"[ERROR] No checkpoint found for video {video_id}")
        return
    
    print(f"\n[LOAD] Found checkpoint for video {video_id}:")
    print(f"   - Last frame: {checkpoint.get('last_processed_frame', 0)}")
    print(f"   - Total frames: {checkpoint.get('total_frames', 0)}")
    print(f"   - Status: {checkpoint.get('status', 'unknown')}")
    
    with app.app_context():
        video = Video.query.get(video_id)
        if not video:
            print(f"[ERROR] Video {video_id} not found in database")
            return
        
        print(f"\n[VIDEO] Video details:")
        print(f"   - Filename: {video.filename}")
        print(f"   - Status: {video.processing_status}")
        
        # Check if we should resume
        response = input("\n[PROCESSING] Resume processing this video? (y/n): ")
        if response.lower() == 'y':
            print("\n[START] Resuming video processing...")
            # Import and call the processing function
            from processing.gpu_enhanced_detection import gpu_person_detection_task
            
            video_path = os.path.join('static', 'uploads', video.filename)
            if not os.path.exists(video_path):
                print(f"[ERROR] Video file not found: {video_path}")
                return
            
            # Update status
            video.processing_status = 'processing'
            video.processing_log = f"{video.processing_log or ''}\nResuming from checkpoint at {datetime.now()}"
            db.session.commit()
            
            # Process video
            result = gpu_person_detection_task(
                video_path=video_path,
                video_id=video.id,
                app=app
            )
            
            if 'error' in result:
                print(f"[ERROR] Error resuming: {result['error']}")
            else:
                print(f"[OK] Video processing resumed successfully")

def cleanup_old_checkpoints(days=7):
    """Clean up checkpoints older than specified days"""
    manager = get_checkpoint_manager()
    
    print(f"\n[CLEANUP] Cleaning up checkpoints older than {days} days...")
    deleted = manager.cleanup_old_checkpoints(days)
    print(f"[OK] Deleted {deleted} old checkpoint(s)")

def delete_checkpoint(video_id):
    """Delete checkpoint for a specific video"""
    manager = get_checkpoint_manager()
    
    if manager.delete_checkpoint(video_id):
        print(f"[OK] Deleted checkpoint for video {video_id}")
    else:
        print(f"[ERROR] No checkpoint found for video {video_id}")

def main():
    parser = argparse.ArgumentParser(description='Manage video processing checkpoints')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # List command
    subparsers.add_parser('list', help='List all checkpoints')
    
    # Check command
    subparsers.add_parser('check', help='Check for videos that need resuming')
    
    # Resume command
    resume_parser = subparsers.add_parser('resume', help='Resume processing a video')
    resume_parser.add_argument('video_id', type=int, help='Video ID to resume')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old checkpoints')
    cleanup_parser.add_argument('--days', type=int, default=7, help='Delete checkpoints older than this many days')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete checkpoint for a video')
    delete_parser.add_argument('video_id', type=int, help='Video ID')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_checkpoints()
    elif args.command == 'check':
        check_pending_videos()
    elif args.command == 'resume':
        resume_video(args.video_id)
    elif args.command == 'cleanup':
        cleanup_old_checkpoints(args.days)
    elif args.command == 'delete':
        delete_checkpoint(args.video_id)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()