#!/usr/bin/env python3
"""
Re-extract OCR data from all uploaded videos
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from hr_management.processing.ocr_extractor import VideoOCRExtractor
from datetime import datetime

def reextract_ocr_for_all_videos(force=False):
    """Re-extract OCR data for all videos"""
    app = create_app()
    
    with app.app_context():
        Video = app.Video
        db = app.db
        
        # Check if OCR extractor is available
        try:
            ocr_extractor = VideoOCRExtractor(ocr_engine='easyocr')
            print("✓ OCR extractor initialized successfully")
        except Exception as e:
            print(f"✗ Failed to initialize OCR extractor: {e}")
            print("\nMake sure you have installed OCR dependencies:")
            print("  pip install easyocr opencv-python")
            return
        
        # Get all videos
        if force:
            videos = Video.query.all()
            print(f"\nFound {len(videos)} total videos (force mode)")
        else:
            videos = Video.query.filter(
                (Video.ocr_extraction_done == False) | 
                (Video.ocr_extraction_done == None)
            ).all()
            print(f"\nFound {len(videos)} videos without OCR data")
        
        if not videos:
            print("No videos need OCR extraction")
            return
        
        # Process each video
        success_count = 0
        for i, video in enumerate(videos, 1):
            print(f"\n[{i}/{len(videos)}] Processing: {video.filename}")
            print(f"  Path: {video.file_path}")
            
            # Check if file exists
            if not os.path.exists(video.file_path):
                print(f"  ✗ File not found!")
                continue
            
            try:
                # Extract OCR data
                print("  🔤 Extracting OCR data...")
                
                # Sample every 10 seconds for efficiency
                fps = 30  # Assume 30fps if not known
                sample_interval = int(fps * 10)
                
                ocr_data = ocr_extractor.extract_video_info(
                    video.file_path, 
                    sample_interval=sample_interval
                )
                
                if ocr_data:
                    # Update video with OCR data
                    video.ocr_location = ocr_data.get('location', 'Unknown')
                    video.ocr_video_date = ocr_data.get('date')
                    video.ocr_video_time = ocr_data.get('time')
                    video.ocr_extraction_confidence = ocr_data.get('confidence', 0.0)
                    video.ocr_extraction_done = True
                    
                    print(f"  ✓ Location: {video.ocr_location}")
                    print(f"  ✓ Date: {video.ocr_video_date}")
                    print(f"  ✓ Time: {video.ocr_video_time}")
                    print(f"  ✓ Confidence: {video.ocr_extraction_confidence:.2%}")
                    
                    success_count += 1
                else:
                    print("  ⚠️  No OCR data extracted")
                    video.ocr_extraction_done = True  # Mark as done even if no data
                    
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
                continue
        
        # Save all changes
        try:
            db.session.commit()
            print(f"\n✅ Successfully processed {success_count}/{len(videos)} videos")
        except Exception as e:
            print(f"\n✗ Failed to save changes: {e}")
            db.session.rollback()

def reset_and_reextract(video_id=None):
    """Reset OCR data and re-extract for specific video or all videos"""
    app = create_app()
    
    with app.app_context():
        Video = app.Video
        db = app.db
        
        if video_id:
            video = Video.query.get(video_id)
            if not video:
                print(f"Video with ID {video_id} not found")
                return
            videos = [video]
        else:
            videos = Video.query.all()
        
        print(f"Resetting OCR data for {len(videos)} video(s)...")
        
        # Reset OCR fields
        for video in videos:
            video.ocr_extraction_done = False
            video.ocr_location = None
            video.ocr_video_date = None
            video.ocr_video_time = None
            video.ocr_extraction_confidence = None
        
        db.session.commit()
        print("✓ OCR data reset successfully")
        
        # Now re-extract
        reextract_ocr_for_all_videos(force=True)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Re-extract OCR data from videos')
    parser.add_argument('--force', action='store_true', 
                       help='Force re-extraction even for videos with existing OCR data')
    parser.add_argument('--reset', action='store_true',
                       help='Reset all OCR data before extraction')
    parser.add_argument('--video-id', type=int,
                       help='Process only a specific video by ID')
    parser.add_argument('--list', action='store_true',
                       help='List all videos and their OCR status')
    
    args = parser.parse_args()
    
    if args.list:
        app = create_app()
        with app.app_context():
            Video = app.Video
            videos = Video.query.all()
            print("\nVideo OCR Status:")
            print("-" * 80)
            for v in videos:
                ocr_status = "✓ Has OCR data" if v.ocr_extraction_done else "✗ No OCR data"
                print(f"ID {v.id}: {v.filename}")
                print(f"  Status: {ocr_status}")
                if v.ocr_extraction_done:
                    print(f"  Location: {v.ocr_location}")
                    print(f"  Date: {v.ocr_video_date}")
                    print(f"  Time: {v.ocr_video_time}")
                print()
    elif args.reset:
        reset_and_reextract(args.video_id)
    else:
        if args.video_id:
            app = create_app()
            with app.app_context():
                Video = app.Video
                video = Video.query.get(args.video_id)
                if video:
                    print(f"Processing video ID {args.video_id}: {video.filename}")
                    # Process single video
                    from hr_management.processing.ocr_extractor import VideoOCRExtractor
                    ocr_extractor = VideoOCRExtractor(ocr_engine='easyocr')
                    ocr_data = ocr_extractor.extract_video_info(video.file_path)
                    if ocr_data:
                        video.ocr_location = ocr_data.get('location', 'Unknown')
                        video.ocr_video_date = ocr_data.get('date')
                        video.ocr_video_time = ocr_data.get('time')
                        video.ocr_extraction_confidence = ocr_data.get('confidence', 0.0)
                        video.ocr_extraction_done = True
                        app.db.session.commit()
                        print("✓ OCR extraction completed")
                else:
                    print(f"Video with ID {args.video_id} not found")
        else:
            reextract_ocr_for_all_videos(force=args.force)

if __name__ == '__main__':
    main()