#!/usr/bin/env python3
"""
Reset OCR data for videos to allow rescanning
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app

def reset_video_ocr(video_id=None, filename=None):
    """Reset OCR data for a specific video"""
    app = create_app()
    
    with app.app_context():
        Video = app.Video
        db = app.db
        
        if video_id:
            video = Video.query.get(video_id)
        elif filename:
            video = Video.query.filter_by(filename=filename).first()
        else:
            # Show all videos
            videos = Video.query.all()
            print("Available videos:")
            for v in videos:
                ocr_status = "[Has OCR data]" if v.ocr_extraction_done else "[No OCR data]"
                print(f"  ID {v.id}: {v.filename} - {ocr_status}")
            return
        
        if not video:
            print("Video not found!")
            return
        
        print(f"Resetting OCR data for: {video.filename}")
        print(f"  Current location: {video.ocr_location}")
        print(f"  Current date: {video.ocr_video_date}")
        print(f"  Current time: {video.ocr_video_time}")
        
        # Reset OCR fields
        video.ocr_extraction_done = False
        video.ocr_location = None
        video.ocr_video_date = None
        video.ocr_video_time = None
        video.ocr_extraction_confidence = None
        
        db.session.commit()
        print("SUCCESS: OCR data reset! You can now rescan this video.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Reset OCR data for videos')
    parser.add_argument('--id', type=int, help='Video ID to reset')
    parser.add_argument('--filename', type=str, help='Video filename to reset')
    parser.add_argument('--all', action='store_true', help='Reset ALL videos')
    
    args = parser.parse_args()
    
    if args.all:
        confirm = input("Are you sure you want to reset OCR for ALL videos? (yes/no): ")
        if confirm.lower() == 'yes':
            app = create_app()
            with app.app_context():
                Video = app.Video
                db = app.db
                
                videos = Video.query.filter(Video.ocr_extraction_done == True).all()
                for video in videos:
                    video.ocr_extraction_done = False
                    video.ocr_location = None
                    video.ocr_video_date = None
                    video.ocr_video_time = None
                    video.ocr_extraction_confidence = None
                
                db.session.commit()
                print(f"SUCCESS: Reset OCR data for {len(videos)} videos")
    elif args.id or args.filename:
        reset_video_ocr(video_id=args.id, filename=args.filename)
    else:
        # Show list of videos
        reset_video_ocr()

if __name__ == '__main__':
    main()