#!/usr/bin/env python3
"""
Convert existing videos to H.264 for web compatibility
"""

import os
import sys
import subprocess
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from pathlib import Path


def check_codec(video_path):
    """Check video codec"""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(video_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout.strip().lower()
    except:
        return 'unknown'


def convert_to_h264(input_path, output_path):
    """Convert video to H.264"""
    cmd = [
        'ffmpeg',
        '-i', str(input_path),
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        '-c:a', 'aac',  # Also convert audio to AAC
        '-y',
        str(output_path)
    ]
    
    try:
        print(f"Converting: {input_path} -> {output_path}")
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr.decode()}")
        return False


def main():
    app = create_app()
    
    with app.app_context():
        db = app.db
        Video = app.Video
        
        # Find all videos
        videos = Video.query.all()
        
        upload_dir = 'static/uploads'
        converted_count = 0
        
        for video in videos:
            # Check original video
            if video.file_path:
                video_path = os.path.join(upload_dir, video.file_path)
                if os.path.exists(video_path):
                    codec = check_codec(video_path)
                    print(f"\nVideo {video.id}: {video.filename}")
                    print(f"  Original codec: {codec}")
                    
                    if codec != 'h264':
                        # Create web-compatible version
                        base_name = Path(video_path).stem
                        web_path = os.path.join(upload_dir, f"{base_name}_web.mp4")
                        
                        if convert_to_h264(video_path, web_path):
                            # Update database
                            video.processed_path = f"{base_name}_web.mp4"
                            converted_count += 1
                            print(f"  ✅ Converted to H.264: {web_path}")
            
            # Check annotated video
            if video.annotated_video_path:
                annotated_path = os.path.join(upload_dir, video.annotated_video_path)
                if os.path.exists(annotated_path):
                    codec = check_codec(annotated_path)
                    print(f"  Annotated codec: {codec}")
                    
                    if codec != 'h264':
                        # Convert in place with temp file
                        temp_path = str(annotated_path) + '.tmp'
                        if convert_to_h264(annotated_path, temp_path):
                            os.replace(temp_path, annotated_path)
                            print(f"  ✅ Converted annotated video to H.264")
        
        db.session.commit()
        print(f"\n✅ Converted {converted_count} videos to H.264")


if __name__ == "__main__":
    main()