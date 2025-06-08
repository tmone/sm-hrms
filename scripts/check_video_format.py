#!/usr/bin/env python3
"""
Check video format and codec
"""

import os
import subprocess
import sys

def check_video(video_path):
    """Check video format using ffprobe"""
    if not os.path.exists(video_path):
        print(f"[ERROR] File not found: {video_path}")
        return
        
    print(f"Checking video: {video_path}")
    print(f"File size: {os.path.getsize(video_path) / (1024*1024):.1f} MB")
    
    # Get video info
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,width,height,r_frame_rate',
        '-of', 'json',
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        import json
        info = json.loads(result.stdout)
        
        if 'streams' in info and info['streams']:
            stream = info['streams'][0]
            print(f"Codec: {stream.get('codec_name', 'unknown')}")
            print(f"Resolution: {stream.get('width', '?')}x{stream.get('height', '?')}")
            print(f"Frame rate: {stream.get('r_frame_rate', '?')}")
            
            # Check if web-compatible
            web_codecs = ['h264', 'vp8', 'vp9']
            codec = stream.get('codec_name', '').lower()
            if codec in web_codecs:
                print("[OK] Web-compatible codec")
            else:
                print(f"[WARNING] Non-web codec: {codec}")
                
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] FFprobe error: {e.stderr}")
    except Exception as e:
        print(f"[ERROR] Error: {e}")

if __name__ == "__main__":
    # Check the specific video
    video_path = "static/uploads/b377f011-e760-4b43-b275-c7c75566f08a_multichunk_annotated_20250606_052358_web.mp4"
    check_video(video_path)
    
    # Also check if it's a valid MP4
    print("\nChecking MP4 structure:")
    cmd = ['ffmpeg', '-v', 'error', '-i', video_path, '-f', 'null', '-']
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print("[OK] Valid MP4 structure")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Invalid MP4: {e.stderr.decode()}")