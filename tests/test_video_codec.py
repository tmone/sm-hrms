#!/usr/bin/env python
"""
Test video codec and verify browser-compatible video generation
"""
import cv2
import numpy as np
import os
from pathlib import Path

def test_video_codecs():
    """Test different video codecs for browser compatibility"""
    print("=" * 60)
    print("Testing Video Codecs for Browser Compatibility")
    print("=" * 60)
    print()
    
    # Create test video frame
    width, height = 640, 480
    fps = 30
    duration = 3  # 3 seconds
    total_frames = fps * duration
    
    # Test different codecs
    codecs_to_test = [
        # (fourcc, extension, description)
        ('H264', '.mp4', 'H.264/AVC in MP4'),
        ('h264', '.mp4', 'H.264 lowercase in MP4'),
        ('avc1', '.mp4', 'H.264/AVC (Apple) in MP4'),
        ('x264', '.mp4', 'x264 encoder in MP4'),
        ('mp4v', '.mp4', 'MPEG-4 in MP4'),
        ('MJPG', '.avi', 'Motion JPEG in AVI'),
        ('XVID', '.avi', 'Xvid in AVI'),
    ]
    
    results = []
    
    for fourcc_str, ext, description in codecs_to_test:
        output_file = f"test_codec_{fourcc_str}{ext}"
        print(f"\nTesting: {description}")
        print(f"  Output: {output_file}")
        
        try:
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
            
            if not out.isOpened():
                print(f"  ❌ Failed to open video writer")
                results.append((fourcc_str, ext, "Failed to open", 0))
                continue
            
            # Write test frames
            for i in range(total_frames):
                # Create frame with moving box
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Add moving rectangle
                x = int((i / total_frames) * (width - 100))
                cv2.rectangle(frame, (x, 200), (x + 100, 280), (0, 255, 0), -1)
                
                # Add frame number
                cv2.putText(frame, f"Frame {i+1}/{total_frames}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Add codec info
                cv2.putText(frame, f"Codec: {fourcc_str}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                out.write(frame)
            
            out.release()
            
            # Check file size
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file) / 1024  # KB
                print(f"  ✅ Success! File size: {file_size:.1f} KB")
                
                # Verify file can be read
                test_cap = cv2.VideoCapture(output_file)
                if test_cap.isOpened():
                    frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    test_cap.release()
                    print(f"  ✅ Readable! Frames: {frame_count}")
                    results.append((fourcc_str, ext, "Success", file_size))
                else:
                    print(f"  ⚠️  File created but cannot be read")
                    results.append((fourcc_str, ext, "Not readable", file_size))
            else:
                print(f"  ❌ No file created")
                results.append((fourcc_str, ext, "No file", 0))
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append((fourcc_str, ext, f"Error: {e}", 0))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - Browser Compatible Codecs:")
    print("=" * 60)
    
    print("\n📹 Recommended for browser compatibility:")
    for fourcc_str, ext, status, size in results:
        if status == "Success" and ext == ".mp4":
            print(f"  ✅ {fourcc_str} in MP4 - {size:.1f} KB")
    
    print("\n🎬 Alternative options:")
    for fourcc_str, ext, status, size in results:
        if status == "Success" and ext == ".avi":
            print(f"  ⚠️  {fourcc_str} in AVI - {size:.1f} KB (may need conversion)")
    
    print("\n❌ Failed codecs:")
    for fourcc_str, ext, status, size in results:
        if status != "Success":
            print(f"  ❌ {fourcc_str} in {ext} - {status}")
    
    # Clean up test files
    print("\nCleaning up test files...")
    for fourcc_str, ext, status, _ in results:
        test_file = f"test_codec_{fourcc_str}{ext}"
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"  Removed: {test_file}")

if __name__ == "__main__":
    test_video_codecs()
    
    # Additional OpenCV info
    print("\n" + "-" * 60)
    print("OpenCV Build Information:")
    print("-" * 60)
    print(f"OpenCV Version: {cv2.__version__}")
    
    # Check video I/O backends
    build_info = cv2.getBuildInformation()
    video_io_start = build_info.find("Video I/O:")
    if video_io_start != -1:
        video_io_end = build_info.find("\n\n", video_io_start)
        video_io_section = build_info[video_io_start:video_io_end]
        print("\nVideo I/O Support:")
        for line in video_io_section.split("\n")[1:]:
            if line.strip():
                print(f"  {line.strip()}")