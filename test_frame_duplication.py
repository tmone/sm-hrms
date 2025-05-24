#!/usr/bin/env python
"""
Test script to check for frame duplication in processed videos
"""
import cv2
import numpy as np
import sys
from pathlib import Path

def analyze_video_for_duplicates(video_path, sample_rate=30):
    """
    Analyze video for duplicate frames
    
    Args:
        video_path: Path to video file
        sample_rate: Check every Nth frame (default 30 = once per second at 30fps)
    """
    print(f"Analyzing video: {video_path}")
    print(f"Checking every {sample_rate} frames for duplicates...")
    print("-" * 60)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {total_frames} frames @ {fps:.1f} fps")
    print(f"Duration: {total_frames/fps:.1f} seconds")
    print()
    
    # Store frame hashes to detect duplicates
    frame_hashes = []
    duplicate_count = 0
    frame_count = 0
    last_frame = None
    consecutive_duplicates = 0
    max_consecutive = 0
    duplicate_positions = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only check every Nth frame for performance
        if frame_count % sample_rate == 0:
            # Convert to grayscale for faster comparison
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Compute frame hash
            frame_hash = hash(gray.tobytes())
            
            # Check if this is a duplicate of the last checked frame
            if last_frame is not None:
                # Calculate difference between frames
                diff = cv2.absdiff(gray, last_frame)
                mean_diff = np.mean(diff)
                
                # If frames are nearly identical (very small difference)
                if mean_diff < 1.0:  # Threshold for considering frames identical
                    duplicate_count += 1
                    consecutive_duplicates += 1
                    duplicate_positions.append(frame_count)
                    
                    time_sec = frame_count / fps
                    print(f"⚠️  Duplicate frame detected at {time_sec:.1f}s (frame {frame_count})")
                else:
                    if consecutive_duplicates > max_consecutive:
                        max_consecutive = consecutive_duplicates
                    consecutive_duplicates = 0
            
            last_frame = gray.copy()
        
        frame_count += 1
        
        # Progress indicator
        if frame_count % 300 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}%", end='\r')
    
    cap.release()
    
    print("\n" + "=" * 60)
    print("Analysis Complete:")
    print("=" * 60)
    print(f"Total frames analyzed: {frame_count // sample_rate}")
    print(f"Duplicate frames found: {duplicate_count}")
    print(f"Duplicate rate: {(duplicate_count / (frame_count // sample_rate)) * 100:.1f}%")
    print(f"Max consecutive duplicates: {max_consecutive}")
    
    if duplicate_count > 0:
        print("\n⚠️  WARNING: Video has duplicate frames!")
        print("This can cause stuttering during playback.")
        if len(duplicate_positions) <= 10:
            print(f"Duplicate positions (frames): {duplicate_positions}")
        else:
            print(f"First 10 duplicate positions: {duplicate_positions[:10]}")
    else:
        print("\n✅ No duplicate frames detected! Video should play smoothly.")
    
    # Estimate file size impact
    if duplicate_count > 0:
        print(f"\n💾 Estimated size impact: ~{duplicate_count * 100 / frame_count:.1f}% larger than necessary")

def compare_videos(original_path, processed_path):
    """Compare original and processed videos"""
    print("\n" + "=" * 70)
    print("Comparing Original vs Processed Video")
    print("=" * 70)
    
    # Get video info
    cap1 = cv2.VideoCapture(str(original_path))
    cap2 = cv2.VideoCapture(str(processed_path))
    
    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Cannot open one or both videos")
        return
    
    # Compare properties
    props = [
        ('Frame Count', cv2.CAP_PROP_FRAME_COUNT),
        ('FPS', cv2.CAP_PROP_FPS),
        ('Width', cv2.CAP_PROP_FRAME_WIDTH),
        ('Height', cv2.CAP_PROP_FRAME_HEIGHT)
    ]
    
    print(f"{'Property':<15} {'Original':<15} {'Processed':<15} {'Match':<10}")
    print("-" * 55)
    
    for prop_name, prop_id in props:
        val1 = cap1.get(prop_id)
        val2 = cap2.get(prop_id)
        match = "✅" if abs(val1 - val2) < 0.1 else "❌"
        print(f"{prop_name:<15} {val1:<15.1f} {val2:<15.1f} {match:<10}")
    
    cap1.release()
    cap2.release()
    
    # Get file sizes
    size1 = Path(original_path).stat().st_size / (1024 * 1024)
    size2 = Path(processed_path).stat().st_size / (1024 * 1024)
    size_ratio = size2 / size1
    
    print(f"\nFile Sizes:")
    print(f"  Original:  {size1:.1f} MB")
    print(f"  Processed: {size2:.1f} MB")
    print(f"  Ratio:     {size_ratio:.2f}x")
    
    if size_ratio > 3:
        print("  ⚠️  WARNING: Processed video is much larger than original!")
        print("  This suggests codec/compression issues.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_frame_duplication.py <video_path> [original_video_path]")
        print("\nExamples:")
        print("  python test_frame_duplication.py processing/outputs/video_annotated_*.mp4")
        print("  python test_frame_duplication.py processed.mp4 original.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Analyze for duplicates
    analyze_video_for_duplicates(video_path)
    
    # If original video provided, compare them
    if len(sys.argv) > 2:
        original_path = sys.argv[2]
        compare_videos(original_path, video_path)