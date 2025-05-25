#!/usr/bin/env python3
"""
Test script to verify that video processing doesn't duplicate frames
"""
import cv2
import numpy as np
import sys
from pathlib import Path

def check_frame_duplication(video_path, max_check_frames=300):
    """
    Check if a video has duplicated frames
    
    Args:
        video_path: Path to video file
        max_check_frames: Maximum number of frames to check
        
    Returns:
        dict with duplication analysis
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": "Could not open video"}
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"🎬 Analyzing video: {video_path}")
    print(f"📊 Total frames: {total_frames}, FPS: {fps}")
    
    previous_frame = None
    duplicates = []
    frame_count = 0
    frames_to_check = min(total_frames, max_check_frames)
    
    while frame_count < frames_to_check:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to grayscale for comparison
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if previous_frame is not None:
            # Calculate frame difference
            diff = cv2.absdiff(previous_frame, gray_frame)
            mean_diff = np.mean(diff)
            
            # If difference is very small, frames are likely duplicated
            if mean_diff < 0.5:  # Threshold for duplicate detection
                duplicates.append({
                    "frame": frame_count,
                    "diff": mean_diff,
                    "time": frame_count / fps
                })
        
        previous_frame = gray_frame.copy()
        frame_count += 1
        
        # Progress indicator
        if frame_count % 50 == 0:
            print(f"  Checked {frame_count}/{frames_to_check} frames...")
    
    cap.release()
    
    # Analyze duplication pattern
    duplicate_count = len(duplicates)
    duplicate_rate = (duplicate_count / frame_count) * 100 if frame_count > 0 else 0
    
    # Check for periodic duplication (e.g., every second)
    periodic_pattern = False
    if duplicate_count > 5:
        # Check if duplicates occur at regular intervals
        intervals = []
        for i in range(1, len(duplicates)):
            interval = duplicates[i]["frame"] - duplicates[i-1]["frame"]
            intervals.append(interval)
        
        if intervals:
            # Check if most intervals are similar (±2 frames)
            most_common_interval = max(set(intervals), key=intervals.count)
            similar_intervals = sum(1 for i in intervals if abs(i - most_common_interval) <= 2)
            if similar_intervals > len(intervals) * 0.7:
                periodic_pattern = True
                period_seconds = most_common_interval / fps
    
    result = {
        "total_frames": total_frames,
        "frames_checked": frame_count,
        "fps": fps,
        "duplicates_found": duplicate_count,
        "duplication_rate": duplicate_rate,
        "periodic_pattern": periodic_pattern,
        "duplicate_frames": duplicates[:10]  # First 10 duplicates
    }
    
    # Summary
    print(f"\n📊 Analysis Results:")
    print(f"  - Frames checked: {frame_count}")
    print(f"  - Duplicates found: {duplicate_count} ({duplicate_rate:.1f}%)")
    
    if periodic_pattern:
        print(f"  - ⚠️  Periodic duplication detected! Period: ~{period_seconds:.2f}s")
    elif duplicate_count > 0:
        print(f"  - ℹ️  Some duplicates found, but no periodic pattern")
    else:
        print(f"  - ✅ No significant frame duplication detected")
    
    return result

def compare_videos(original_path, processed_path):
    """Compare original and processed videos"""
    print("\n" + "="*60)
    print("COMPARING ORIGINAL AND PROCESSED VIDEOS")
    print("="*60)
    
    # Get file sizes
    original_size = Path(original_path).stat().st_size / (1024 * 1024)
    processed_size = Path(processed_path).stat().st_size / (1024 * 1024)
    
    print(f"\n📁 File sizes:")
    print(f"  - Original:  {original_size:.1f} MB")
    print(f"  - Processed: {processed_size:.1f} MB")
    print(f"  - Reduction: {((original_size - processed_size) / original_size * 100):.1f}%")
    
    # Check original
    print(f"\n🔍 Checking ORIGINAL video:")
    original_result = check_frame_duplication(original_path)
    
    # Check processed
    print(f"\n🔍 Checking PROCESSED video:")
    processed_result = check_frame_duplication(processed_path)
    
    return {
        "original": original_result,
        "processed": processed_result,
        "size_comparison": {
            "original_mb": original_size,
            "processed_mb": processed_size,
            "reduction_percent": ((original_size - processed_size) / original_size * 100)
        }
    }

if __name__ == "__main__":
    if len(sys.argv) == 2:
        # Single video check
        video_path = sys.argv[1]
        result = check_frame_duplication(video_path)
    elif len(sys.argv) == 3:
        # Compare two videos
        original = sys.argv[1]
        processed = sys.argv[2]
        result = compare_videos(original, processed)
    else:
        print("Usage:")
        print("  Check single video:    python test_frame_duplication.py <video_path>")
        print("  Compare two videos:    python test_frame_duplication.py <original> <processed>")
        sys.exit(1)