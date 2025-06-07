#!/usr/bin/env python3
"""
Test script to verify H.264 video chunking functionality
"""
import os
import sys
import subprocess
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from processing.video_chunk_manager import VideoChunkManager
from processing.h264_video_writer import convert_to_h264


def check_video_codec(video_path):
    """Check the codec of a video file"""
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


def test_h264_conversion(test_video_path):
    """Test H.264 conversion"""
    print("\n=== Testing H.264 Conversion ===")
    
    if not os.path.exists(test_video_path):
        print(f"❌ Test video not found: {test_video_path}")
        return False
        
    # Check original codec
    original_codec = check_video_codec(test_video_path)
    print(f"Original video codec: {original_codec}")
    
    # Convert to H.264
    output_path = Path(test_video_path).parent / f"{Path(test_video_path).stem}_h264.mp4"
    print(f"Converting to H.264...")
    
    h264_path = convert_to_h264(test_video_path, output_path)
    
    # Check converted codec
    converted_codec = check_video_codec(h264_path)
    print(f"Converted video codec: {converted_codec}")
    
    if converted_codec == 'h264':
        print("✅ H.264 conversion successful!")
        return True
    else:
        print("❌ H.264 conversion failed!")
        return False


def test_video_chunking(test_video_path):
    """Test video chunking with H.264 output"""
    print("\n=== Testing Video Chunking ===")
    
    if not os.path.exists(test_video_path):
        print(f"❌ Test video not found: {test_video_path}")
        return False
        
    # Create chunk manager
    chunk_manager = VideoChunkManager(chunk_duration=30)
    
    # Check if video should be chunked
    duration = chunk_manager.get_video_duration(test_video_path)
    print(f"Video duration: {duration:.1f}s")
    
    should_chunk = chunk_manager.should_chunk_video(test_video_path, threshold=60)
    print(f"Should chunk (>60s): {should_chunk}")
    
    if should_chunk:
        # Create output directory
        output_dir = Path("test_chunks_output")
        output_dir.mkdir(exist_ok=True)
        
        # Split video
        print("Splitting video into chunks...")
        chunk_paths = chunk_manager.split_video_to_chunks(test_video_path, str(output_dir))
        
        print(f"Created {len(chunk_paths)} chunks:")
        for i, chunk_path in enumerate(chunk_paths):
            codec = check_video_codec(chunk_path)
            print(f"  Chunk {i}: {Path(chunk_path).name} - codec: {codec}")
            
        # Test merging annotated videos
        print("\nTesting annotated video merge...")
        
        # Create dummy parent video object
        class DummyVideo:
            id = 1
            filename = Path(test_video_path).name
            
        parent_video = DummyVideo()
        
        # Test merge (would need actual annotated videos)
        merged_path = chunk_manager._merge_annotated_videos(parent_video, chunk_paths[:2])
        
        if merged_path and os.path.exists(merged_path):
            merged_codec = check_video_codec(merged_path)
            print(f"Merged video codec: {merged_codec}")
            
            if merged_codec == 'h264':
                print("✅ Merged video is H.264!")
            else:
                print("❌ Merged video is not H.264!")
                
        return True
    else:
        print("Video too short for chunking test")
        return False


def main():
    """Run tests"""
    print("H.264 Video Chunking Test Suite")
    print("================================")
    
    # Find a test video
    test_videos = [
        "static/uploads/test_video.mp4",
        "static/uploads/sample_video.mp4",
        "static/uploads/demo_video.mp4"
    ]
    
    test_video = None
    for video_path in test_videos:
        if os.path.exists(video_path):
            test_video = video_path
            break
            
    if not test_video:
        # Look for any video in uploads
        uploads_dir = Path("static/uploads")
        if uploads_dir.exists():
            videos = list(uploads_dir.glob("*.mp4")) + list(uploads_dir.glob("*.avi"))
            if videos:
                test_video = str(videos[0])
                
    if not test_video:
        print("❌ No test video found in static/uploads/")
        print("Please upload a video first.")
        return
        
    print(f"Using test video: {test_video}")
    
    # Run tests
    test_h264_conversion(test_video)
    test_video_chunking(test_video)
    
    print("\n✅ Tests completed!")


if __name__ == "__main__":
    main()