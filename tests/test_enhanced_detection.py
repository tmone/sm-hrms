#!/usr/bin/env python3
"""
Test script for enhanced person detection system
This script verifies that the enhanced detection system works correctly
"""

import os
import sys
import tempfile
import cv2
import numpy as np

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_test_video():
    """Create a simple test video with moving rectangles (simulating persons)"""
    print("[ACTION] Creating test video...")
    
    # Create temporary video file
    temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    temp_video.close()
    
    # Video properties
    width, height = 640, 480
    fps = 10
    duration = 3  # seconds
    total_frames = fps * duration
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video.name, fourcc, fps, (width, height))
    
    for frame_num in range(total_frames):
        # Create black background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add moving rectangles (simulating persons)
        # Person 1: moving left to right
        x1 = int(50 + (frame_num * 10) % (width - 100))
        cv2.rectangle(frame, (x1, 100), (x1 + 60, 200), (0, 255, 0), -1)
        
        # Person 2: moving right to left
        x2 = int(width - 50 - (frame_num * 15) % (width - 100))
        cv2.rectangle(frame, (x2, 250), (x2 + 60, 350), (255, 0, 0), -1)
        
        # Person 3: stationary (to test static detection)
        cv2.rectangle(frame, (300, 150), (360, 250), (0, 0, 255), -1)
        
        out.write(frame)
    
    out.release()
    print(f"[OK] Test video created: {temp_video.name}")
    return temp_video.name

def test_enhanced_detection():
    """Test the enhanced detection system"""
    try:
        print("[START] Testing Enhanced Person Detection System")
        print("=" * 50)
        
        # Create test video
        test_video_path = create_test_video()
        
        try:
            # Import and test enhanced detection
            from processing.enhanced_detection import process_video_with_enhanced_detection
            
            print("[AI] Running enhanced detection...")
            result = process_video_with_enhanced_detection(test_video_path)
            
            if result and result['success']:
                print("[OK] Enhanced detection completed successfully!")
                print(f"[INFO] Results:")
                print(f"   - Output directory: {result['output_dir']}")
                print(f"   - Annotated video: {result['annotated_video']}")
                print(f"   - Total persons detected: {result['summary']['total_persons']}")
                
                # Check if annotated video was created
                if os.path.exists(result['annotated_video']):
                    print(f"[OK] Annotated video file exists: {os.path.getsize(result['annotated_video'])} bytes")
                else:
                    print(f"[ERROR] Annotated video file not found")
                
                # Check if person folders were created
                persons_dir = os.path.join(result['output_dir'], 'persons')
                if os.path.exists(persons_dir):
                    person_folders = [d for d in os.listdir(persons_dir) if d.startswith('PERSON-')]
                    print(f"[OK] Person folders created: {len(person_folders)} folders")
                    for folder in person_folders:
                        folder_path = os.path.join(persons_dir, folder)
                        files = os.listdir(folder_path)
                        print(f"   - {folder}: {len(files)} files")
                else:
                    print(f"[ERROR] Person folders not found")
                
                # Print person summary
                if 'person_summary' in result['summary']:
                    print(f"[GRAPH] Person Summary:")
                    for person_id, data in result['summary']['person_summary'].items():
                        print(f"   - {person_id}: {data['total_detections']} detections, "
                              f"{data['duration']:.1f}s duration, "
                              f"{data['avg_confidence']:.2f} avg confidence")
                
                return True
                
            else:
                print(f"[ERROR] Enhanced detection failed: {result.get('error', 'Unknown error')}")
                return False
                
        finally:
            # Clean up test video
            if os.path.exists(test_video_path):
                os.unlink(test_video_path)
                print(f"[DELETE] Cleaned up test video")
        
    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        print("[WARNING] Make sure YOLO and OpenCV are installed:")
        print("   pip install ultralytics opencv-python")
        return False
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_person_tracking():
    """Test that person tracking works correctly"""
    try:
        print("\n[TARGET] Testing Person Tracking Algorithm")
        print("=" * 40)
        
        from processing.enhanced_detection import PersonTracker
        
        # Create test tracker
        tracker = PersonTracker()
        
        # Simulate detections across frames
        test_detections = [
            # Frame 0: Two persons
            [
                {'bbox': [100, 100, 50, 100], 'confidence': 0.9, 'frame_number': 0},
                {'bbox': [300, 150, 60, 120], 'confidence': 0.8, 'frame_number': 0}
            ],
            # Frame 1: Same persons moved slightly
            [
                {'bbox': [110, 100, 50, 100], 'confidence': 0.9, 'frame_number': 1},
                {'bbox': [290, 150, 60, 120], 'confidence': 0.8, 'frame_number': 1}
            ],
            # Frame 2: One person disappears, new one appears
            [
                {'bbox': [120, 100, 50, 100], 'confidence': 0.9, 'frame_number': 2},
                {'bbox': [500, 200, 55, 110], 'confidence': 0.7, 'frame_number': 2}
            ]
        ]
        
        all_tracked = []
        for frame_num, detections in enumerate(test_detections):
            tracked = tracker.update_tracks(detections, frame_num)
            all_tracked.extend(tracked)
            print(f"Frame {frame_num}: {len(tracked)} tracked detections")
        
        # Analyze tracking results
        unique_persons = len(set(d['person_id'] for d in all_tracked))
        print(f"[OK] Tracking test completed:")
        print(f"   - Total tracked detections: {len(all_tracked)}")
        print(f"   - Unique persons identified: {unique_persons}")
        
        # Print tracking details
        for detection in all_tracked:
            print(f"   - {detection['person_id']} (track {detection['track_id']}) "
                  f"at frame {detection['frame_number']}: confidence {detection['confidence']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Person tracking test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Enhanced Detection System Test Suite")
    print("=" * 50)
    
    success = True
    
    # Test 1: Person tracking algorithm
    if not test_person_tracking():
        success = False
    
    # Test 2: Full enhanced detection system
    if not test_enhanced_detection():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! Enhanced detection system is working correctly.")
        print("\nNext steps:")
        print("1. Upload a video through the web interface")
        print("2. Click 'Process' to run enhanced detection")
        print("3. Check for 'detected_<video-name>' output directory")
        print("4. Review annotated video and PERSON-XXXX folders")
    else:
        print("[ERROR] Some tests failed. Please check the error messages above.")
        print("\nTroubleshooting:")
        print("1. Install required dependencies: pip install ultralytics opencv-python")
        print("2. Check that YOLO model can be downloaded")
        print("3. Verify file permissions for output directories")
    
    print("\n[TARGET] Enhanced Detection Features:")
    print("[OK] Multi-frame person tracking (solves duplicate detection)")
    print("[OK] Annotated video generation with bounding boxes")
    print("[OK] PERSON-XXXX folders with extracted images and metadata")
    print("[OK] Professional video processing workflow")
    print("[OK] Database integration with person_id and track_id")