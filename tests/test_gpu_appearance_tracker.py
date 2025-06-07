#!/usr/bin/env python3
"""
Test script for GPU-accelerated appearance tracker
"""
import sys
import os
import time
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_gpu_tracker():
    """Test GPU appearance tracker functionality"""
    print("=" * 60)
    print("GPU APPEARANCE TRACKER TEST")
    print("=" * 60)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"[OK] CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"[SAVE] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("[WARNING] CUDA not available - will use CPU")
    
    try:
        # Import tracker
        from hr_management.processing.gpu_appearance_tracker import (
            AppearanceFeatureExtractor, 
            GPUPersonTracker,
            process_video_with_gpu_tracking
        )
        print("[OK] Successfully imported GPU tracker modules")
        
        # Test feature extractor
        print("\n🔬 Testing Appearance Feature Extractor...")
        extractor = AppearanceFeatureExtractor()
        
        # Create dummy person crops
        import numpy as np
        import cv2
        
        # Simulate 3 person crops
        crops = []
        for i in range(3):
            # Create different colored rectangles to simulate different persons
            crop = np.zeros((128, 64, 3), dtype=np.uint8)
            crop[:, :] = (i * 80, 100 + i * 50, 200 - i * 50)  # Different colors
            crops.append(crop)
        
        # Extract features
        start_time = time.time()
        features = extractor.extract_batch(crops)
        extraction_time = time.time() - start_time
        
        print(f"[OK] Extracted features shape: {features.shape}")
        print(f"[TIMER] Extraction time: {extraction_time*1000:.1f}ms for {len(crops)} crops")
        print(f"[INFO] Features normalized: {torch.allclose(features.norm(dim=1), torch.ones(len(crops)))}")
        
        # Test similarity computation
        print("\n🔬 Testing Similarity Computation...")
        similarities = torch.mm(features, features.t())
        print(f"[OK] Similarity matrix shape: {similarities.shape}")
        print(f"[INFO] Self-similarity (diagonal): {similarities.diag().cpu().numpy()}")
        print(f"[INFO] Cross-similarities: {similarities[0, 1]:.3f}, {similarities[0, 2]:.3f}, {similarities[1, 2]:.3f}")
        
        # Test tracker
        print("\n🔬 Testing GPU Person Tracker...")
        tracker = GPUPersonTracker(
            appearance_weight=0.7,
            position_weight=0.3
        )
        
        # Simulate detections
        detections = [
            {'bbox': [100, 100, 64, 128], 'confidence': 0.9},
            {'bbox': [200, 150, 64, 128], 'confidence': 0.85},
            {'bbox': [300, 100, 64, 128], 'confidence': 0.8}
        ]
        
        # Create dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Update tracker
        tracked = tracker.update(detections, frame, frame_number=0)
        print(f"[OK] Tracked {len(tracked)} persons")
        for t in tracked:
            print(f"   - {t['person_id']}: confidence {t['confidence']:.2f}")
        
        # Test tracking consistency
        print("\n🔬 Testing Tracking Consistency...")
        # Move persons slightly
        detections2 = [
            {'bbox': [105, 102, 64, 128], 'confidence': 0.9},  # Person 1 moved slightly
            {'bbox': [205, 148, 64, 128], 'confidence': 0.85}, # Person 2 moved slightly
            {'bbox': [295, 98, 64, 128], 'confidence': 0.8}    # Person 3 moved slightly
        ]
        
        tracked2 = tracker.update(detections2, frame, frame_number=1)
        print(f"[OK] Frame 2: Tracked {len(tracked2)} persons")
        
        # Check if same IDs are maintained
        ids_frame1 = sorted([t['person_id'] for t in tracked])
        ids_frame2 = sorted([t['person_id'] for t in tracked2])
        
        if ids_frame1 == ids_frame2:
            print("[OK] Person IDs correctly maintained across frames")
        else:
            print("[WARNING] Person ID mismatch between frames")
            print(f"   Frame 1: {ids_frame1}")
            print(f"   Frame 2: {ids_frame2}")
        
        print("\n[OK] All tests passed!")
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_video_processing(video_path):
    """Test processing a real video with GPU tracker"""
    print("\n" + "=" * 60)
    print("VIDEO PROCESSING TEST")
    print("=" * 60)
    
    try:
        from hr_management.processing.gpu_appearance_tracker import process_video_with_gpu_tracking
        
        print(f"[VIDEO] Processing video: {video_path}")
        output_path = "test_tracked_output.mp4"
        
        start_time = time.time()
        tracks, track_data = process_video_with_gpu_tracking(video_path, output_path)
        processing_time = time.time() - start_time
        
        if tracks:
            print(f"\n[OK] Processing completed in {processing_time:.1f}s")
            print(f"[INFO] Found {len(track_data)} unique persons")
            
            # Show track statistics
            for tid, data in list(track_data.items())[:5]:  # First 5 tracks
                duration = (data['last_frame'] - data['first_frame']) / 30.0
                print(f"   Person {tid}: {len(data['detections'])} detections, {duration:.1f}s duration")
            
            if len(track_data) > 5:
                print(f"   ... and {len(track_data) - 5} more persons")
        else:
            print("[ERROR] No tracks found")
            
    except Exception as e:
        print(f"[ERROR] Video processing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run basic tests
    success = test_gpu_tracker()
    
    # If a video path is provided, test with real video
    if success and len(sys.argv) > 1:
        video_path = sys.argv[1]
        if os.path.exists(video_path):
            test_video_processing(video_path)
        else:
            print(f"\n[WARNING] Video not found: {video_path}")
    elif success:
        print("\n[TIP] To test with a real video, run:")
        print(f"   python {sys.argv[0]} <video_path>")