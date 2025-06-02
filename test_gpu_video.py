#!/usr/bin/env python3
"""
Test script to verify GPU usage in video processing
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from hr_management.processing.real_detection import detect_persons_yolo, get_best_available_detector
from hr_management.processing.transformer_detection import detect_persons_yolo_v8

def test_gpu_video_processing():
    """Test GPU usage in video processing"""
    print("=" * 60)
    print("GPU VIDEO PROCESSING TEST")
    print("=" * 60)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available")
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("❌ CUDA not available - will use CPU")
    
    # Check best detector
    detector = get_best_available_detector()
    print(f"🤖 Best available detector: {detector}")
    
    # Create a test video (or use existing one)
    test_video = "yolov8n.pt"  # This will just test model loading
    
    if os.path.exists(test_video):
        print(f"\n🎬 Testing with: {test_video}")
        
        try:
            print("\n🔍 Testing YOLO detection...")
            # This will test the model loading with GPU configuration
            from hr_management.processing.real_detection import _model_cache
            from ultralytics import YOLO
            
            # Clear cache to force reloading
            _model_cache.clear()
            
            # Load model and check device
            model_path = "yolov8n.pt"
            model = YOLO(model_path)
            
            if torch.cuda.is_available():
                model.to('cuda')
                print(f"🚀 Model moved to GPU")
                
                # Test with dummy image
                import numpy as np
                dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                
                print("🔄 Running GPU inference test...")
                results = model(dummy_frame, device='cuda', verbose=False)
                print("✅ GPU inference successful!")
                
                # Check GPU memory usage
                print(f"💾 GPU Memory used: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
                
            else:
                print("⚠️ Model will use CPU")
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
    else:
        print(f"⚠️ Test video not found: {test_video}")
    
    print("\n" + "=" * 60)
    print("✅ GPU VIDEO PROCESSING TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_gpu_video_processing()
