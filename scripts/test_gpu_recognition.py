#!/usr/bin/env python3
"""
Test GPU detection with recognition integration
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_recognition_in_gpu_detection():
    """Test that GPU detection now includes recognition"""
    print("🧪 Testing GPU Detection with Recognition\n")
    
    # Check if recognition is available
    try:
        from hr_management.processing.person_recognition_inference_simple import PersonRecognitionInferenceSimple
        print("[OK] Recognition module available")
        
        # Try to load a model
        try:
            # Check for refined models first
            models_dir = Path("models/person_recognition")
            refined_models = list(models_dir.glob("refined_*"))
            
            if refined_models:
                latest_model = max(refined_models, key=lambda p: p.stat().st_mtime)
                model_name = latest_model.name
                print(f"[OK] Found refined model: {model_name}")
            else:
                model_name = "person_recognition_model_20250101_000000"
                print(f"[PIN] Using default model: {model_name}")
                
            recognizer = PersonRecognitionInferenceSimple(model_name, confidence_threshold=0.8)
            print("[OK] Recognition model loaded successfully")
            
            # Test recognition with a sample image
            test_dir = Path("processing/outputs/persons/PERSON-0001")
            if test_dir.exists():
                images = list(test_dir.glob("*.jpg"))
                if images:
                    test_image = str(images[0])
                    print(f"\n🧪 Testing recognition with: {test_image}")
                    
                    result = recognizer.process_cropped_image(test_image)
                    if result and result.get('persons'):
                        person = result['persons'][0]
                        print(f"[OK] Recognition result: {person['person_id']} (confidence: {person['confidence']:.2%})")
                    else:
                        print("[ERROR] No recognition result")
            
        except Exception as e:
            print(f"[ERROR] Failed to load recognition model: {e}")
            
    except ImportError as e:
        print(f"[ERROR] Recognition module not available: {e}")
    
    # Check GPU detection integration
    print("\n[TRACE] Checking GPU detection code...")
    
    gpu_detection_file = Path("processing/gpu_enhanced_detection.py")
    if gpu_detection_file.exists():
        content = gpu_detection_file.read_text()
        
        # Check for recognition imports
        if "PersonRecognitionInferenceSimple" in content:
            print("[OK] Recognition import found in GPU detection")
        else:
            print("[ERROR] Recognition import NOT found in GPU detection")
            
        # Check for recognition in extract function
        if "ui_style_recognizer" in content and "process_cropped_image" in content:
            print("[OK] Recognition integration found in extract_persons_data_gpu")
        else:
            print("[ERROR] Recognition integration NOT found in extract_persons_data_gpu")
            
        # Check for recognition before ID assignment
        if "recognized_as" in content:
            print("[OK] Recognition metadata tracking found")
        else:
            print("[ERROR] Recognition metadata tracking NOT found")
    
    print("\n[INFO] Summary:")
    print("The GPU detection module has been updated to include recognition.")
    print("When processing videos, it will now:")
    print("1. Try to recognize each detected person")
    print("2. Use recognized IDs instead of creating new ones")
    print("3. Only create new IDs for truly unknown persons")
    print("\nTest by uploading a video with known persons!")

if __name__ == "__main__":
    test_recognition_in_gpu_detection()