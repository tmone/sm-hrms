#!/usr/bin/env python3
"""Test the simple recognition fix"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

print("🧪 Testing Simple Recognition Fix\n")

# Test 1: Load model
print("1. Testing model loading...")
try:
    from processing.simple_recognition_fix import get_recognition_model
    
    model = get_recognition_model()
    print(f"   Model loaded: {model.loaded}")
    
    if model.loaded:
        print(f"   [OK] Model components:")
        print(f"      - Model: {'[CHECK]' if model.model is not None else '✗'}")
        print(f"      - Scaler: {'[CHECK]' if model.scaler is not None else '✗'}")
        print(f"      - Label encoder: {'[CHECK]' if model.label_encoder is not None else '✗'}")
        print(f"      - Person mapping: {len(model.person_mapping)} persons")
        
        if model.person_mapping:
            print(f"\n   Persons in model:")
            for idx, person_id in sorted(model.person_mapping.items())[:5]:
                print(f"      {idx}: {person_id}")
                
except Exception as e:
    print(f"   [ERROR] Failed to load model: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Test recognition on existing person image
print("\n2. Testing recognition on known person...")

# Find a test image
test_person = "PERSON-0001"
person_dir = Path(f"processing/outputs/persons/{test_person}")

if person_dir.exists():
    images = list(person_dir.glob("*.jpg"))
    if images:
        test_image_path = images[0]
        print(f"   Using image: {test_image_path}")
        
        # Load and test
        img = cv2.imread(str(test_image_path))
        if img is not None:
            print(f"   Image shape: {img.shape}")
            
            try:
                from processing.simple_recognition_fix import recognize_person
                
                # Test with different thresholds
                for threshold in [0.5, 0.6, 0.7, 0.8]:
                    result = recognize_person(img, confidence_threshold=threshold)
                    
                    if result:
                        print(f"\n   Threshold {threshold}:")
                        print(f"      Person ID: {result['person_id']}")
                        print(f"      Confidence: {result['confidence']:.3f}")
                        print(f"      Class idx: {result['class_idx']}")
                        
                        if result['person_id'] == test_person:
                            print(f"      [OK] Correctly recognized!")
                        elif result['person_id'] == 'unknown':
                            print(f"      [WARNING]  Not recognized (confidence too low)")
                        else:
                            print(f"      [ERROR] Misrecognized as {result['person_id']}")
                    else:
                        print(f"\n   Threshold {threshold}: No result")
                        
            except Exception as e:
                print(f"   [ERROR] Recognition failed: {e}")
                import traceback
                traceback.print_exc()
    else:
        print(f"   No images found for {test_person}")
else:
    print(f"   Person directory not found: {person_dir}")

print("\n[OK] Test complete!")
print("\n[TIP] If recognition is working, you should see:")
print("   - Model loaded successfully")
print("   - Correct person IDs returned with good confidence")
print("   - Lower thresholds = more recognitions but less accurate")
print("   - Higher thresholds = fewer recognitions but more accurate")