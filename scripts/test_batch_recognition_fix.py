#!/usr/bin/env python3
"""
Test script to verify batch recognition fix
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import json
import cv2
import numpy as np

def test_batch_recognition():
    """Test batch recognition with corrected method"""
    
    # Import recognition module
    try:
        from hr_management.processing.person_recognition_inference_simple import PersonRecognitionInferenceSimple
        
        print("Loading recognition model...")
        recognizer = PersonRecognitionInferenceSimple(
            model_name='person_model_svm_20250607_181818',
            confidence_threshold=0.5
        )
        print("✓ Model loaded successfully")
        
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Test with a specific image
    test_person = "PERSON-0022"
    test_image = "75d7a48d-3c16-4949-80bd-6ae052f35fee.jpg"
    
    persons_dir = Path("processing/outputs/persons")
    img_path = persons_dir / test_person / test_image
    
    if not img_path.exists():
        print(f"Test image not found: {img_path}")
        # Try any image from PERSON-0022
        person_dir = persons_dir / test_person
        if person_dir.exists():
            images = list(person_dir.glob("*.jpg"))
            if images:
                img_path = images[0]
                print(f"Using alternative image: {img_path.name}")
            else:
                print("No images found in PERSON-0022")
                return
        else:
            print(f"Person folder not found: {test_person}")
            return
    
    print(f"\nTesting recognition on: {img_path}")
    print("-" * 80)
    
    try:
        # Test the corrected method
        result = recognizer.process_cropped_image(str(img_path))
        
        print(f"Raw result: {json.dumps(result, indent=2)}")
        
        if result.get('persons') and len(result['persons']) > 0:
            person_result = result['persons'][0]
            predicted_id = person_result.get('person_id', 'unknown')
            confidence = person_result.get('confidence', 0.0)
            
            print(f"\n✓ Recognition successful!")
            print(f"  Predicted: {predicted_id}")
            print(f"  Confidence: {confidence:.3f}")
            
            if 'all_probabilities' in person_result:
                print("\n  All probabilities:")
                for pid, prob in sorted(person_result['all_probabilities'].items(), 
                                       key=lambda x: x[1], reverse=True):
                    print(f"    {pid}: {prob:.3f}")
        else:
            print("✗ No person detected in image")
            
    except Exception as e:
        print(f"✗ Error during recognition: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_batch_recognition()