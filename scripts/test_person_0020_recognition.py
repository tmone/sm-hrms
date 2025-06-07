#!/usr/bin/env python3
"""
Test recognition specifically for PERSON-0020
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import cv2

def test_person_0020():
    """Test if PERSON-0020 can be recognized"""
    print("Testing recognition for PERSON-0020\n")
    
    # Get test image
    person_dir = Path("processing/outputs/persons/PERSON-0020")
    test_images = list(person_dir.glob("*.jpg"))[:3]
    
    if not test_images:
        print("ERROR: No test images found")
        return
    
    print(f"Found {len(test_images)} test images")
    
    # Test with different models
    models_to_test = [
        "refined_quick_20250606_054446",  # The one used in GPU detection
        "person_model_svm_20250607_181818",  # Latest model that includes PERSON-0020
        "refined_quick_20250529_165210"  # Another refined model
    ]
    
    from hr_management.processing.person_recognition_inference_simple import PersonRecognitionInferenceSimple
    
    for model_name in models_to_test:
        print(f"\nTesting with model: {model_name}")
        
        try:
            recognizer = PersonRecognitionInferenceSimple(model_name, confidence_threshold=0.5)
            
            # Test each image
            for i, img_path in enumerate(test_images):
                print(f"\n   Image {i+1}: {img_path.name}")
                
                # Test with file path (like UI does)
                result = recognizer.process_cropped_image(str(img_path))
                
                if result and result.get('persons'):
                    person = result['persons'][0]
                    print(f"   SUCCESS: Predicted: {person['person_id']} (confidence: {person['confidence']:.2%})")
                    
                    # Show top predictions
                    if 'all_probabilities' in result:
                        probs = sorted(result['all_probabilities'].items(), 
                                     key=lambda x: x[1], reverse=True)[:3]
                        print("   Top predictions:")
                        for pid, prob in probs:
                            print(f"      - {pid}: {prob:.2%}")
                else:
                    print("   ERROR: No prediction")
                    
        except Exception as e:
            print(f"   ERROR: Error loading model: {e}")
    
    # Check model metadata
    print("\nChecking model training data for PERSON-0020...")
    model_path = Path("models/person_recognition/person_model_svm_20250607_181818")
    dataset_path = model_path.parent / "datasets"
    
    # Find the dataset used
    for ds_dir in dataset_path.glob("*"):
        if ds_dir.is_dir():
            ds_info_path = ds_dir / "dataset_info.json"
            if ds_info_path.exists():
                import json
                with open(ds_info_path) as f:
                    ds_info = json.load(f)
                
                if "PERSON-0020" in ds_info.get('persons', {}):
                    person_data = ds_info['persons']['PERSON-0020']
                    print(f"\n   Dataset: {ds_dir.name}")
                    print(f"   Images used: {len(person_data.get('images', []))}")
                    print(f"   Features: {person_data.get('feature_dim', 'Unknown')}")

if __name__ == "__main__":
    test_person_0020()