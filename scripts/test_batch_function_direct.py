#!/usr/bin/env python3
"""
Test the batch recognition function directly
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import json
import cv2
import numpy as np
from collections import defaultdict

def test_batch_function():
    """Test batch recognition function directly"""
    
    # Import recognition module
    try:
        from hr_management.processing.person_recognition_inference_simple import PersonRecognitionInferenceSimple
        
        print("Loading recognition model...")
        
        # Load the default model
        config_path = Path("models/person_recognition/config.json")
        with open(config_path) as f:
            config = json.load(f)
        model_name = config.get('default_model', 'person_model_svm_20250607_181818')
        
        recognizer = PersonRecognitionInferenceSimple(
            model_name=model_name,
            confidence_threshold=0.5  # Lower threshold to catch more potential matches
        )
        
        # Get trained persons list
        trained_persons = list(recognizer.person_id_mapping.values())
        print(f"✓ Model loaded: {model_name}")
        print(f"  Trained persons: {trained_persons}")
        
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Test PERSON-0022
    person_ids = ['PERSON-0022']
    persons_dir = Path('processing/outputs/persons')
    
    print(f"\nTesting {len(person_ids)} persons for misidentification...")
    print("=" * 80)
    
    for person_id in person_ids:
        person_dir = persons_dir / person_id
        if not person_dir.exists():
            print(f"✗ {person_id} not found")
            continue
        
        print(f"\nTesting {person_id}...")
        
        # Get all images for this person
        image_files = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
        
        # Test up to 10 images
        test_images = image_files[:10]
        
        predictions = defaultdict(list)
        images_to_move = defaultdict(list)
        
        print(f"  Testing {len(test_images)} images from {person_id} (total: {len(image_files)})")
        
        for img_path in test_images:
            try:
                # Use process_cropped_image method
                result = recognizer.process_cropped_image(str(img_path))
                
                # Extract the first person result
                if result.get('persons') and len(result['persons']) > 0:
                    person_result = result['persons'][0]
                    predicted_id = person_result.get('person_id', 'unknown')
                    confidence = person_result.get('confidence', 0.0)
                else:
                    predicted_id = 'unknown'
                    confidence = 0.0
                
                predictions[predicted_id].append({
                    'image': img_path.name,
                    'confidence': confidence
                })
                
                # Track images that should be moved to trained persons
                if predicted_id != 'unknown' and predicted_id in trained_persons and predicted_id != person_id:
                    images_to_move[predicted_id].append({
                        'image': img_path.name,
                        'confidence': confidence,
                        'full_path': str(img_path)
                    })
                    print(f"    ⚠️  {img_path.name} -> {predicted_id} (conf: {confidence:.3f})")
                else:
                    print(f"    ✓ {img_path.name} -> {predicted_id} (conf: {confidence:.3f})")
                
            except Exception as e:
                print(f"    ✗ Error processing {img_path.name}: {e}")
        
        # Summary
        print(f"\n  Summary for {person_id}:")
        for pred_id, pred_list in predictions.items():
            avg_confidence = np.mean([p['confidence'] for p in pred_list])
            print(f"    {pred_id}: {len(pred_list)} images (avg conf: {avg_confidence:.3f})")
        
        if images_to_move:
            print(f"\n  ⚠️  Misidentified images found!")
            for target_id, images in images_to_move.items():
                print(f"    → {len(images)} images should be moved to {target_id}")
                for img in images[:3]:  # Show first 3
                    print(f"      - {img['image']} (conf: {img['confidence']:.3f})")
                if len(images) > 3:
                    print(f"      ... and {len(images) - 3} more")

if __name__ == "__main__":
    test_batch_function()