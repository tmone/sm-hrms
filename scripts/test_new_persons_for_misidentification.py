#!/usr/bin/env python3
"""
Test all NEW person folders to check if they contain images of trained persons
that were incorrectly assigned new person codes
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import json
import cv2
import numpy as np
from collections import defaultdict
import time

def test_new_persons_for_trained_faces():
    """Test if new person folders contain faces that should be recognized as trained persons"""
    
    # The 9 trained persons
    trained_persons = ['PERSON-0001', 'PERSON-0002', 'PERSON-0007', 'PERSON-0008', 
                      'PERSON-0010', 'PERSON-0017', 'PERSON-0019', 'PERSON-0020', 'PERSON-0021']
    
    persons_dir = Path("processing/outputs/persons")
    
    # Get all person folders
    all_persons = sorted([d.name for d in persons_dir.iterdir() 
                         if d.is_dir() and d.name.startswith('PERSON-')])
    
    # Get NEW persons (not in trained list)
    new_persons = [p for p in all_persons if p not in trained_persons]
    
    print(f"Found {len(new_persons)} NEW person folders (not in trained model)")
    print(f"These should all be 'unknown' persons, but let's check if any are actually trained persons...")
    print("=" * 80)
    
    # Import recognition module with direct model loading
    try:
        # Try to load model using simple inference
        from processing.simple_person_recognition_inference import SimplePersonRecognitionInference
        
        print("Loading recognition model...")
        recognizer = SimplePersonRecognitionInference(
            model_name='person_model_svm_20250607_181818',
            model_dir=Path('models/person_recognition')
        )
        print(f"✓ Model loaded successfully")
        print(f"  Model can recognize: {trained_persons}")
        print()
        
    except Exception as e:
        print(f"✗ Failed to load recognition model: {e}")
        print("\nTrying alternative loading method...")
        
        try:
            # Try web UI recognition wrapper
            from processing.web_ui_recognition_wrapper import WebUIRecognitionWrapper
            recognizer = WebUIRecognitionWrapper()
            print("✓ Loaded via WebUI wrapper")
        except Exception as e2:
            print(f"✗ Also failed with WebUI wrapper: {e2}")
            return
    
    # Test each NEW person folder
    misidentified_persons = defaultdict(list)
    total_tested = 0
    
    print("\nTesting NEW person folders for misidentification:")
    print("-" * 80)
    
    for idx, person_id in enumerate(new_persons):
        person_folder = persons_dir / person_id
        images = list(person_folder.glob("*.jpg"))[:5]  # Test up to 5 images per person
        
        if not images:
            continue
            
        print(f"\n[{idx+1}/{len(new_persons)}] Testing {person_id} ({len(list(person_folder.glob('*.jpg')))} total images):")
        
        predictions = defaultdict(int)
        
        for img_path in images:
            try:
                # Read image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Convert to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Get prediction
                result = recognizer.recognize_person(img_rgb)
                predicted_id = result.get('person_id', 'unknown')
                confidence = result.get('confidence', 0.0)
                
                predictions[predicted_id] += 1
                total_tested += 1
                
                # If predicted as a trained person, this is a misidentification
                if predicted_id != 'unknown' and predicted_id in trained_persons:
                    misidentified_persons[person_id].append({
                        'predicted_as': predicted_id,
                        'confidence': confidence,
                        'image': img_path.name
                    })
                    print(f"  ⚠️  {img_path.name} -> Predicted as {predicted_id} (conf: {confidence:.3f})")
                else:
                    print(f"  ✓ {img_path.name} -> {predicted_id} (conf: {confidence:.3f})")
                    
            except Exception as e:
                print(f"  ✗ Error processing {img_path.name}: {e}")
        
        # Summary for this person
        if predictions:
            most_common = max(predictions.items(), key=lambda x: x[1])
            print(f"  Summary: Most common prediction: {most_common[0]} ({most_common[1]}/{len(images)} images)")
    
    # Final report
    print("\n" + "=" * 80)
    print("MISIDENTIFICATION REPORT:")
    print("=" * 80)
    
    if misidentified_persons:
        print(f"\n⚠️  Found {len(misidentified_persons)} NEW persons that might be misidentified:")
        
        for new_person, misidentifications in misidentified_persons.items():
            print(f"\n{new_person}:")
            
            # Count predictions
            pred_counts = defaultdict(int)
            for m in misidentifications:
                pred_counts[m['predicted_as']] += 1
            
            for trained_person, count in pred_counts.items():
                avg_conf = np.mean([m['confidence'] for m in misidentifications if m['predicted_as'] == trained_person])
                print(f"  - Predicted as {trained_person}: {count} times (avg confidence: {avg_conf:.3f})")
                print(f"    This NEW person might actually be {trained_person}!")
    else:
        print("\n✓ No misidentifications found!")
        print("  All NEW persons are correctly identified as 'unknown'")
        print("  This suggests the system is working correctly")
    
    print(f"\nTotal images tested: {total_tested}")
    
    # Check for patterns
    print("\n" + "=" * 80)
    print("ANALYSIS:")
    print("=" * 80)
    print("If NEW persons are being created for trained persons, possible causes:")
    print("1. Model not loading correctly during video processing")
    print("2. Face quality/angle issues causing recognition failure")
    print("3. Confidence threshold too high")
    print("4. Race condition in person ID assignment")

if __name__ == "__main__":
    test_new_persons_for_trained_faces()