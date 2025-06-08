#!/usr/bin/env python3
"""
Test recognition predictions on all exported person images
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

def test_person_predictions():
    """Test predictions on all person images in the persons folder"""
    
    # Import recognition module
    try:
        from hr_management.processing.person_recognition_inference_simple import PersonRecognitionInferenceSimple
        
        # Load model configuration
        config_path = Path("models/person_recognition/config.json")
        with open(config_path) as f:
            config = json.load(f)
        
        model_name = config.get('default_model')
        print(f"Loading model: {model_name}")
        
        # Initialize recognizer
        recognizer = PersonRecognitionInferenceSimple(
            model_name=model_name,
            confidence_threshold=0.7  # Lower threshold for testing
        )
        
        print(f"✓ Model loaded successfully!")
        print(f"  Trained persons: {list(recognizer.person_id_mapping.values())}")
        print()
        
    except Exception as e:
        print(f"✗ Failed to load recognition model: {e}")
        return
    
    # Process all person folders
    persons_dir = Path("processing/outputs/persons")
    if not persons_dir.exists():
        print(f"✗ Persons directory not found: {persons_dir}")
        return
    
    # Statistics
    total_images = 0
    correct_predictions = 0
    wrong_predictions = 0
    unknown_predictions = 0
    prediction_details = defaultdict(lambda: {
        'total': 0,
        'correct': 0,
        'wrong': 0,
        'unknown': 0,
        'wrong_as': defaultdict(int),
        'confidence_scores': []
    })
    
    # Test each person folder
    person_folders = sorted([d for d in persons_dir.iterdir() if d.is_dir() and d.name.startswith('PERSON-')])
    
    print(f"Found {len(person_folders)} person folders to test")
    print("=" * 80)
    
    for person_folder in person_folders:
        person_id = person_folder.name
        print(f"\nTesting {person_id}:")
        
        # Get all image files
        image_files = list(person_folder.glob("*.jpg")) + list(person_folder.glob("*.png"))
        if not image_files:
            print(f"  No images found in {person_id}")
            continue
        
        print(f"  Found {len(image_files)} images")
        
        # Test each image
        for img_path in image_files[:20]:  # Test max 20 images per person
            try:
                # Read and preprocess image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Convert to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Get prediction
                start_time = time.time()
                result = recognizer.recognize_person(img_rgb)
                pred_time = time.time() - start_time
                
                predicted_id = result.get('person_id', 'unknown')
                confidence = result.get('confidence', 0.0)
                
                total_images += 1
                prediction_details[person_id]['total'] += 1
                prediction_details[person_id]['confidence_scores'].append(confidence)
                
                # Check if prediction is correct
                if predicted_id == 'unknown':
                    unknown_predictions += 1
                    prediction_details[person_id]['unknown'] += 1
                    status = "❓"
                elif predicted_id == person_id:
                    correct_predictions += 1
                    prediction_details[person_id]['correct'] += 1
                    status = "✓"
                else:
                    wrong_predictions += 1
                    prediction_details[person_id]['wrong'] += 1
                    prediction_details[person_id]['wrong_as'][predicted_id] += 1
                    status = "✗"
                
                # Print result for first few images
                if prediction_details[person_id]['total'] <= 5:
                    print(f"    {status} {img_path.name}: Predicted {predicted_id} (conf: {confidence:.3f}) in {pred_time:.3f}s")
                
            except Exception as e:
                print(f"    Error processing {img_path.name}: {e}")
        
        # Summary for this person
        stats = prediction_details[person_id]
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total'] * 100
            avg_confidence = np.mean(stats['confidence_scores'])
            
            print(f"  Summary: {stats['correct']}/{stats['total']} correct ({accuracy:.1f}%), avg confidence: {avg_confidence:.3f}")
            
            if stats['wrong'] > 0:
                print(f"  Misclassified as: {dict(stats['wrong_as'])}")
    
    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL RESULTS:")
    print("=" * 80)
    
    if total_images > 0:
        overall_accuracy = correct_predictions / total_images * 100
        print(f"Total images tested: {total_images}")
        print(f"Correct predictions: {correct_predictions} ({overall_accuracy:.1f}%)")
        print(f"Wrong predictions: {wrong_predictions} ({wrong_predictions/total_images*100:.1f}%)")
        print(f"Unknown predictions: {unknown_predictions} ({unknown_predictions/total_images*100:.1f}%)")
        
        print(f"\nPer-person accuracy:")
        for person_id, stats in sorted(prediction_details.items()):
            if stats['total'] > 0:
                accuracy = stats['correct'] / stats['total'] * 100
                avg_conf = np.mean(stats['confidence_scores'])
                
                # Check if this person is in the trained model
                is_trained = person_id in recognizer.person_id_mapping.values()
                trained_marker = "📚" if is_trained else "❌"
                
                print(f"  {trained_marker} {person_id}: {stats['correct']}/{stats['total']} ({accuracy:.1f}%), avg conf: {avg_conf:.3f}")
                
                if stats['wrong'] > 0:
                    print(f"     Confused with: {dict(stats['wrong_as'])}")
        
        # Show which persons are not in the trained model
        untrained_persons = [p for p in prediction_details.keys() 
                           if p not in recognizer.person_id_mapping.values()]
        if untrained_persons:
            print(f"\n⚠️  Persons not in trained model: {untrained_persons}")
            print("   These will always be predicted as 'unknown' or misclassified")
    else:
        print("No images found to test!")

def test_single_person(person_id):
    """Test a specific person's images"""
    print(f"\nTesting only {person_id}...")
    # Implementation for testing single person
    pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test person recognition on exported images')
    parser.add_argument('--person', type=str, help='Test specific person ID (e.g., PERSON-0001)')
    parser.add_argument('--threshold', type=float, default=0.7, help='Confidence threshold')
    
    args = parser.parse_args()
    
    if args.person:
        test_single_person(args.person)
    else:
        test_person_predictions()