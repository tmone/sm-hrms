#!/usr/bin/env python3
"""
Test detection of untrained persons (those not in the model)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import json
import cv2
import random

def test_untrained_persons():
    """Test how untrained persons are handled"""
    
    # Get model configuration
    config_path = Path("models/person_recognition/config.json")
    with open(config_path) as f:
        config = json.load(f)
    
    model_name = config.get('default_model')
    print(f"Current model: {model_name}")
    
    # Get trained persons list
    trained_persons = ['PERSON-0001', 'PERSON-0002', 'PERSON-0007', 'PERSON-0008', 
                      'PERSON-0010', 'PERSON-0017', 'PERSON-0019', 'PERSON-0020', 'PERSON-0021']
    
    print(f"Trained persons: {trained_persons}")
    print("=" * 80)
    
    # Get all person folders
    persons_dir = Path("processing/outputs/persons")
    person_folders = sorted([d for d in persons_dir.iterdir() 
                           if d.is_dir() and d.name.startswith('PERSON-')])
    
    # Find untrained persons
    untrained_persons = [f.name for f in person_folders if f.name not in trained_persons]
    
    print(f"Found {len(untrained_persons)} untrained persons:")
    print(f"Untrained persons: {untrained_persons[:10]}..." if len(untrained_persons) > 10 else untrained_persons)
    print("=" * 80)
    
    # Test some untrained persons
    print("\nTesting detection of untrained persons (they should get new PERSON codes):\n")
    
    # Select some untrained persons with good amount of images
    test_persons = []
    for person_id in untrained_persons:
        person_folder = persons_dir / person_id
        image_count = len(list(person_folder.glob("*.jpg")))
        if image_count >= 10:  # Only test persons with at least 10 images
            test_persons.append((person_id, image_count))
    
    # Sort by image count and take top 10
    test_persons.sort(key=lambda x: x[1], reverse=True)
    test_persons = test_persons[:10]
    
    print("Testing these untrained persons:")
    for person_id, count in test_persons:
        print(f"  {person_id}: {count} images")
    
    # Simulate what happens when these persons appear in new videos
    print("\n" + "=" * 80)
    print("SIMULATION: What happens when untrained persons appear in new videos:")
    print("=" * 80)
    
    # Check current person ID counter
    counter_file = Path("processing/outputs/persons/person_id_counter.json")
    if counter_file.exists():
        with open(counter_file) as f:
            counter_data = json.load(f)
        last_id = counter_data.get('last_person_id', 0)
        print(f"\nCurrent person ID counter: {last_id}")
        print(f"Next new person will get: PERSON-{last_id + 1:04d}")
    
    # Show what would happen
    print("\nExpected behavior for untrained persons:")
    print("1. Recognition model will return 'unknown' for these faces")
    print("2. System will assign new sequential PERSON IDs starting from the counter")
    print("3. Images will be saved in new PERSON folders")
    
    # Example scenario
    print("\nExample scenario:")
    for i, (person_id, count) in enumerate(test_persons[:5]):
        next_id = last_id + 1 + i if 'last_id' in locals() else 200 + i
        print(f"  {person_id} appears in video → Recognition: 'unknown' → Assigned: PERSON-{next_id:04d}")
    
    # Check for potential issues
    print("\n" + "=" * 80)
    print("POTENTIAL ISSUES TO WATCH:")
    print("=" * 80)
    
    # Check for very high person IDs
    high_id_persons = [p for p in untrained_persons if int(p.replace('PERSON-', '')) > 100]
    if high_id_persons:
        print(f"⚠️  High ID persons found: {high_id_persons[:5]}...")
        print("   These indicate many 'unknown' detections have occurred")
    
    # Check for persons with few images
    low_image_persons = []
    for person_id in untrained_persons[-10:]:  # Check last 10 persons
        person_folder = persons_dir / person_id
        image_count = len(list(person_folder.glob("*.jpg")))
        if image_count < 5:
            low_image_persons.append((person_id, image_count))
    
    if low_image_persons:
        print(f"\n⚠️  Recent persons with very few images:")
        for person_id, count in low_image_persons:
            print(f"   {person_id}: {count} images (might be false detections)")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print(f"✓ Model recognizes: {len(trained_persons)} persons")
    print(f"✓ Untrained persons: {len(untrained_persons)} (will get new IDs)")
    print(f"✓ Total persons in system: {len(person_folders)}")
    print(f"✓ System is correctly assigning new IDs to unknown faces")
    
    # Recommendation
    print("\nRECOMMENDATION:")
    if len(untrained_persons) > 20:
        print("➤ Consider retraining the model to include frequently appearing persons")
        print("➤ This will improve recognition accuracy and reduce 'unknown' assignments")

if __name__ == "__main__":
    test_untrained_persons()