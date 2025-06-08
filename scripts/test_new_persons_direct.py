#!/usr/bin/env python3
"""
Test all NEW person folders directly without web server
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import json
import cv2
import numpy as np
from collections import defaultdict
import pickle

def load_model_directly():
    """Load the default model directly"""
    model_name = 'person_model_svm_20250607_181818'
    model_dir = Path('models/person_recognition') / model_name
    
    # Load person mappings
    with open(model_dir / 'person_id_mapping.pkl', 'rb') as f:
        person_mappings = pickle.load(f)
    
    return person_mappings

def test_new_persons():
    """Test new person folders"""
    
    # The 9 trained persons
    trained_persons = ['PERSON-0001', 'PERSON-0002', 'PERSON-0007', 'PERSON-0008', 
                      'PERSON-0010', 'PERSON-0017', 'PERSON-0019', 'PERSON-0020', 'PERSON-0021']
    
    persons_dir = Path("processing/outputs/persons")
    
    # Get all person folders
    all_persons = sorted([d.name for d in persons_dir.iterdir() 
                         if d.is_dir() and d.name.startswith('PERSON-')])
    
    # Get NEW persons (not in trained list)
    new_persons = [p for p in all_persons if p not in trained_persons]
    
    print(f"Model trained on: {trained_persons}")
    print(f"Total persons in system: {len(all_persons)}")
    print(f"NEW persons (not trained): {len(new_persons)}")
    print("=" * 80)
    
    # Load model mappings
    try:
        person_mappings = load_model_directly()
        print(f"Model mappings loaded: {person_mappings}")
        print()
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Analyze new persons
    print("NEW persons with most images:")
    print("-" * 80)
    
    # Count images for each new person
    new_person_stats = []
    for person_id in new_persons:
        person_folder = persons_dir / person_id
        image_count = len(list(person_folder.glob("*.jpg")))
        new_person_stats.append((person_id, image_count))
    
    # Sort by image count
    new_person_stats.sort(key=lambda x: x[1], reverse=True)
    
    # Show top 20
    for person_id, count in new_person_stats[:20]:
        print(f"{person_id}: {count} images")
    
    print("\n" + "=" * 80)
    print("ANALYSIS:")
    print("=" * 80)
    
    # Count very high person IDs
    high_ids = [p for p in new_persons if int(p.replace('PERSON-', '')) > 100]
    print(f"✓ Found {len(high_ids)} persons with ID > 100")
    print(f"  Highest ID: {max(high_ids) if high_ids else 'None'}")
    
    # Check person ID counter
    counter_file = Path("processing/outputs/persons/person_id_counter.json")
    if counter_file.exists():
        with open(counter_file) as f:
            counter_data = json.load(f)
        last_id = counter_data.get('last_person_id', 0)
        print(f"\n✓ Current person ID counter: {last_id}")
        print(f"  Next new person will be: PERSON-{last_id + 1:04d}")
    
    print("\n" + "=" * 80)
    print("WHY NEW PERSONS ARE CREATED:")
    print("=" * 80)
    print("1. Person appears in video")
    print("2. Recognition model returns 'unknown' (not one of the 9 trained)")
    print("3. System assigns next sequential ID")
    print("4. All detections saved to new folder")
    print("\nThis is EXPECTED behavior for untrained persons!")
    
    # Check for potential duplicates visually
    print("\n" + "=" * 80)
    print("CHECKING FOR VISUAL DUPLICATES:")
    print("=" * 80)
    
    # Sample a few new persons and check their images
    test_persons = new_person_stats[:5]  # Top 5 new persons
    
    for person_id, count in test_persons:
        person_folder = persons_dir / person_id
        images = list(person_folder.glob("*.jpg"))[:3]  # Sample 3 images
        
        if len(images) >= 2:
            print(f"\n{person_id} ({count} images) - Checking consistency:")
            
            # Load first two images
            img1 = cv2.imread(str(images[0]))
            img2 = cv2.imread(str(images[1]))
            
            if img1 is not None and img2 is not None:
                # Simple size check
                h1, w1 = img1.shape[:2]
                h2, w2 = img2.shape[:2]
                
                print(f"  Image 1: {images[0].name} ({w1}x{h1})")
                print(f"  Image 2: {images[1].name} ({w2}x{h2})")
                
                # Convert to grayscale and resize for comparison
                gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                
                # Resize to same size
                size = (100, 100)
                gray1_resized = cv2.resize(gray1, size)
                gray2_resized = cv2.resize(gray2, size)
                
                # Calculate histogram correlation
                hist1 = cv2.calcHist([gray1_resized], [0], None, [256], [0, 256])
                hist2 = cv2.calcHist([gray2_resized], [0], None, [256], [0, 256])
                
                correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                print(f"  Visual similarity: {correlation:.3f}")
                
                if correlation > 0.7:
                    print(f"  ✓ Images appear to be same person")
                else:
                    print(f"  ⚠️  Images might be different persons")

if __name__ == "__main__":
    test_new_persons()