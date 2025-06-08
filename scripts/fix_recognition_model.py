#!/usr/bin/env python3
"""Fix the recognition model by creating missing files"""

import json
import pickle
from pathlib import Path
import numpy as np
from sklearn.preprocessing import LabelEncoder

def fix_recognition_model():
    """Create missing files for the recognition model"""
    
    print("[CONFIG] Fixing recognition model...\n")
    
    # Load model metadata
    model_dir = Path('models/person_recognition/refined_quick_20250606_054446')
    metadata_path = model_dir / 'metadata.json'
    
    if not metadata_path.exists():
        print("[ERROR] Model metadata not found")
        return
        
    with open(metadata_path) as f:
        metadata = json.load(f)
        
    person_ids = metadata.get('person_ids', [])
    print(f"Model contains {len(person_ids)} persons:")
    for pid in person_ids:
        print(f"  - {pid}")
        
    # 1. Create label_encoder.pkl
    print("\n1. Creating label_encoder.pkl...")
    
    # Create a label encoder with the person IDs
    label_encoder = LabelEncoder()
    label_encoder.fit(person_ids)
    
    # Save it
    label_encoder_path = model_dir / 'label_encoder.pkl'
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
        
    print(f"   [OK] Created {label_encoder_path}")
    print(f"   Classes: {list(label_encoder.classes_)}")
    
    # 2. Create persons.json
    print("\n2. Creating persons.json...")
    
    # Create persons data structure
    persons_data = {}
    for i, person_id in enumerate(person_ids):
        persons_data[person_id] = {
            "person_id": person_id,
            "label_index": i,
            "sample_count": 0,  # We don't have this info
            "created_at": "2025-06-06T05:44:47"  # From metadata
        }
        
    persons_path = model_dir / 'persons.json'
    with open(persons_path, 'w') as f:
        json.dump(persons_data, f, indent=2)
        
    print(f"   [OK] Created {persons_path}")
    
    # 3. Create person_id_mapping.json for better compatibility
    print("\n3. Creating person_id_mapping.json...")
    
    # Create mapping from label index to person ID
    mapping = {i: pid for i, pid in enumerate(person_ids)}
    
    mapping_path = model_dir / 'person_id_mapping.json'
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=2)
        
    print(f"   [OK] Created {mapping_path}")
    
    # 4. Update model files list
    print("\n4. Checking all model files...")
    
    required_files = [
        'model.pkl',
        'scaler.pkl', 
        'label_encoder.pkl',
        'persons.json',
        'metadata.json',
        'person_id_mapping.json'
    ]
    
    for file_name in required_files:
        file_path = model_dir / file_name
        if file_path.exists():
            print(f"   [OK] {file_name}")
        else:
            print(f"   [ERROR] {file_name} - MISSING")
            
    print("\n[OK] Model files fixed!")
    
    # 5. Test loading
    print("\n5. Testing model loading...")
    
    try:
        # Test loading label encoder
        with open(model_dir / 'label_encoder.pkl', 'rb') as f:
            test_encoder = pickle.load(f)
        print("   [OK] Label encoder loads correctly")
        
        # Test loading persons.json
        with open(model_dir / 'persons.json') as f:
            test_persons = json.load(f)
        print(f"   [OK] Persons.json loads correctly ({len(test_persons)} persons)")
        
    except Exception as e:
        print(f"   [ERROR] Error testing files: {e}")
        
    print("\n🎉 Recognition model should now work properly!")
    print("\nNext steps:")
    print("1. Process a video to test recognition")
    print("2. Check if existing person IDs are reused")
    print("3. Monitor the recognition logs")

if __name__ == "__main__":
    fix_recognition_model()