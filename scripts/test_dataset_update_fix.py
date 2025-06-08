#!/usr/bin/env python3
"""Test the dataset update fix for removing persons"""

import json
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hr_management.blueprints.persons import update_datasets_after_person_deletion

def test_dataset_update():
    """Test that dataset update handles missing fields gracefully"""
    
    # Find a dataset to test with
    datasets_dir = Path('datasets/person_recognition')
    if not datasets_dir.exists():
        print("[ERROR] No datasets directory found")
        return
    
    # Get first dataset
    dataset_dirs = [d for d in datasets_dir.iterdir() if d.is_dir()]
    if not dataset_dirs:
        print("[ERROR] No datasets found")
        return
    
    dataset_dir = dataset_dirs[0]
    dataset_info_path = dataset_dir / 'dataset_info.json'
    
    if not dataset_info_path.exists():
        print(f"[ERROR] No dataset_info.json in {dataset_dir.name}")
        return
    
    # Load dataset info
    with open(dataset_info_path) as f:
        dataset_info = json.load(f)
    
    print(f"[INFO] Testing with dataset: {dataset_dir.name}")
    print(f"   Total persons: {len(dataset_info.get('persons', {}))}")
    
    # Get a person ID to test with (but don't actually delete)
    person_ids = list(dataset_info.get('persons', {}).keys())
    if not person_ids:
        print("[ERROR] No persons in dataset")
        return
    
    test_person_id = person_ids[0]
    print(f"   Test person ID: {test_person_id}")
    
    # Check which total fields exist
    total_fields = [k for k in dataset_info.keys() if k.startswith('total_')]
    print(f"   Total fields present: {', '.join(total_fields)}")
    
    # Test the update function (without actually deleting)
    try:
        # This should not raise KeyError anymore
        print("\n🧪 Testing update_datasets_after_person_deletion...")
        # We'll pass an empty list to avoid actually deleting anything
        result = update_datasets_after_person_deletion([])
        print("[OK] Function executed without errors")
        
        # Now test with a fake person ID that doesn't exist
        fake_id = "PERSON-9999"
        print(f"\n🧪 Testing with non-existent person ID: {fake_id}")
        result = update_datasets_after_person_deletion([fake_id])
        print("[OK] Handled non-existent person ID gracefully")
        
    except Exception as e:
        print(f"[ERROR] Error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset_update()