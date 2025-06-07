"""
Test recreating a dataset with the simple creator
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hr_management.processing.person_dataset_creator_simple import PersonDatasetCreatorSimple
from pathlib import Path
import shutil

def test_recreate_dataset():
    """Recreate the dataset to test feature extraction"""
    
    # Get persons from the existing dataset
    dataset_name = "test_person_dataset"
    person_ids = ["PERSON-0001", "PERSON-0007", "PERSON-0008", "PERSON-0010", "PERSON-0011", "PERSON-0012"]
    
    # Remove old test dataset if exists
    test_dataset_path = Path('datasets/person_recognition') / dataset_name
    if test_dataset_path.exists():
        shutil.rmtree(test_dataset_path)
        print(f"[DELETE]  Removed old test dataset: {test_dataset_path}")
    
    # Create new dataset
    creator = PersonDatasetCreatorSimple()
    print(f"\n[PROCESSING] Creating test dataset with {len(person_ids)} persons...")
    
    dataset_info = creator.create_dataset_from_persons(person_ids, dataset_name)
    
    print(f"\n[OK] Dataset created:")
    print(f"   Total persons: {len(dataset_info['persons'])}")
    print(f"   Total images: {dataset_info['total_images']}")
    print(f"   Total features: {dataset_info['total_features']}")
    
    print(f"\n[INFO] Per-person stats:")
    for person_id, person_data in dataset_info['persons'].items():
        print(f"   {person_id}:")
        print(f"      - Images: {person_data.get('images_count', 0)}")
        print(f"      - Features: {person_data.get('features_count', 0)}")
        print(f"      - Success: {person_data.get('success', False)}")
    
    # Check the features directory
    features_dir = test_dataset_path / 'features'
    if features_dir.exists():
        print(f"\n[FILE] Features directory contents:")
        for person_dir in features_dir.iterdir():
            if person_dir.is_dir():
                files = list(person_dir.glob('*.pkl'))
                print(f"   {person_dir.name}: {len(files)} feature files")
    else:
        print(f"\n[WARNING]  Features directory not found!")
    
    # Test loading the dataset
    print(f"\n[PROCESSING] Testing dataset loading...")
    X, y, loaded_person_ids = creator.prepare_training_data(dataset_name)
    
    print(f"\n[INFO] Loaded training data:")
    print(f"   X shape: {X.shape if len(X) > 0 else 'empty'}")
    print(f"   y shape: {y.shape if len(y) > 0 else 'empty'}")
    print(f"   Unique persons: {len(loaded_person_ids)}")
    print(f"   Person IDs: {loaded_person_ids}")
    
    if len(X) > 0:
        unique_y = set(y)
        print(f"   Unique labels in y: {unique_y}")
        for label in unique_y:
            count = sum(1 for l in y if l == label)
            print(f"      Label {label} ({loaded_person_ids[label] if label < len(loaded_person_ids) else 'unknown'}): {count} samples")


if __name__ == '__main__':
    test_recreate_dataset()