"""
Check training data to debug training issues
"""

import sys
sys.path.append('.')

from hr_management.processing.person_dataset_creator_simple import PersonDatasetCreatorSimple
import numpy as np

def check_training_data(dataset_name):
    """Check the training data"""
    creator = PersonDatasetCreatorSimple()
    
    print(f"Loading training data from: {dataset_name}")
    X, y, person_ids = creator.prepare_training_data(dataset_name)
    
    print(f"\nTraining data summary:")
    print(f"Total samples: {len(X)}")
    print(f"Total persons: {len(person_ids)}")
    print(f"Feature dimension: {X.shape[1] if len(X) > 0 else 'N/A'}")
    
    # Check class distribution
    print(f"\nClass distribution:")
    unique_classes = np.unique(y)
    print(f"Unique classes in y: {len(unique_classes)}")
    
    for i, person_id in enumerate(person_ids):
        count = np.sum(y == i)
        print(f"  Class {i} ({person_id}): {count} samples")
    
    # Check if any class is missing
    expected_classes = set(range(len(person_ids)))
    actual_classes = set(unique_classes)
    missing_classes = expected_classes - actual_classes
    
    if missing_classes:
        print(f"\n[WARNING]  WARNING: Missing classes: {missing_classes}")
        for i in missing_classes:
            print(f"   Class {i} ({person_ids[i]}) has no samples in y")

if __name__ == "__main__":
    check_training_data('all_persons_2025_05_27')