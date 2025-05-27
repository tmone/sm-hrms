"""
Regenerate features for a dataset
"""

import sys
sys.path.append('.')

from hr_management.processing.person_dataset_creator_simple import PersonDatasetCreatorSimple
from pathlib import Path
import json

def regenerate_features(dataset_name):
    """Regenerate all features for a dataset"""
    creator = PersonDatasetCreatorSimple()
    dataset_path = Path('datasets/person_recognition') / dataset_name
    
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_name}")
        return
    
    # Load dataset info
    with open(dataset_path / 'dataset_info.json') as f:
        info = json.load(f)
    
    # Regenerate features for each person
    print(f"Regenerating features for dataset: {dataset_name}")
    
    images_dir = dataset_path / 'images'
    for person_id in info['persons'].keys():
        person_images_dir = images_dir / person_id
        if not person_images_dir.exists():
            continue
            
        print(f"Processing {person_id}...")
        
        # Process images and create features
        features_dir = dataset_path / 'features' / person_id
        features_dir.mkdir(parents=True, exist_ok=True)
        
        feature_count = 0
        for img_path in person_images_dir.glob('*.jpg'):
            features = creator._extract_simple_features(str(img_path), person_id, img_path.name)
            if features is not None:
                feature_count += 1
        
        print(f"  Generated {feature_count} features for {person_id}")
    
    print("✅ Feature regeneration complete")

if __name__ == "__main__":
    regenerate_features('all_persons_2025_05_27')