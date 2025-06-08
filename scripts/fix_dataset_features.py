"""
Fix existing datasets by extracting appearance-based features
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hr_management.processing.person_dataset_creator_simple import PersonDatasetCreatorSimple
from pathlib import Path
import json
import shutil

def fix_dataset_features(dataset_name):
    """Extract features for an existing dataset that only has faces/embeddings"""
    
    dataset_path = Path('datasets/person_recognition') / dataset_name
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_name}")
        return False
    
    print(f"[PROCESSING] Fixing dataset: {dataset_name}")
    
    # Load dataset info
    info_path = dataset_path / 'dataset_info.json'
    if not info_path.exists():
        print(f"[ERROR] Dataset info not found")
        return False
    
    with open(info_path) as f:
        dataset_info = json.load(f)
    
    # Create features directory if missing
    features_dir = dataset_path / 'features'
    features_dir.mkdir(exist_ok=True)
    
    # Get creator
    creator = PersonDatasetCreatorSimple()
    
    # Process each person
    total_features = 0
    for person_id, person_data in dataset_info['persons'].items():
        print(f"\n[INFO] Processing {person_id}...")
        
        person_features_dir = features_dir / person_id
        person_features_dir.mkdir(exist_ok=True)
        
        # Get images from the images directory
        person_images_dir = dataset_path / 'images' / person_id
        if not person_images_dir.exists():
            print(f"   [WARNING]  No images directory for {person_id}")
            continue
        
        features_count = 0
        for img_path in person_images_dir.glob('*.jpg'):
            # Extract features
            features = creator._extract_simple_features(str(img_path), person_id, img_path.name)
            if features is not None:
                # Save features
                feature_filename = f"{person_id}_features_{img_path.stem}.pkl"
                feature_path = person_features_dir / feature_filename
                
                import pickle
                feature_data = {
                    'person_id': person_id,
                    'source_image': img_path.name,
                    'features': features,
                    'bbox': None,
                    'confidence': 0.9
                }
                
                with open(feature_path, 'wb') as f:
                    pickle.dump(feature_data, f)
                
                features_count += 1
        
        print(f"   [OK] Extracted {features_count} features")
        total_features += features_count
        
        # Update person data
        person_data['features_count'] = features_count
    
    # Update dataset info
    dataset_info['total_features'] = total_features
    dataset_info['modified_at'] = Path(os.path.abspath(__file__)).stem + '_fixed'
    
    # Save updated info
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"\n[OK] Fixed dataset {dataset_name}:")
    print(f"   Total features extracted: {total_features}")
    
    # Test loading
    print(f"\n[PROCESSING] Testing dataset loading...")
    X, y, person_ids = creator.prepare_training_data(dataset_name)
    print(f"   Loaded {len(X)} samples for {len(person_ids)} persons")
    
    return True


if __name__ == '__main__':
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
    else:
        # List available datasets
        datasets_dir = Path('datasets/person_recognition')
        if datasets_dir.exists():
            datasets = [d.name for d in datasets_dir.iterdir() if d.is_dir()]
            if datasets:
                print("Available datasets:")
                for d in datasets:
                    print(f"  - {d}")
                print("\nFixing all datasets...")
                for dataset_name in datasets:
                    fix_dataset_features(dataset_name)
                    print("-" * 50)
            else:
                print("No datasets found")
        else:
            print("Datasets directory not found")