"""
Debug script to check dataset features
"""
import os
import sys
from pathlib import Path
import json
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def debug_dataset(dataset_name):
    """Debug a dataset to see what features are available"""
    dataset_path = Path('datasets/person_recognition') / dataset_name
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    print(f"🔍 Debugging dataset: {dataset_name}")
    print(f"   Path: {dataset_path}")
    
    # Check dataset info
    info_path = dataset_path / 'dataset_info.json'
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        print(f"\n📊 Dataset Info:")
        print(f"   Created: {info.get('created_at', 'Unknown')}")
        print(f"   Total persons: {len(info.get('persons', {}))}")
        print(f"   Total images: {info.get('total_images', 0)}")
        print(f"   Total features: {info.get('total_features', 0)}")
        
        print(f"\n👥 Persons in dataset:")
        for person_id, person_data in info.get('persons', {}).items():
            print(f"   {person_id}:")
            print(f"      - Images: {person_data.get('images_count', 0)}")
            print(f"      - Features: {person_data.get('features_count', 0)}")
            print(f"      - Success: {person_data.get('success', False)}")
    
    # Check directory structure
    print(f"\n📁 Directory structure:")
    for subdir in ['images', 'features', 'faces', 'embeddings']:
        subdir_path = dataset_path / subdir
        if subdir_path.exists():
            print(f"   {subdir}/")
            # Count files per person
            for person_dir in subdir_path.iterdir():
                if person_dir.is_dir():
                    files = list(person_dir.glob('*'))
                    print(f"      {person_dir.name}: {len(files)} files")
                    
                    # Sample first few files
                    for f in files[:3]:
                        print(f"         - {f.name}")
                    if len(files) > 3:
                        print(f"         ... and {len(files) - 3} more")
    
    # Check features specifically
    features_dir = dataset_path / 'features'
    if features_dir.exists():
        print(f"\n🔬 Checking features:")
        total_features = 0
        valid_features = 0
        
        for person_dir in features_dir.iterdir():
            if person_dir.is_dir():
                person_features = 0
                person_valid = 0
                
                for feature_file in person_dir.glob('*.pkl'):
                    total_features += 1
                    try:
                        with open(feature_file, 'rb') as f:
                            data = pickle.load(f)
                            if 'features' in data and data['features'] is not None:
                                person_valid += 1
                                valid_features += 1
                                # Check feature shape
                                if person_valid == 1:  # First valid feature
                                    features = data['features']
                                    print(f"   {person_dir.name}: Feature shape = {features.shape if hasattr(features, 'shape') else len(features)}")
                    except Exception as e:
                        print(f"   ⚠️  Error reading {feature_file.name}: {e}")
                    person_features += 1
                
                if person_features > 0:
                    print(f"      Valid: {person_valid}/{person_features}")
        
        print(f"\n📊 Total features: {total_features}")
        print(f"   Valid features: {valid_features}")
    else:
        print(f"\n⚠️  No features directory found!")


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
                print("\nUsage: python scripts/debug_dataset_features.py <dataset_name>")
                if datasets:
                    print(f"\nDebugging first dataset: {datasets[0]}")
                    dataset_name = datasets[0]
                else:
                    sys.exit(1)
            else:
                print("No datasets found")
                sys.exit(1)
        else:
            print("Datasets directory not found")
            sys.exit(1)
    
    debug_dataset(dataset_name)