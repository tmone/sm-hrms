"""
Fix dataset compatibility by adding missing fields
"""

import json
from pathlib import Path

def fix_dataset_files():
    """Add missing fields to existing dataset info files"""
    datasets_dir = Path('datasets/person_recognition')
    
    if not datasets_dir.exists():
        print("No datasets directory found")
        return
    
    for dataset_dir in datasets_dir.iterdir():
        if dataset_dir.is_dir():
            info_file = dataset_dir / 'dataset_info.json'
            if info_file.exists():
                print(f"Checking dataset: {dataset_dir.name}")
                
                # Read existing info
                with open(info_file, 'r') as f:
                    info = json.load(f)
                
                # Check if total_faces is missing
                updated = False
                if 'total_faces' not in info:
                    # Use total_features if available, otherwise count from persons
                    if 'total_features' in info:
                        info['total_faces'] = info['total_features']
                    else:
                        total_faces = 0
                        for person_data in info.get('persons', {}).values():
                            total_faces += person_data.get('features_count', 0)
                        info['total_faces'] = total_faces
                    updated = True
                    print(f"  Added total_faces: {info['total_faces']}")
                
                # Update person data
                for person_id, person_data in info.get('persons', {}).items():
                    if 'faces_count' not in person_data:
                        person_data['faces_count'] = person_data.get('features_count', 0)
                        updated = True
                    if 'embeddings_count' not in person_data:
                        person_data['embeddings_count'] = person_data.get('features_count', 0)
                        updated = True
                
                # Save updated info
                if updated:
                    with open(info_file, 'w') as f:
                        json.dump(info, f, indent=2)
                    print(f"  Updated {dataset_dir.name}")
                else:
                    print(f"  No updates needed for {dataset_dir.name}")

if __name__ == "__main__":
    fix_dataset_files()