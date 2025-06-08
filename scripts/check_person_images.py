#!/usr/bin/env python3
"""
Check all person images in the persons folder
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import json
from collections import defaultdict

def check_person_images():
    """Check all person folders and count images"""
    
    persons_dir = Path("processing/outputs/persons")
    if not persons_dir.exists():
        print(f"✗ Persons directory not found: {persons_dir}")
        return
    
    # Get all person folders
    person_folders = sorted([d for d in persons_dir.iterdir() 
                           if d.is_dir() and d.name.startswith('PERSON-')])
    
    print(f"Found {len(person_folders)} person folders")
    print("=" * 80)
    
    # Statistics
    total_images = 0
    person_stats = {}
    
    # Check each folder
    for person_folder in person_folders:
        person_id = person_folder.name
        
        # Count images
        jpg_files = list(person_folder.glob("*.jpg"))
        png_files = list(person_folder.glob("*.png"))
        total_files = len(jpg_files) + len(png_files)
        
        if total_files > 0:
            person_stats[person_id] = {
                'total': total_files,
                'jpg': len(jpg_files),
                'png': len(png_files)
            }
            total_images += total_files
        
        # Check metadata if exists
        metadata_file = person_folder / "metadata.json"
        metadata_info = ""
        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    if 'recognized_name' in metadata:
                        metadata_info = f" [Recognized: {metadata['recognized_name']}]"
            except:
                pass
        
        if total_files > 0:
            print(f"{person_id}: {total_files} images (jpg: {len(jpg_files)}, png: {len(png_files)}){metadata_info}")
    
    print("=" * 80)
    print(f"Total: {len(person_stats)} persons with {total_images} images")
    
    # Show persons with most images
    print("\nTop 10 persons by image count:")
    sorted_persons = sorted(person_stats.items(), key=lambda x: x[1]['total'], reverse=True)
    for person_id, stats in sorted_persons[:10]:
        print(f"  {person_id}: {stats['total']} images")
    
    # Check which model should recognize these persons
    print("\nChecking model configuration...")
    config_path = Path("models/person_recognition/config.json")
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print(f"Default model: {config.get('default_model')}")
        
        # List all available models
        models_dir = Path("models/person_recognition")
        print("\nAvailable models:")
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir() and not model_dir.name.startswith('.'):
                model_files = list(model_dir.glob("*.pkl"))
                print(f"  - {model_dir.name} ({len(model_files)} pkl files)")
                
                # Check for person mappings
                mapping_file = model_dir / "person_id_mapping.pkl"
                if mapping_file.exists():
                    try:
                        import pickle
                        with open(mapping_file, 'rb') as f:
                            mappings = pickle.load(f)
                        print(f"    Person mappings: {list(mappings.values())}")
                    except Exception as e:
                        print(f"    Error reading mappings: {e}")

if __name__ == "__main__":
    check_person_images()