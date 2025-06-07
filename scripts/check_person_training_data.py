#!/usr/bin/env python3
"""
Check training data for a specific person
Helps diagnose why a person is not being predicted correctly
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import json
import cv2
import numpy as np

def check_person_training_data(person_id="PERSON-0020"):
    """Check all training-related data for a person"""
    print(f"[SEARCH] Checking training data for {person_id}\n")
    
    # 1. Check person folder
    person_dir = Path("processing/outputs/persons") / person_id
    if not person_dir.exists():
        print(f"[ERROR] Person folder not found: {person_dir}")
        return
    
    # Count images
    person_images = list(person_dir.glob("*.jpg"))
    print(f"[FILE] Person folder: {person_dir}")
    print(f"   Images: {len(person_images)}")
    
    # Check metadata
    metadata_path = person_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        print(f"   Recognition: {metadata.get('recognized', False)}")
        print(f"   Recognition confidence: {metadata.get('recognition_confidence', 0):.2%}")
    
    # 2. Check dataset folder
    dataset_dir = Path("datasets") / person_id
    dataset_images = []
    if dataset_dir.exists():
        dataset_images = list(dataset_dir.glob("*.jpg"))
        print(f"\n[INFO] Dataset folder: {dataset_dir}")
        print(f"   Images: {len(dataset_images)}")
    else:
        print(f"\n[WARNING]  No dataset folder found for {person_id}")
    
    # 3. Check if person is in any trained models
    print(f"\n[AI] Checking trained models...")
    models_dir = Path("models/person_recognition")
    
    # Check each model
    person_in_models = []
    for model_dir in models_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        # Check metadata
        model_meta_path = model_dir / "metadata.json"
        if model_meta_path.exists():
            with open(model_meta_path) as f:
                model_meta = json.load(f)
            
            # Check if person is in this model
            if person_id in model_meta.get('person_ids', []):
                person_in_models.append({
                    'model': model_dir.name,
                    'created': model_meta.get('created_at', 'Unknown'),
                    'accuracy': model_meta.get('test_score', 0),
                    'num_persons': model_meta.get('num_persons', 0)
                })
    
    if person_in_models:
        print(f"[OK] Found in {len(person_in_models)} models:")
        for model_info in person_in_models:
            print(f"   - {model_info['model']}")
            print(f"     Created: {model_info['created']}")
            print(f"     Accuracy: {model_info['accuracy']:.2%}")
            print(f"     Total persons: {model_info['num_persons']}")
    else:
        print(f"[ERROR] {person_id} NOT found in any trained models!")
    
    # 4. Check dataset used for training
    print(f"\n[PACKAGE] Checking training datasets...")
    latest_dataset = None
    for dataset_path in models_dir.glob("datasets/*/dataset_info.json"):
        with open(dataset_path) as f:
            dataset_info = json.load(f)
        
        if person_id in dataset_info.get('persons', {}):
            person_data = dataset_info['persons'][person_id]
            print(f"\n   Dataset: {dataset_path.parent.name}")
            print(f"   Images for {person_id}: {len(person_data.get('images', []))}")
            
            if not latest_dataset or dataset_path.stat().st_mtime > latest_dataset['mtime']:
                latest_dataset = {
                    'path': dataset_path.parent,
                    'mtime': dataset_path.stat().st_mtime,
                    'num_images': len(person_data.get('images', []))
                }
    
    if not latest_dataset:
        print(f"[ERROR] {person_id} NOT found in any training datasets!")
    
    # 5. Check image quality
    print(f"\n[IMAGE] Checking image quality...")
    if person_images:
        sizes = []
        for img_path in person_images[:5]:  # Check first 5
            img = cv2.imread(str(img_path))
            if img is not None:
                sizes.append(img.shape)
        
        if sizes:
            avg_height = np.mean([s[0] for s in sizes])
            avg_width = np.mean([s[1] for s in sizes])
            print(f"   Average image size: {avg_width:.0f}x{avg_height:.0f}")
            
            if avg_width < 128:
                print(f"   [WARNING]  Images might be too small for good recognition!")
    
    # 6. Check recognition attempts
    print(f"\n[TARGET] Recent recognition attempts:")
    # Check if this person was recently detected but not recognized
    recent_persons = sorted(
        [d for d in Path("processing/outputs/persons").iterdir() if d.is_dir()],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )[:10]
    
    misrecognized = []
    for recent_dir in recent_persons:
        if recent_dir.name == person_id:
            continue
            
        recent_meta_path = recent_dir / "metadata.json"
        if recent_meta_path.exists():
            with open(recent_meta_path) as f:
                recent_meta = json.load(f)
            
            # Check if this was supposed to be PERSON-0020
            if recent_meta.get('recognized') and recent_meta.get('original_tracking_id'):
                # This person was recognized as someone else
                pass
    
    # Summary and recommendations
    print(f"\n[TIP] Summary & Recommendations:")
    
    if not dataset_images:
        print(f"[ERROR] No images in dataset folder - need to confirm images first")
        print(f"   -> Go to person review page and confirm correct images")
    
    if not person_in_models:
        print(f"[ERROR] Not in any trained models - need to retrain")
        print(f"   -> Run: python scripts/retrain_person_model.py")
    
    if len(person_images) < 10:
        print(f"[WARNING]  Only {len(person_images)} images - might need more for better recognition")
        print(f"   -> Process more videos with this person")
    
    print(f"\n[TRACE] Next steps:")
    print(f"1. Ensure {person_id} has confirmed images in dataset")
    print(f"2. Retrain the model to include {person_id}")
    print(f"3. Test recognition with the new model")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--person", default="PERSON-0020", help="Person ID to check")
    args = parser.parse_args()
    
    check_person_training_data(args.person)