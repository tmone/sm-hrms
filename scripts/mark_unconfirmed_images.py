#!/usr/bin/env python3
"""
Mark images that are not in the dataset for review
This helps identify potentially misrecognized images
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import json
import shutil
from datetime import datetime

def get_dataset_images():
    """Get all images that are already in datasets"""
    dataset_images = set()
    
    # Check refined datasets
    datasets_dir = Path("datasets")
    if datasets_dir.exists():
        for person_dir in datasets_dir.iterdir():
            if person_dir.is_dir() and person_dir.name.startswith("PERSON-"):
                for img in person_dir.glob("*.jpg"):
                    dataset_images.add(img.name)
    
    # Check training datasets
    training_dir = Path("models/person_recognition/datasets")
    if training_dir.exists():
        for dataset_dir in training_dir.iterdir():
            if dataset_dir.is_dir():
                dataset_info = dataset_dir / "dataset_info.json"
                if dataset_info.exists():
                    with open(dataset_info) as f:
                        info = json.load(f)
                    for person_id, person_data in info.get("persons", {}).items():
                        for img_path in person_data.get("images", []):
                            dataset_images.add(Path(img_path).name)
    
    return dataset_images

def mark_unconfirmed_images():
    """Mark images not in dataset as unconfirmed"""
    print("🔍 Marking unconfirmed images for review\n")
    
    dataset_images = get_dataset_images()
    print(f"📊 Found {len(dataset_images)} images in datasets\n")
    
    persons_dir = Path("processing/outputs/persons")
    if not persons_dir.exists():
        print("❌ No persons directory found")
        return
    
    stats = {
        "total_persons": 0,
        "persons_with_unconfirmed": 0,
        "total_images": 0,
        "confirmed_images": 0,
        "unconfirmed_images": 0
    }
    
    # Process each person folder
    for person_dir in persons_dir.iterdir():
        if not person_dir.is_dir() or not person_dir.name.startswith("PERSON-"):
            continue
        
        stats["total_persons"] += 1
        
        # Load or create review status file
        review_status_file = person_dir / "review_status.json"
        if review_status_file.exists():
            with open(review_status_file) as f:
                review_status = json.load(f)
        else:
            review_status = {
                "person_id": person_dir.name,
                "created_at": datetime.now().isoformat(),
                "images": {}
            }
        
        # Check each image
        unconfirmed_count = 0
        confirmed_count = 0
        
        for img_path in person_dir.glob("*.jpg"):
            stats["total_images"] += 1
            img_name = img_path.name
            
            # Check if image is in dataset
            is_confirmed = img_name in dataset_images
            
            if is_confirmed:
                confirmed_count += 1
                stats["confirmed_images"] += 1
                status = "confirmed"
            else:
                unconfirmed_count += 1
                stats["unconfirmed_images"] += 1
                status = "unconfirmed"
            
            # Update review status
            if img_name not in review_status["images"]:
                review_status["images"][img_name] = {
                    "status": status,
                    "in_dataset": is_confirmed,
                    "added_at": datetime.now().isoformat()
                }
            else:
                # Update existing status
                review_status["images"][img_name]["in_dataset"] = is_confirmed
                if is_confirmed and review_status["images"][img_name]["status"] == "unconfirmed":
                    review_status["images"][img_name]["status"] = "confirmed"
        
        # Update summary
        review_status["summary"] = {
            "total_images": confirmed_count + unconfirmed_count,
            "confirmed": confirmed_count,
            "unconfirmed": unconfirmed_count,
            "last_updated": datetime.now().isoformat()
        }
        
        # Save review status
        with open(review_status_file, 'w') as f:
            json.dump(review_status, f, indent=2)
        
        if unconfirmed_count > 0:
            stats["persons_with_unconfirmed"] += 1
            print(f"⚠️  {person_dir.name}: {unconfirmed_count} unconfirmed, {confirmed_count} confirmed")
        else:
            print(f"✅ {person_dir.name}: All {confirmed_count} images confirmed")
    
    # Print summary
    print(f"\n📊 Summary:")
    print(f"   Total persons: {stats['total_persons']}")
    print(f"   Persons with unconfirmed images: {stats['persons_with_unconfirmed']}")
    print(f"   Total images: {stats['total_images']}")
    print(f"   Confirmed images: {stats['confirmed_images']}")
    print(f"   Unconfirmed images: {stats['unconfirmed_images']}")
    
    if stats['unconfirmed_images'] > 0:
        print(f"\n⚠️  Found {stats['unconfirmed_images']} unconfirmed images that need review!")
        print("   These may be misrecognized and should be verified.")

if __name__ == "__main__":
    mark_unconfirmed_images()