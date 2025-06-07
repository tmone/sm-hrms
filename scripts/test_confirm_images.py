#!/usr/bin/env python3
"""
Test confirming images - simulate what happens when user confirms images
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import json
import shutil
from datetime import datetime

def test_confirm_images(person_id="PERSON-0001", num_images=5):
    """Test confirming some images for a person"""
    print(f"🧪 Testing image confirmation for {person_id}\n")
    
    person_dir = Path("processing/outputs/persons") / person_id
    if not person_dir.exists():
        print(f"[ERROR] Person {person_id} not found")
        return
    
    # Get some images to confirm
    images = list(person_dir.glob("*.jpg"))[:num_images]
    if not images:
        print("[ERROR] No images found")
        return
    
    print(f"[CAMERA] Found {len(images)} images to confirm")
    
    # Create dataset directory
    dataset_dir = Path("datasets") / person_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    print(f"[FILE] Created dataset directory: {dataset_dir}")
    
    # Copy images to dataset
    copied = 0
    for img_path in images:
        dest_path = dataset_dir / img_path.name
        try:
            shutil.copy2(img_path, dest_path)
            copied += 1
            print(f"   [OK] Copied {img_path.name}")
        except Exception as e:
            print(f"   [ERROR] Failed to copy {img_path.name}: {e}")
    
    print(f"\n[OK] Copied {copied} images to dataset")
    
    # Update review status
    review_status_path = person_dir / "review_status.json"
    if review_status_path.exists():
        with open(review_status_path) as f:
            review_status = json.load(f)
    else:
        review_status = {"person_id": person_id, "images": {}, "summary": {}}
    
    # Mark images as confirmed
    for img_path in images:
        img_name = img_path.name
        review_status["images"][img_name] = {
            "status": "confirmed",
            "in_dataset": True,
            "confirmed_at": datetime.now().isoformat(),
            "confirmed_by": "test_script"
        }
    
    # Update summary
    total = len(list(person_dir.glob("*.jpg")))
    confirmed = len([v for v in review_status["images"].values() if v.get("status") == "confirmed"])
    review_status["summary"] = {
        "total_images": total,
        "confirmed": confirmed,
        "unconfirmed": total - confirmed,
        "in_dataset": copied
    }
    
    with open(review_status_path, 'w') as f:
        json.dump(review_status, f, indent=2)
    
    print(f"\n[INFO] Review status updated:")
    print(f"   Total images: {total}")
    print(f"   Confirmed: {confirmed}")
    print(f"   Unconfirmed: {total - confirmed}")
    
    # Create dataset metadata
    dataset_meta_path = dataset_dir / "dataset_info.json"
    dataset_meta = {
        "person_id": person_id,
        "created_at": datetime.now().isoformat(),
        "images": [img.name for img in images],
        "total_images": len(images)
    }
    
    with open(dataset_meta_path, 'w') as f:
        json.dump(dataset_meta, f, indent=2)
    
    print(f"\n[OK] Test complete! Check the persons page - {person_id} should now show:")
    print(f"   - Reduced unconfirmed count")
    print(f"   - Some images marked as confirmed in review page")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--person", default="PERSON-0001", help="Person ID to test")
    parser.add_argument("--count", type=int, default=5, help="Number of images to confirm")
    args = parser.parse_args()
    
    test_confirm_images(args.person, args.count)