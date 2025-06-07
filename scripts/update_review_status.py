#!/usr/bin/env python3
"""
Update review status to properly reflect confirmed images
Run this after confirming images to update the counts
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import json

def update_review_status():
    """Update review status for all persons"""
    print("📊 Updating review status...\n")
    
    persons_dir = Path("processing/outputs/persons")
    datasets_dir = Path("datasets")
    
    if not persons_dir.exists():
        print("❌ No persons directory found")
        return
    
    updated_count = 0
    
    for person_dir in persons_dir.iterdir():
        if not person_dir.is_dir() or not person_dir.name.startswith("PERSON-"):
            continue
        
        person_id = person_dir.name
        review_status_path = person_dir / "review_status.json"
        
        # Load or create review status
        if review_status_path.exists():
            with open(review_status_path) as f:
                review_status = json.load(f)
        else:
            review_status = {
                "person_id": person_id,
                "images": {},
                "summary": {}
            }
        
        # Check dataset directory
        dataset_dir = datasets_dir / person_id
        dataset_images = set()
        if dataset_dir.exists():
            dataset_images = {img.name for img in dataset_dir.glob("*.jpg")}
        
        # Update status for each image
        total = 0
        confirmed = 0
        in_dataset = 0
        
        for img_path in person_dir.glob("*.jpg"):
            img_name = img_path.name
            total += 1
            
            # Check if in dataset
            if img_name in dataset_images:
                in_dataset += 1
                
                # Update or create image entry
                if img_name not in review_status["images"]:
                    review_status["images"][img_name] = {}
                
                review_status["images"][img_name]["status"] = "confirmed"
                review_status["images"][img_name]["in_dataset"] = True
                confirmed += 1
            else:
                # Mark as unconfirmed if not in dataset
                if img_name not in review_status["images"]:
                    review_status["images"][img_name] = {
                        "status": "unconfirmed",
                        "in_dataset": False
                    }
                elif review_status["images"][img_name].get("status") != "confirmed":
                    review_status["images"][img_name]["status"] = "unconfirmed"
                
                if review_status["images"][img_name].get("status") == "confirmed":
                    confirmed += 1
        
        # Update summary
        unconfirmed = total - confirmed
        review_status["summary"] = {
            "total_images": total,
            "confirmed": confirmed,
            "unconfirmed": unconfirmed,
            "in_dataset": in_dataset
        }
        
        # Save updated status
        with open(review_status_path, 'w') as f:
            json.dump(review_status, f, indent=2)
        
        if unconfirmed > 0:
            print(f"⚠️  {person_id}: {unconfirmed} unconfirmed, {confirmed} confirmed")
        else:
            print(f"✅ {person_id}: All {confirmed} images confirmed")
        
        updated_count += 1
    
    print(f"\n✅ Updated {updated_count} person folders")

if __name__ == "__main__":
    update_review_status()