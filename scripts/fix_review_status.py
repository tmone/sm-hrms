#!/usr/bin/env python3
"""
Fix review status by recalculating based on actual images in folders
This ensures the counts are accurate after deletions
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import json
from datetime import datetime

def fix_review_status_for_person(person_dir):
    """Fix review status for a single person"""
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
    
    # Get all actual images in folder
    actual_images = set(img.name for img in person_dir.glob("*.jpg"))
    
    # Get images in dataset
    dataset_dir = Path("datasets") / person_id
    dataset_images = set()
    if dataset_dir.exists():
        dataset_images = set(img.name for img in dataset_dir.glob("*.jpg"))
    
    # Clean up review status - remove entries for non-existent images
    to_remove = []
    for img_name in review_status['images']:
        if img_name not in actual_images:
            to_remove.append(img_name)
    
    for img_name in to_remove:
        del review_status['images'][img_name]
    
    # Ensure all actual images have entries
    for img_name in actual_images:
        if img_name not in review_status['images']:
            review_status['images'][img_name] = {
                'status': 'unconfirmed',
                'in_dataset': False
            }
        
        # Update in_dataset status
        if img_name in dataset_images:
            review_status['images'][img_name]['in_dataset'] = True
            # If in dataset but not marked confirmed, mark it now
            if review_status['images'][img_name].get('status') != 'confirmed':
                review_status['images'][img_name]['status'] = 'confirmed'
                review_status['images'][img_name]['auto_confirmed'] = True
                review_status['images'][img_name]['confirmed_at'] = datetime.now().isoformat()
    
    # Recalculate summary
    total = len(actual_images)
    confirmed = 0
    unconfirmed = 0
    in_dataset = 0
    
    for img_name in actual_images:
        img_data = review_status['images'][img_name]
        if img_data.get('status') == 'confirmed':
            confirmed += 1
        else:
            unconfirmed += 1
        
        if img_data.get('in_dataset'):
            in_dataset += 1
    
    review_status['summary'] = {
        'total_images': total,
        'confirmed': confirmed,
        'unconfirmed': unconfirmed,
        'in_dataset': in_dataset,
        'last_updated': datetime.now().isoformat()
    }
    
    # Save fixed status
    with open(review_status_path, 'w') as f:
        json.dump(review_status, f, indent=2)
    
    return {
        'total': total,
        'confirmed': confirmed,
        'unconfirmed': unconfirmed,
        'in_dataset': in_dataset
    }

def fix_all_review_statuses():
    """Fix review status for all persons"""
    print("[CONFIG] Fixing review statuses...\n")
    
    persons_dir = Path("processing/outputs/persons")
    if not persons_dir.exists():
        print("[ERROR] No persons directory found")
        return
    
    fixed_count = 0
    all_confirmed_count = 0
    
    for person_dir in persons_dir.iterdir():
        if not person_dir.is_dir() or not person_dir.name.startswith("PERSON-"):
            continue
        
        result = fix_review_status_for_person(person_dir)
        fixed_count += 1
        
        if result['unconfirmed'] == 0 and result['total'] > 0:
            all_confirmed_count += 1
            print(f"[OK] {person_dir.name}: All {result['confirmed']} images confirmed")
        elif result['unconfirmed'] > 0:
            print(f"[WARNING]  {person_dir.name}: {result['unconfirmed']} unconfirmed, {result['confirmed']} confirmed")
        else:
            print(f"[FILE] {person_dir.name}: No images")
    
    print(f"\n[INFO] Summary:")
    print(f"   Fixed: {fixed_count} persons")
    print(f"   All confirmed: {all_confirmed_count} persons")
    print(f"\n[OK] Review statuses fixed!")

if __name__ == "__main__":
    fix_all_review_statuses()