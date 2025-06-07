#!/usr/bin/env python3
"""
Check person data quality - unconfirmed images and duplicates
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import json

def check_person_quality():
    """Check for quality issues in person data"""
    print("🔍 Checking Person Data Quality\n")
    
    persons_dir = Path("processing/outputs/persons")
    if not persons_dir.exists():
        print("❌ No persons directory found")
        return
    
    issues = {
        "unconfirmed_images": [],
        "similar_images": [],
        "misrecognized": [],
        "total_persons": 0,
        "total_images": 0
    }
    
    # Check each person
    for person_dir in persons_dir.iterdir():
        if not person_dir.is_dir() or not person_dir.name.startswith("PERSON-"):
            continue
        
        issues["total_persons"] += 1
        
        # Load metadata
        metadata_path = person_dir / "metadata.json"
        if not metadata_path.exists():
            continue
            
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # Check if this was a recognized person
        if metadata.get("recognized"):
            original_id = metadata.get("original_tracking_id")
            confidence = metadata.get("recognition_confidence", 0)
            
            # Flag potential misrecognitions (high confidence but many unconfirmed)
            review_status_path = person_dir / "review_status.json"
            if review_status_path.exists():
                with open(review_status_path) as f:
                    review_status = json.load(f)
                summary = review_status.get("summary", {})
                unconfirmed = summary.get("unconfirmed", 0)
                total = summary.get("total_images", 0)
                
                if unconfirmed > 0:
                    issues["unconfirmed_images"].append({
                        "person_id": person_dir.name,
                        "unconfirmed": unconfirmed,
                        "total": total,
                        "recognized": True,
                        "confidence": confidence
                    })
                    
                    # High confidence but many unconfirmed = potential misrecognition
                    if confidence > 0.8 and unconfirmed / total > 0.5:
                        issues["misrecognized"].append({
                            "person_id": person_dir.name,
                            "confidence": confidence,
                            "unconfirmed_ratio": unconfirmed / total
                        })
        
        # Count images
        image_count = len(list(person_dir.glob("*.jpg")))
        issues["total_images"] += image_count
    
    # Print summary
    print("📊 Summary:")
    print(f"   Total persons: {issues['total_persons']}")
    print(f"   Total images: {issues['total_images']}")
    print(f"   Average images per person: {issues['total_images'] / issues['total_persons']:.1f}")
    
    if issues["unconfirmed_images"]:
        print(f"\n⚠️  Persons with unconfirmed images: {len(issues['unconfirmed_images'])}")
        for data in sorted(issues["unconfirmed_images"], key=lambda x: x["unconfirmed"], reverse=True)[:10]:
            print(f"   - {data['person_id']}: {data['unconfirmed']}/{data['total']} unconfirmed")
            if data["recognized"]:
                print(f"     (Recognized with {data['confidence']:.1%} confidence)")
    
    if issues["misrecognized"]:
        print(f"\n🚨 Potential misrecognitions: {len(issues['misrecognized'])}")
        for data in issues["misrecognized"]:
            print(f"   - {data['person_id']}: {data['confidence']:.1%} confidence but "
                  f"{data['unconfirmed_ratio']:.1%} unconfirmed")
            print(f"     → Should review and possibly move to correct person")
    
    print("\n💡 Recommended actions:")
    print("1. Run: python scripts/mark_unconfirmed_images.py")
    print("2. Run: python scripts/remove_similar_images.py --threshold 0.95")
    print("3. Review persons with high unconfirmed ratios in the web UI")
    print("4. Use the Review button to confirm/move/delete misrecognized images")

if __name__ == "__main__":
    check_person_quality()