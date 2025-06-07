#!/usr/bin/env python3
"""
Verify that person recognition is working correctly in video processing
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import json

def check_recognition_results():
    """Check if recognition is properly using recognized IDs"""
    print("Verifying Recognition Fix\n")
    
    persons_dir = Path("processing/outputs/persons")
    if not persons_dir.exists():
        print("ERROR: No persons directory found")
        return
    
    # Get all person folders
    person_folders = sorted([f for f in persons_dir.iterdir() if f.is_dir() and f.name.startswith('PERSON-')])
    
    print(f"Found {len(person_folders)} person folders\n")
    
    # Check recent folders
    recent_folders = sorted(person_folders, key=lambda x: x.stat().st_mtime, reverse=True)[:10]
    
    recognized_count = 0
    new_count = 0
    
    for folder in recent_folders:
        metadata_file = folder / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
            
            if metadata.get("recognized"):
                recognized_count += 1
                original_id = metadata.get("original_tracking_id", "?")
                confidence = metadata.get("recognition_confidence", 0)
                print(f"SUCCESS: {folder.name} - Recognized (was tracking ID {original_id}, confidence: {confidence:.2%})")
            else:
                new_count += 1
                print(f"NEW: {folder.name} - New person (not recognized)")
    
    print(f"\nSummary:")
    print(f"  Recognized persons: {recognized_count}")
    print(f"  New persons: {new_count}")
    print(f"  Total: {recognized_count + new_count}")
    
    if recognized_count > 0:
        print("\nSUCCESS: Recognition is working correctly!")
        print("Known persons are being saved with their recognized IDs.")
    else:
        print("\nWARNING: No recognized persons found.")
        print("This could mean:")
        print("1. All persons in the video are truly new")
        print("2. Recognition confidence is below 80% threshold")
        print("3. Recognition is still not working properly")
    
    # Check for specific person IDs that should be recognized
    expected_ids = ["PERSON-0001", "PERSON-0002", "PERSON-0007", "PERSON-0008", "PERSON-0010", "PERSON-0017", "PERSON-0019", "PERSON-0021"]
    found_ids = [f.name for f in person_folders if f.name in expected_ids]
    
    if found_ids:
        print(f"\nFound expected person IDs: {', '.join(found_ids)}")
        print("These should be reused for known persons, not creating new IDs.")

if __name__ == "__main__":
    check_recognition_results()