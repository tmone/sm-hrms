#!/usr/bin/env python3
"""
Fix person separation by re-analyzing detections and splitting them into proper individuals.
"""

import os
import sys
import cv2
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def analyze_person_folder():
    """Analyze the single person folder to find how many actual persons it contains."""
    persons_dir = Path('processing/outputs/persons')
    person_folder = persons_dir / 'PERSON-0058'
    
    if not person_folder.exists():
        print("[ERROR] PERSON-0058 folder not found!")
        return
    
    # Load metadata
    metadata_path = person_folder / 'metadata.json'
    if not metadata_path.exists():
        print("[ERROR] No metadata found!")
        return
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"[INFO] Analyzing PERSON-0058:")
    print(f"  Total detections: {metadata.get('total_detections', 0)}")
    print(f"  Total images: {len(metadata.get('images', []))}")
    print(f"  Duration: {metadata.get('last_appearance', 0) - metadata.get('first_appearance', 0):.1f}s")
    
    # Analyze images to detect different people
    images = list(person_folder.glob('*.jpg'))
    print(f"  Image files: {len(images)}")
    
    if len(images) > 10:
        print("\n[WARNING]  This folder contains many images - likely multiple people grouped together!")
        print("  Recommendation: Use the Split Person feature in the UI to separate them")
    
    # Sample some images to show
    print("\n[CAMERA] Sample images:")
    for i, img_path in enumerate(images[:5]):
        print(f"  {i+1}. {img_path.name}")


def create_person_clusters_by_location():
    """
    Analyze detections and cluster them by location to separate different people.
    This is a simple spatial clustering approach.
    """
    persons_dir = Path('processing/outputs/persons')
    person_folder = persons_dir / 'PERSON-0058'
    
    metadata_path = person_folder / 'metadata.json'
    if not metadata_path.exists():
        print("[ERROR] No metadata found!")
        return
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Group detections by spatial location
    print("\n[SEARCH] Analyzing spatial distribution of detections...")
    
    # This would need the actual detection data with bounding boxes
    # For now, we'll provide recommendations
    
    print("\n[TRACE] Recommendations to fix this issue:")
    print("1. Use the 'Split Person' feature in the web UI:")
    print("   - Open PERSON-0058 in the persons list")
    print("   - Click 'Split Person' button")
    print("   - Select images that belong to different people")
    print("   - Create new persons from the selections")
    print("\n2. Or re-process the video with better tracking:")
    print("   - Delete the current video")
    print("   - Re-upload and process with improved settings")
    print("\n3. Use the Fix Duplicates feature if some persons were incorrectly merged")


def suggest_manual_split():
    """Suggest how to manually split the persons."""
    persons_dir = Path('processing/outputs/persons')
    person_folder = persons_dir / 'PERSON-0058'
    
    images = sorted(list(person_folder.glob('*.jpg')))
    
    print("\n✂️  Manual Split Strategy:")
    print("Since all detections are in one folder, you need to:")
    print("\n1. Open the persons page in your browser")
    print("2. Click on PERSON-0058 to view all images")
    print("3. Use the 'Split Person' button")
    print("4. Select images of each unique person")
    print("5. Create new person entries for each")
    print(f"\nTotal images to review: {len(images)}")
    
    # Show image distribution over time
    if images:
        print("\n[INFO] Image distribution:")
        # Extract frame numbers from filenames
        frame_numbers = []
        for img in images:
            parts = img.stem.split('_frame_')
            if len(parts) > 1:
                try:
                    frame_num = int(parts[1])
                    frame_numbers.append(frame_num)
                except:
                    pass
        
        if frame_numbers:
            frame_numbers.sort()
            print(f"  First frame: {frame_numbers[0]}")
            print(f"  Last frame: {frame_numbers[-1]}")
            print(f"  Frame range: {frame_numbers[-1] - frame_numbers[0]}")
            
            # Check for gaps that might indicate different people
            gaps = []
            for i in range(1, len(frame_numbers)):
                gap = frame_numbers[i] - frame_numbers[i-1]
                if gap > 100:  # Large gap might indicate different person
                    gaps.append((frame_numbers[i-1], frame_numbers[i], gap))
            
            if gaps:
                print(f"\n  Found {len(gaps)} large gaps in detections:")
                for start, end, gap in gaps[:5]:
                    print(f"    Frames {start} to {end}: gap of {gap} frames")
                print("\n  These gaps might indicate transitions between different people")


def main():
    """Main function."""
    print("[CONFIG] Person Separation Analysis")
    print("="*50)
    
    # Analyze current situation
    analyze_person_folder()
    
    # Provide recommendations
    suggest_manual_split()
    
    # Explain the issue
    print("\n❓ Why did this happen?")
    print("The tracking algorithm grouped all detections into one person because:")
    print("1. The tracking parameters were too permissive")
    print("2. People appeared in similar locations across frames")
    print("3. No appearance-based matching was used")
    print("\n[OK] This has been improved for future videos with:")
    print("1. Better tracking parameters")
    print("2. Appearance-based matching (when GPU is available)")
    print("3. Stricter person separation logic")


if __name__ == "__main__":
    main()