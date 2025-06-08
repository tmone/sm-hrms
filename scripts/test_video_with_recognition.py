#!/usr/bin/env python3
"""
Test video processing with recognition enabled
Shows whether the system recognizes known persons
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import json

def check_recognition_status():
    """Check if recognition is working in video processing"""
    print("🧪 Testing Video Processing with Recognition\n")
    
    # Check latest processing results
    persons_dir = Path("processing/outputs/persons")
    if not persons_dir.exists():
        print("[ERROR] No persons directory found. Process a video first.")
        return
    
    # Get recent person folders
    person_folders = sorted(
        [f for f in persons_dir.iterdir() if f.is_dir() and f.name.startswith('PERSON-')],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )[:10]  # Last 10 persons
    
    print(f"[FILE] Checking {len(person_folders)} recent person folders...\n")
    
    recognized_count = 0
    new_count = 0
    
    for folder in person_folders:
        metadata_file = folder / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
            
            # Check if this person was recognized
            recognized = False
            for img in metadata.get("images", []):
                if img.get("recognized_as"):
                    recognized = True
                    recognized_count += 1
                    print(f"[OK] {folder.name} -> Recognized as {img['recognized_as']} "
                          f"(confidence: {img.get('recognition_confidence', 0):.2%})")
                    break
            
            if not recognized:
                new_count += 1
                print(f"[NEW] {folder.name} -> New person (not recognized)")
    
    print(f"\n[INFO] Summary:")
    print(f"   Recognized: {recognized_count}")
    print(f"   New persons: {new_count}")
    print(f"   Total: {recognized_count + new_count}")
    
    if recognized_count == 0 and new_count > 0:
        print("\n[WARNING]  Recognition may not be working properly!")
        print("   All persons are being created as new.")
        print("\n[TIP] Possible issues:")
        print("   1. NumPy version incompatibility")
        print("   2. Recognition model not loaded")
        print("   3. Confidence threshold too high")
    elif recognized_count > 0:
        print("\n[OK] Recognition is working!")
        print("   Known persons are being recognized correctly.")

if __name__ == "__main__":
    check_recognition_status()