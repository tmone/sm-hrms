#!/usr/bin/env python3
"""
Fix person IDs after GPU detection by running recognition
This is easier than modifying GPU detection code
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import json
import shutil
from pathlib import Path
from datetime import datetime

print("🔧 Fixing Person IDs with Recognition\n")

# Load recognition model
try:
    from processing.simple_person_recognition_inference import SimplePersonRecognitionInference
    recognizer = SimplePersonRecognitionInference()
    
    if recognizer.inference is None:
        print("❌ Recognition model not loaded - cannot fix IDs")
        print("The model needs to be retrained or fixed")
        sys.exit(1)
    else:
        print("✅ Recognition model loaded")
except Exception as e:
    print(f"❌ Failed to load recognition: {e}")
    sys.exit(1)

# Find recent person folders
persons_dir = Path('processing/outputs/persons')
if not persons_dir.exists():
    print("❌ No persons directory found")
    sys.exit(1)

# Get person folders sorted by creation time (newest first)
person_folders = sorted(
    [f for f in persons_dir.iterdir() if f.is_dir() and f.name.startswith('PERSON-')],
    key=lambda x: x.stat().st_mtime,
    reverse=True
)

print(f"Found {len(person_folders)} person folders")

# Process recent folders (e.g., from last processing)
recent_folders = person_folders[:10]  # Process last 10 persons

print(f"\nProcessing {len(recent_folders)} recent person folders...")

# Track merges
merge_map = {}  # new_id -> recognized_id

for folder in recent_folders:
    person_id = folder.name
    print(f"\n📁 Processing {person_id}...")
    
    # Get images from folder
    images = list(folder.glob('*.jpg')) + list(folder.glob('*.png'))
    
    if not images:
        print(f"   No images found")
        continue
        
    # Try recognition on multiple images for better accuracy
    recognition_results = []
    
    for img_path in images[:5]:  # Test up to 5 images
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            result = recognizer.predict_single(img)
            
            if result and result.get('person_id') != 'unknown':
                recognition_results.append({
                    'person_id': result['person_id'],
                    'confidence': result.get('confidence', 0)
                })
        except Exception as e:
            continue
    
    if recognition_results:
        # Get most common recognition
        from collections import Counter
        person_counts = Counter(r['person_id'] for r in recognition_results)
        recognized_id, count = person_counts.most_common(1)[0]
        
        # Calculate average confidence
        avg_confidence = sum(r['confidence'] for r in recognition_results if r['person_id'] == recognized_id) / count
        
        if avg_confidence > 0.7 and recognized_id != person_id:
            print(f"   🎯 Recognized as {recognized_id} (confidence: {avg_confidence:.2f})")
            merge_map[person_id] = recognized_id
            
            # Check if target folder exists
            target_folder = persons_dir / recognized_id
            
            if target_folder.exists():
                print(f"   📂 Merging into existing {recognized_id} folder")
                
                # Move images to target folder
                moved_count = 0
                for img_path in images:
                    new_name = f"{recognized_id}_{img_path.name}"
                    target_path = target_folder / new_name
                    
                    try:
                        shutil.move(str(img_path), str(target_path))
                        moved_count += 1
                    except Exception as e:
                        print(f"   ⚠️ Failed to move {img_path.name}: {e}")
                
                print(f"   ✅ Moved {moved_count} images")
                
                # Remove empty folder
                try:
                    shutil.rmtree(folder)
                    print(f"   🗑️ Removed duplicate folder {person_id}")
                except:
                    pass
                    
            else:
                print(f"   📝 Renaming folder to {recognized_id}")
                try:
                    folder.rename(target_folder)
                except Exception as e:
                    print(f"   ❌ Failed to rename: {e}")
        else:
            print(f"   ❓ Not recognized or low confidence")
    else:
        print(f"   ❓ No recognition results")

# Summary
print(f"\n📊 Summary:")
print(f"   Processed: {len(recent_folders)} folders")
print(f"   Merged: {len(merge_map)} duplicates")

if merge_map:
    print(f"\n🔄 Merges performed:")
    for old_id, new_id in merge_map.items():
        print(f"   {old_id} → {new_id}")

print("\n✅ Person ID fixing complete!")

# Save merge log
if merge_map:
    log_file = persons_dir / f"merge_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'merges': merge_map,
            'folders_processed': len(recent_folders)
        }, f, indent=2)
    print(f"\n📝 Merge log saved to: {log_file}")