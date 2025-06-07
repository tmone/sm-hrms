#!/usr/bin/env python3
"""
Remove similar/duplicate images to improve training performance
Uses perceptual hashing and feature similarity
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from pathlib import Path
import json
import shutil
from datetime import datetime
import imagehash
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

def compute_image_hash(image_path, hash_size=16):
    """Compute perceptual hash of image"""
    try:
        img = Image.open(image_path)
        # Use different hash methods for better accuracy
        ahash = imagehash.average_hash(img, hash_size=hash_size)
        phash = imagehash.phash(img, hash_size=hash_size)
        dhash = imagehash.dhash(img, hash_size=hash_size)
        return {
            'ahash': str(ahash),
            'phash': str(phash),
            'dhash': str(dhash),
            'combined': f"{ahash}{phash}{dhash}"
        }
    except Exception as e:
        print(f"Error hashing {image_path}: {e}")
        return None

def extract_histogram_features(image_path):
    """Extract color histogram features for similarity comparison"""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        # Calculate histograms for each channel
        hist_features = []
        for i in range(3):  # BGR channels
            hist = cv2.calcHist([img], [i], None, [64], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            hist_features.extend(hist)
        
        return np.array(hist_features)
    except Exception as e:
        print(f"Error extracting features from {image_path}: {e}")
        return None

def find_similar_images(person_dir, similarity_threshold=0.95, hash_threshold=5):
    """Find similar images in a person directory"""
    image_paths = list(person_dir.glob("*.jpg"))
    if len(image_paths) < 2:
        return []
    
    print(f"  Analyzing {len(image_paths)} images...")
    
    # Compute hashes and features for all images
    image_data = {}
    for img_path in image_paths:
        hash_data = compute_image_hash(img_path)
        features = extract_histogram_features(img_path)
        
        if hash_data and features is not None:
            image_data[img_path] = {
                'hash': hash_data,
                'features': features,
                'size': img_path.stat().st_size,
                'mtime': img_path.stat().st_mtime
            }
    
    # Find similar pairs
    similar_groups = []
    processed = set()
    
    for img1, data1 in image_data.items():
        if img1 in processed:
            continue
            
        similar_group = [img1]
        processed.add(img1)
        
        for img2, data2 in image_data.items():
            if img2 == img1 or img2 in processed:
                continue
            
            # Check perceptual hash similarity
            hash_similar = False
            for hash_type in ['ahash', 'phash', 'dhash']:
                hash1 = imagehash.hex_to_hash(data1['hash'][hash_type])
                hash2 = imagehash.hex_to_hash(data2['hash'][hash_type])
                if hash1 - hash2 <= hash_threshold:
                    hash_similar = True
                    break
            
            # Check feature similarity
            if hash_similar or data1['features'] is not None and data2['features'] is not None:
                similarity = cosine_similarity(
                    data1['features'].reshape(1, -1),
                    data2['features'].reshape(1, -1)
                )[0][0]
                
                if similarity >= similarity_threshold or hash_similar:
                    similar_group.append(img2)
                    processed.add(img2)
        
        if len(similar_group) > 1:
            similar_groups.append(similar_group)
    
    return similar_groups

def remove_similar_images_from_person(person_dir, dry_run=True):
    """Remove similar images from a person directory"""
    similar_groups = find_similar_images(person_dir)
    
    if not similar_groups:
        return 0
    
    removed_count = 0
    removed_dir = person_dir / "removed_similar"
    
    if not dry_run:
        removed_dir.mkdir(exist_ok=True)
    
    for group in similar_groups:
        # Sort by file size (keep larger) then by modification time (keep newer)
        group.sort(key=lambda p: (-p.stat().st_size, -p.stat().st_mtime))
        
        # Keep the first one, remove others
        keep = group[0]
        remove = group[1:]
        
        print(f"  Found {len(group)} similar images, keeping: {keep.name}")
        
        for img_path in remove:
            print(f"    - Removing: {img_path.name}")
            removed_count += 1
            
            if not dry_run:
                # Move to removed folder instead of deleting
                dest = removed_dir / img_path.name
                shutil.move(str(img_path), str(dest))
    
    return removed_count

def process_all_persons(dry_run=True):
    """Process all person directories to remove similar images"""
    print("[SEARCH] Removing similar images to improve training performance\n")
    
    if dry_run:
        print("[PERSON] DRY RUN MODE - No files will be actually removed\n")
    
    persons_dir = Path("processing/outputs/persons")
    if not persons_dir.exists():
        print("[ERROR] No persons directory found")
        return
    
    total_removed = 0
    persons_processed = 0
    
    # Process each person
    for person_dir in persons_dir.iterdir():
        if not person_dir.is_dir() or not person_dir.name.startswith("PERSON-"):
            continue
        
        print(f"\n[FILE] Processing {person_dir.name}...")
        removed = remove_similar_images_from_person(person_dir, dry_run)
        
        if removed > 0:
            total_removed += removed
            persons_processed += 1
            print(f"  [OK] Would remove {removed} similar images")
    
    print(f"\n[INFO] Summary:")
    print(f"   Persons processed: {persons_processed}")
    print(f"   Similar images found: {total_removed}")
    
    if dry_run and total_removed > 0:
        print(f"\n[TIP] To actually remove similar images, run:")
        print(f"   python {__file__} --remove")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Remove similar images from person folders')
    parser.add_argument('--remove', action='store_true', help='Actually remove files (default is dry run)')
    parser.add_argument('--threshold', type=float, default=0.95, help='Similarity threshold (0-1)')
    args = parser.parse_args()
    
    process_all_persons(dry_run=not args.remove)

if __name__ == "__main__":
    main()