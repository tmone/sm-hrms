#!/usr/bin/env python3
"""
Remove visually duplicate images from person folders using deep learning similarity
Uses MobileNetV2 for feature extraction and visual similarity comparison
"""
import os
import sys
from pathlib import Path
import json
from datetime import datetime
import shutil
import numpy as np
import cv2
from collections import defaultdict
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

# Initialize MobileNetV2 for feature extraction
print("Loading MobileNetV2 model...")
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
print("Model loaded successfully")

def extract_features(img_path, model):
    """Extract features from an image using MobileNetV2"""
    try:
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Extract features
        features = model.predict(img_array, verbose=0)
        return features.flatten()
    except Exception as e:
        print(f"Error extracting features from {img_path}: {e}")
        return None

def calculate_visual_similarity(img1_path, img2_path):
    """Calculate visual similarity using color histograms and structure"""
    try:
        # Read images
        img1 = cv2.imread(str(img1_path))
        img2 = cv2.imread(str(img2_path))
        
        if img1 is None or img2 is None:
            return 0.0
        
        # Resize to same size for comparison
        size = (128, 128)
        img1 = cv2.resize(img1, size)
        img2 = cv2.resize(img2, size)
        
        # Calculate color histograms
        hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist1 = cv2.normalize(hist1, hist1).flatten()
        
        hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.normalize(hist2, hist2).flatten()
        
        # Calculate histogram correlation
        color_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # Calculate structural similarity
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Simple structural similarity using normalized cross-correlation
        correlation = cv2.matchTemplate(gray1, gray2, cv2.TM_CCORR_NORMED)[0][0]
        
        # Combine similarities
        visual_similarity = (color_similarity * 0.5 + correlation * 0.5)
        
        return float(visual_similarity)
        
    except Exception as e:
        print(f"Error calculating visual similarity: {e}")
        return 0.0

def find_visual_duplicates_in_person(person_dir, similarity_threshold=0.90):
    """Find visually similar duplicate images in a person folder"""
    duplicates = []
    
    # Get all image files
    image_files = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
    
    if len(image_files) < 2:
        return duplicates
    
    print(f"\nProcessing {person_dir.name} with {len(image_files)} images...")
    print("Extracting features...")
    
    # Extract features for all images
    features = {}
    for i, img_file in enumerate(image_files):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(image_files)}")
        
        feat = extract_features(img_file, base_model)
        if feat is not None:
            features[img_file] = feat
    
    print(f"  Extracted features for {len(features)} images")
    print("  Finding duplicates...")
    
    # Compare all pairs
    image_list = list(features.keys())
    checked_pairs = set()
    duplicate_groups = []
    
    for i in range(len(image_list)):
        if image_list[i] in checked_pairs:
            continue
            
        current_group = [image_list[i]]
        
        for j in range(i + 1, len(image_list)):
            if image_list[j] in checked_pairs:
                continue
            
            # Calculate MobileNetV2 similarity
            feat1 = features[image_list[i]].reshape(1, -1)
            feat2 = features[image_list[j]].reshape(1, -1)
            mobilenet_sim = cosine_similarity(feat1, feat2)[0][0]
            
            # Calculate visual similarity
            visual_sim = calculate_visual_similarity(image_list[i], image_list[j])
            
            # Combined similarity (weighted average)
            combined_similarity = (mobilenet_sim * 0.7 + visual_sim * 0.3)
            
            if combined_similarity >= similarity_threshold:
                current_group.append(image_list[j])
                checked_pairs.add(image_list[j])
        
        if len(current_group) > 1:
            duplicate_groups.append(current_group)
            for img in current_group:
                checked_pairs.add(img)
    
    return duplicate_groups

def remove_visual_duplicates(person_dir, duplicate_groups, backup_dir=None):
    """Remove visually duplicate images, keeping the best quality one"""
    removed_count = 0
    
    for group in duplicate_groups:
        # Sort by file size (larger = better quality) and name
        group.sort(key=lambda x: (-x.stat().st_size, x.name))
        
        # Keep the first (best quality) file, remove the rest
        keep_file = group[0]
        remove_files = group[1:]
        
        print(f"  Keeping: {keep_file.name} (size: {keep_file.stat().st_size})")
        
        for remove_file in remove_files:
            try:
                if backup_dir:
                    # Create backup
                    backup_path = backup_dir / person_dir.name
                    backup_path.mkdir(exist_ok=True)
                    backup_file = backup_path / remove_file.name
                    shutil.copy2(remove_file, backup_file)
                
                # Remove the duplicate
                remove_file.unlink()
                print(f"  Removed: {remove_file.name} (size: {remove_file.stat().st_size})")
                removed_count += 1
                
            except Exception as e:
                print(f"  Error removing {remove_file.name}: {e}")
    
    return removed_count

def update_metadata_after_removal(person_dir, removed_files):
    """Update metadata.json after removing duplicates"""
    metadata_path = person_dir / 'metadata.json'
    
    if not metadata_path.exists():
        return
    
    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # Get list of removed filenames
        removed_names = {f.name for f in removed_files}
        
        # Filter out removed images from metadata
        original_count = len(metadata.get('images', []))
        metadata['images'] = [
            img for img in metadata.get('images', [])
            if img['filename'] not in removed_names
        ]
        new_count = len(metadata['images'])
        
        if new_count < original_count:
            # Update counts
            metadata['total_images'] = new_count
            metadata['total_detections'] = new_count
            metadata['updated_at'] = datetime.now().isoformat()
            metadata['visual_duplicates_removed'] = original_count - new_count
            
            # Save updated metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"  Updated metadata: {original_count} -> {new_count} images")
            
    except Exception as e:
        print(f"  Error updating metadata: {e}")

def main():
    """Main function to remove visual duplicates from all person folders"""
    persons_dir = Path('processing/outputs/persons')
    
    if not persons_dir.exists():
        print("No persons directory found")
        return
    
    # Similarity threshold
    threshold = 0.90  # 90% similarity
    
    # Optional: Create backup directory
    create_backup = input("Create backup of removed files? (y/n): ").lower() == 'y'
    backup_dir = None
    
    if create_backup:
        backup_dir = Path('processing/outputs/visual_duplicates_backup')
        backup_dir.mkdir(exist_ok=True)
        print(f"Backup directory: {backup_dir}")
    
    total_duplicates_found = 0
    total_removed = 0
    persons_with_duplicates = []
    
    # Process each person folder
    for person_dir in sorted(persons_dir.iterdir()):
        if person_dir.is_dir() and person_dir.name.startswith('PERSON-'):
            # Find visual duplicates
            duplicate_groups = find_visual_duplicates_in_person(person_dir, threshold)
            
            if duplicate_groups:
                persons_with_duplicates.append(person_dir.name)
                
                # Count total duplicate files (excluding the ones we'll keep)
                duplicate_count = sum(len(group) - 1 for group in duplicate_groups)
                total_duplicates_found += duplicate_count
                
                print(f"  Found {duplicate_count} visual duplicates in {len(duplicate_groups)} groups")
                
                # Collect files to remove
                removed_files = []
                for group in duplicate_groups:
                    removed_files.extend(group[1:])  # All except the first
                
                # Remove duplicates
                removed_count = remove_visual_duplicates(person_dir, duplicate_groups, backup_dir)
                total_removed += removed_count
                
                # Update metadata
                update_metadata_after_removal(person_dir, removed_files)
    
    # Summary
    print("\n" + "="*60)
    print(f"SUMMARY:")
    print(f"Total persons processed: {len(list(persons_dir.glob('PERSON-*')))}")
    print(f"Persons with visual duplicates: {len(persons_with_duplicates)}")
    print(f"Total visual duplicates found: {total_duplicates_found}")
    print(f"Total images removed: {total_removed}")
    print(f"Similarity threshold used: {threshold * 100}%")
    
    if persons_with_duplicates:
        print(f"\nPersons cleaned up:")
        for person in sorted(persons_with_duplicates):
            print(f"  - {person}")
    
    if backup_dir and backup_dir.exists():
        backup_size = sum(f.stat().st_size for f in backup_dir.rglob('*') if f.is_file())
        print(f"\nBackup size: {backup_size / (1024*1024):.2f} MB")

if __name__ == "__main__":
    main()