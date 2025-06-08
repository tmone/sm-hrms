#!/usr/bin/env python3
"""
Remove duplicate images from person folders by comparing file hashes
Keeps only one copy of each unique image
"""
import hashlib
import os
from pathlib import Path
import json
from datetime import datetime
import shutil
from collections import defaultdict

def get_file_hash(filepath):
    """Calculate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def find_duplicates_in_person(person_dir):
    """Find duplicate images in a person folder"""
    duplicates = defaultdict(list)
    
    # Get all image files
    image_files = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
    
    if not image_files:
        return duplicates
    
    print(f"\nProcessing {person_dir.name} with {len(image_files)} images...")
    
    # Calculate hash for each file
    for img_file in image_files:
        try:
            file_hash = get_file_hash(img_file)
            duplicates[file_hash].append(img_file)
        except Exception as e:
            print(f"  Error processing {img_file.name}: {e}")
    
    # Filter out non-duplicates
    actual_duplicates = {
        hash_val: files for hash_val, files in duplicates.items() 
        if len(files) > 1
    }
    
    return actual_duplicates

def remove_duplicates(person_dir, duplicates, backup_dir=None):
    """Remove duplicate images, keeping only one of each"""
    removed_count = 0
    
    for hash_val, duplicate_files in duplicates.items():
        # Sort by filename to keep consistent behavior
        duplicate_files.sort(key=lambda x: x.name)
        
        # Keep the first file, remove the rest
        keep_file = duplicate_files[0]
        remove_files = duplicate_files[1:]
        
        print(f"  Keeping: {keep_file.name}")
        
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
                print(f"  Removed: {remove_file.name}")
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
            metadata['duplicates_removed'] = original_count - new_count
            
            # Save updated metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"  Updated metadata: {original_count} -> {new_count} images")
            
    except Exception as e:
        print(f"  Error updating metadata: {e}")

def main():
    """Main function to remove duplicates from all person folders"""
    persons_dir = Path('processing/outputs/persons')
    
    if not persons_dir.exists():
        print("No persons directory found")
        return
    
    # Optional: Create backup directory
    create_backup = input("Create backup of removed files? (y/n): ").lower() == 'y'
    backup_dir = None
    
    if create_backup:
        backup_dir = Path('processing/outputs/duplicates_backup')
        backup_dir.mkdir(exist_ok=True)
        print(f"Backup directory: {backup_dir}")
    
    total_duplicates_found = 0
    total_removed = 0
    persons_with_duplicates = []
    
    # Process each person folder
    for person_dir in sorted(persons_dir.iterdir()):
        if person_dir.is_dir() and person_dir.name.startswith('PERSON-'):
            # Find duplicates
            duplicates = find_duplicates_in_person(person_dir)
            
            if duplicates:
                persons_with_duplicates.append(person_dir.name)
                
                # Count total duplicate files (excluding the ones we'll keep)
                duplicate_count = sum(len(files) - 1 for files in duplicates.values())
                total_duplicates_found += duplicate_count
                
                print(f"  Found {duplicate_count} duplicate images in {len(duplicates)} groups")
                
                # Remove duplicates
                removed_files = []
                for hash_val, files in duplicates.items():
                    removed_files.extend(files[1:])  # All except the first
                
                removed_count = remove_duplicates(person_dir, duplicates, backup_dir)
                total_removed += removed_count
                
                # Update metadata
                update_metadata_after_removal(person_dir, removed_files)
    
    # Summary
    print("\n" + "="*60)
    print(f"SUMMARY:")
    print(f"Total persons processed: {len(list(persons_dir.glob('PERSON-*')))}")
    print(f"Persons with duplicates: {len(persons_with_duplicates)}")
    print(f"Total duplicate images found: {total_duplicates_found}")
    print(f"Total images removed: {total_removed}")
    
    if persons_with_duplicates:
        print(f"\nPersons cleaned up:")
        for person in sorted(persons_with_duplicates):
            print(f"  - {person}")
    
    if backup_dir and backup_dir.exists():
        backup_size = sum(f.stat().st_size for f in backup_dir.rglob('*') if f.is_file())
        print(f"\nBackup size: {backup_size / (1024*1024):.2f} MB")

if __name__ == "__main__":
    main()