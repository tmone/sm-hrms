#!/usr/bin/env python3
"""
Update all person metadata files to include ALL images, not just first 100
"""
from pathlib import Path
import json
from datetime import datetime

def update_all_metadata():
    persons_dir = Path('processing/outputs/persons')
    if not persons_dir.exists():
        print("No persons directory found")
        return
    
    updated_count = 0
    
    for person_dir in persons_dir.iterdir():
        if person_dir.is_dir() and person_dir.name.startswith('PERSON-'):
            metadata_path = person_dir / 'metadata.json'
            
            # Get all image files
            image_files = sorted(list(person_dir.glob('*.jpg')) + list(person_dir.glob('*.png')))
            
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                
                # Check if we need to update
                current_image_count = len(metadata.get('images', []))
                actual_image_count = len(image_files)
                
                if current_image_count < actual_image_count:
                    print(f"\nUpdating {person_dir.name}:")
                    print(f"  Current metadata images: {current_image_count}")
                    print(f"  Actual images in folder: {actual_image_count}")
                    
                    # Update images list with ALL images
                    metadata['images'] = []
                    for img_file in image_files:
                        metadata['images'].append({
                            'filename': img_file.name,
                            'confidence': 0.95  # Default confidence
                        })
                    
                    # Update counts
                    metadata['total_images'] = actual_image_count
                    metadata['total_detections'] = actual_image_count
                    metadata['updated_at'] = datetime.now().isoformat()
                    
                    # Save updated metadata
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    print(f"  Updated to include all {actual_image_count} images")
                    updated_count += 1
    
    print(f"\nTotal metadata files updated: {updated_count}")

if __name__ == "__main__":
    update_all_metadata()