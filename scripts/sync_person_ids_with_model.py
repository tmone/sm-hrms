#!/usr/bin/env python3
"""
Sync existing PERSON IDs with the recognition model's persons.json file.
This ensures that recognized persons will reuse their existing PERSON IDs.
"""

import sys
import os
import json
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def sync_person_ids():
    """Sync PERSON folder IDs with the recognition model's persons.json"""
    
    # Get the default model
    config_path = Path('models/person_recognition/config.json')
    if not config_path.exists():
        logger.error("No person recognition config found")
        return
        
    with open(config_path) as f:
        config = json.load(f)
        default_model = config.get('default_model')
        
    if not default_model:
        logger.error("No default model configured")
        return
        
    model_dir = Path('models/person_recognition') / default_model
    persons_file = model_dir / 'persons.json'
    
    # Load existing persons.json
    persons_data = {}
    if persons_file.exists():
        with open(persons_file) as f:
            persons_data = json.load(f)
            
    logger.info(f"Loaded {len(persons_data)} persons from model")
    
    # Scan PERSON folders
    persons_dir = Path('processing/outputs/persons')
    if not persons_dir.exists():
        logger.warning("No persons directory found")
        return
        
    person_folders = list(persons_dir.glob('PERSON-*'))
    logger.info(f"Found {len(person_folders)} PERSON folders")
    
    # Update persons.json with PERSON IDs
    updates = 0
    for folder in person_folders:
        person_id = folder.name  # e.g., "PERSON-0001"
        
        # Check if this person is in the model
        for person_name, person_info in persons_data.items():
            # Match by checking if the person folder contains images for this person
            # This is a simple heuristic - in production you'd want more sophisticated matching
            metadata_file = folder / 'metadata.json'
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    
                # Check if this folder's person_id matches the one in persons.json
                if 'person_id' in metadata and metadata['person_id'] == person_id:
                    # Update the persons.json entry with the PERSON ID
                    if 'person_id' not in person_info or person_info['person_id'] != person_id:
                        person_info['person_id'] = person_id
                        updates += 1
                        logger.info(f"Updated {person_name} with PERSON ID: {person_id}")
                        
    # For any person in the model without a PERSON ID, try to find their folder
    for person_name, person_info in persons_data.items():
        if 'person_id' not in person_info:
            # Try to find a matching PERSON folder
            # This would require more sophisticated matching logic
            # For now, just log it
            logger.warning(f"Person {person_name} in model has no PERSON ID assigned")
            
    # Save updated persons.json
    if updates > 0:
        with open(persons_file, 'w') as f:
            json.dump(persons_data, f, indent=2)
        logger.info(f"Updated {updates} person entries with PERSON IDs")
    else:
        logger.info("No updates needed")
        
    # Also create a reverse mapping file for quick lookups
    reverse_mapping = {}
    for person_name, person_info in persons_data.items():
        if 'person_id' in person_info:
            reverse_mapping[person_info['person_id']] = person_name
            
    mapping_file = model_dir / 'person_id_mapping.json'
    with open(mapping_file, 'w') as f:
        json.dump(reverse_mapping, f, indent=2)
    logger.info(f"Created reverse mapping file: {mapping_file}")


if __name__ == "__main__":
    sync_person_ids()