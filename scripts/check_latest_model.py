#!/usr/bin/env python3
"""
Check the latest model and its accuracy
"""

import json
from pathlib import Path

def check_latest_model():
    """Check the current default model and its accuracy"""
    
    # Read config
    config_path = Path('models/person_recognition/config.json')
    if not config_path.exists():
        print("No model configuration found")
        return
    
    with open(config_path) as f:
        config = json.load(f)
    
    default_model = config.get('default_model')
    if not default_model:
        print("No default model set")
        return
    
    print(f"Current default model: {default_model}")
    
    # Check if it was auto-refined
    if config.get('auto_refined'):
        print(f"Model was auto-refined using: {config.get('refinement_type')}")
    
    print(f"Last updated: {config.get('last_updated', 'Unknown')}")
    
    # Load model metadata
    model_path = Path('models/person_recognition') / default_model / 'metadata.json'
    if not model_path.exists():
        print(f"Model metadata not found at {model_path}")
        return
    
    with open(model_path) as f:
        metadata = json.load(f)
    
    print(f"\nModel Details:")
    print(f"Type: {metadata.get('model_type')}")
    print(f"Accuracy: {metadata.get('test_score', 0):.1%}")
    print(f"Number of persons: {len(metadata.get('person_ids', []))}")
    
    if 'original_model' in metadata:
        print(f"Refined from: {metadata['original_model']}")
        print(f"Improvement: {metadata.get('improvement', 0):.1%}")
    
    print(f"\nPersons in model:")
    for person in metadata.get('person_ids', []):
        print(f"  - {person}")


if __name__ == '__main__':
    check_latest_model()