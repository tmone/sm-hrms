#!/usr/bin/env python3
"""
Verify that the correct recognition model is being loaded
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import json

def check_model_configuration():
    """Check which model is configured as default"""
    config_path = Path("models/person_recognition/config.json")
    
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        
        print(f"Default model: {config.get('default_model')}")
        print(f"Last updated: {config.get('last_updated')}")
        print(f"Note: {config.get('note')}")
        
        # Check if the model exists
        model_path = Path(f"models/person_recognition/{config.get('default_model')}")
        if model_path.exists():
            print(f"\n✓ Model directory exists: {model_path}")
            
            # Check model files
            files = list(model_path.glob("*"))
            print(f"\nModel files:")
            for f in files:
                print(f"  - {f.name} ({f.stat().st_size} bytes)")
            
            # Check person mappings
            mapping_file = model_path / "person_id_mapping.pkl"
            if mapping_file.exists():
                import pickle
                mappings = pickle.load(open(mapping_file, 'rb'))
                print(f"\nPerson ID mappings ({len(mappings)} persons):")
                for class_id, person_id in mappings.items():
                    print(f"  Class {class_id} -> {person_id}")
        else:
            print(f"\n✗ Model directory NOT found: {model_path}")
    else:
        print("✗ Config file not found!")

def test_model_loading():
    """Test actual model loading"""
    try:
        from hr_management.processing.person_recognition_inference_simple import PersonRecognitionInferenceSimple
        
        # Get model name from config
        config_path = Path("models/person_recognition/config.json")
        with open(config_path) as f:
            config = json.load(f)
        model_name = config.get('default_model')
        
        print(f"\nTesting model loading: {model_name}")
        recognizer = PersonRecognitionInferenceSimple(
            model_name=model_name,
            confidence_threshold=0.8
        )
        
        print("✓ Model loaded successfully!")
        print(f"  Model: {recognizer.model}")
        print(f"  Scaler: {recognizer.scaler}")
        print(f"  Person mappings: {recognizer.person_id_mapping}")
        
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=== Checking Model Configuration ===")
    check_model_configuration()
    
    print("\n=== Testing Model Loading ===")
    test_model_loading()