#!/usr/bin/env python3
"""Test the complete recognition flow to see where it breaks"""

import sys
import os
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_recognition_modules():
    """Test each module in the recognition flow"""
    
    print("🧪 Testing Recognition Flow\n")
    
    # 1. Test SimplePersonRecognitionInference
    print("1. Testing SimplePersonRecognitionInference...")
    try:
        from processing.simple_person_recognition_inference import SimplePersonRecognitionInference
        recognizer = SimplePersonRecognitionInference()
        
        if recognizer.inference is None:
            print("   [ERROR] Recognition model failed to load")
        else:
            print("   [OK] Recognition model loaded")
            
    except Exception as e:
        print(f"   [ERROR] Failed to import: {e}")
        
    # 2. Test SharedStateManager
    print("\n2. Testing ImprovedSharedStateManagerV3...")
    try:
        from processing.shared_state_manager_improved import ImprovedSharedStateManagerV3
        state_manager = ImprovedSharedStateManagerV3()
        
        print(f"   [OK] SharedStateManager created")
        print(f"   - Starting person counter: {state_manager.person_counter}")
        print(f"   - Recognized mappings: {state_manager.recognized_to_person_id}")
        
    except Exception as e:
        print(f"   [ERROR] Failed to import: {e}")
        
    # 3. Test PersonIDManager
    print("\n3. Testing PersonIDManager...")
    try:
        from processing.person_id_manager import get_person_id_manager
        person_id_manager = get_person_id_manager()
        
        print(f"   [OK] PersonIDManager created")
        print(f"   - Next person ID: PERSON-{person_id_manager.next_person_id:04d}")
        print(f"   - Loaded mappings: {len(person_id_manager.recognized_to_person_id)}")
        
        # Show some mappings
        if person_id_manager.recognized_to_person_id:
            print("   - Sample mappings:")
            for name, pid in list(person_id_manager.recognized_to_person_id.items())[:3]:
                print(f"     {name} -> {pid}")
                
    except Exception as e:
        print(f"   [ERROR] Failed to import: {e}")
        
    # 4. Test recognition model files
    print("\n4. Checking recognition model files...")
    config_path = Path('models/person_recognition/config.json')
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            
        default_model = config.get('default_model')
        print(f"   Default model: {default_model}")
        
        if default_model:
            model_dir = Path('models/person_recognition') / default_model
            print(f"   Model directory: {model_dir}")
            
            # Check required files
            required_files = {
                'model.pkl': 'Main model file',
                'scaler.pkl': 'Feature scaler',
                'label_encoder.pkl': 'Label encoder for person IDs',
                'persons.json': 'Person ID mappings'
            }
            
            for file_name, description in required_files.items():
                file_path = model_dir / file_name
                if file_path.exists():
                    print(f"   [OK] {file_name}: {description}")
                else:
                    print(f"   [ERROR] {file_name}: {description} - MISSING")
                    
            # Check metadata
            metadata_path = model_dir / 'metadata.json'
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                print(f"\n   Model contains {metadata.get('num_persons', 0)} persons:")
                for pid in metadata.get('person_ids', [])[:5]:
                    print(f"     - {pid}")
                    
    # 5. Test the flow
    print("\n5. Testing recognition flow...")
    try:
        # Simulate what happens during video processing
        state_manager = ImprovedSharedStateManagerV3()
        
        # Simulate a recognized person
        test_recognized_id = "PERSON-0001"  # This should be in the model
        test_track_id = 1
        test_chunk_idx = 0
        test_frame_num = 10
        test_bbox = (100, 100, 50, 100)
        test_confidence = 0.95
        test_timestamp = 0.5
        
        print(f"\n   Simulating recognition of {test_recognized_id}...")
        
        # Assign temporary ID
        unknown_id = state_manager.assign_temporary_id(
            test_recognized_id, test_track_id, test_chunk_idx,
            test_frame_num, test_bbox, test_confidence, test_timestamp
        )
        print(f"   Assigned temporary ID: {unknown_id}")
        
        # Resolve IDs
        final_mappings = state_manager.resolve_person_ids()
        final_id = final_mappings.get(unknown_id)
        print(f"   Final ID assigned: {final_id}")
        
        # Check if it reused existing ID
        if final_id == test_recognized_id:
            print(f"   [OK] Successfully reused existing person ID!")
        else:
            print(f"   [ERROR] Created new ID instead of reusing {test_recognized_id}")
            
    except Exception as e:
        print(f"   [ERROR] Flow test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_recognition_modules()