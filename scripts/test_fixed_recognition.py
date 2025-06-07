#!/usr/bin/env python3
"""Test the fixed recognition model"""

import sys
import os
import json
from pathlib import Path
import numpy as np
import cv2

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_fixed_recognition():
    """Test that recognition now works with the fixed model"""
    
    print("🧪 Testing Fixed Recognition Model\n")
    
    # 1. Test loading PersonIDManager
    print("1. Testing PersonIDManager...")
    try:
        from processing.person_id_manager import get_person_id_manager
        
        person_id_manager = get_person_id_manager()
        print(f"   ✅ PersonIDManager loaded")
        print(f"   - Next person ID: PERSON-{person_id_manager.next_person_id:04d}")
        print(f"   - Loaded mappings: {len(person_id_manager.recognized_to_person_id)}")
        
        # Show mappings
        if person_id_manager.recognized_to_person_id:
            print("   - Mappings:")
            for name, pid in sorted(person_id_manager.recognized_to_person_id.items()):
                if name.startswith('PERSON-'):
                    print(f"     {name} -> {pid}")
                    
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        
    # 2. Test SharedStateManager
    print("\n2. Testing ImprovedSharedStateManagerV3...")
    try:
        from processing.shared_state_manager_improved import ImprovedSharedStateManagerV3
        
        state_manager = ImprovedSharedStateManagerV3()
        print(f"   ✅ SharedStateManager loaded")
        print(f"   - Recognized mappings: {len(state_manager.recognized_to_person_id)}")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        
    # 3. Test recognition flow
    print("\n3. Testing recognition flow...")
    
    # Find a test image
    test_person_id = "PERSON-0001"
    person_dir = Path(f"processing/outputs/persons/{test_person_id}")
    
    if person_dir.exists():
        images = list(person_dir.glob("*.jpg"))
        if images:
            test_image_path = images[0]
            print(f"   Using test image: {test_image_path}")
            
            # Load image
            img = cv2.imread(str(test_image_path))
            
            if img is not None:
                print(f"   Image shape: {img.shape}")
                
                # Test SimplePersonRecognitionInference
                try:
                    from processing.simple_person_recognition_inference import SimplePersonRecognitionInference
                    
                    recognizer = SimplePersonRecognitionInference()
                    
                    if recognizer.inference is not None:
                        print("   ✅ Recognition model loaded")
                        
                        # Test prediction
                        result = recognizer.predict_single(img)
                        
                        if result:
                            print(f"   Recognition result:")
                            print(f"     - Person ID: {result['person_id']}")
                            print(f"     - Confidence: {result['confidence']:.2f}")
                            
                            if result['person_id'] == test_person_id:
                                print(f"   ✅ Correctly recognized {test_person_id}!")
                            else:
                                print(f"   ⚠️  Recognized as {result['person_id']} instead of {test_person_id}")
                        else:
                            print("   ❌ No recognition result")
                    else:
                        print("   ❌ Recognition model failed to load")
                        
                except Exception as e:
                    print(f"   ❌ Recognition test failed: {e}")
                    import traceback
                    traceback.print_exc()
                    
    # 4. Test the complete flow
    print("\n4. Testing complete person ID assignment flow...")
    
    try:
        state_manager = ImprovedSharedStateManagerV3()
        
        # Test cases
        test_cases = [
            ("PERSON-0001", "Should reuse existing ID"),
            ("PERSON-0010", "Should reuse existing ID"),
            ("unknown", "Should create new ID"),
            (None, "Should create new ID")
        ]
        
        for recognized_id, description in test_cases:
            print(f"\n   Test: {description}")
            print(f"   Recognized ID: {recognized_id}")
            
            # Assign temporary ID
            unknown_id = state_manager.assign_temporary_id(
                recognized_id=recognized_id,
                track_id=100 + len(test_cases),
                chunk_idx=0,
                frame_num=10,
                bbox=(100, 100, 50, 100),
                confidence=0.95,
                timestamp=0.5
            )
            
            print(f"   Temporary ID: {unknown_id}")
            
        # Resolve all IDs
        print("\n   Resolving person IDs...")
        final_mappings = state_manager.resolve_person_ids()
        
        print("\n   Final mappings:")
        for unknown_id, person_id in sorted(final_mappings.items()):
            det_info = state_manager.unknown_to_detection_info.get(unknown_id)
            if det_info:
                print(f"     {unknown_id} (recognized: {det_info.recognized_id}) -> {person_id}")
                
    except Exception as e:
        print(f"   ❌ Flow test failed: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n✅ Testing complete!")

if __name__ == "__main__":
    test_fixed_recognition()