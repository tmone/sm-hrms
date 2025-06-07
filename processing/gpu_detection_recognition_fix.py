#!/usr/bin/env python3
"""
Fix for GPU detection to include recognition
Add this recognition check before creating new person folders
"""

def add_recognition_to_person_extraction():
    """
    This shows where to add recognition in gpu_enhanced_detection.py
    Add this code at line 1037, right after extracting person_img
    """
    
    # This is the code to add in extract_person_images_gpu function
    # Right after: person_img = frame[y1:y2, x1:x2]
    
    recognition_code = '''
                # TRY RECOGNITION BEFORE USING ASSIGNED ID
                if person_img.size > 0 and _recognizer is not None:
                    try:
                        # Try to recognize this person
                        result = _recognizer.predict_single(person_img)
                        
                        if result and result.get('person_id') != 'unknown' and result.get('confidence', 0) > 0.7:
                            recognized_id = result['person_id']
                            confidence = result['confidence']
                            
                            print(f"🎯 Recognized {recognized_id} with confidence {confidence:.2f}")
                            
                            # Use the recognized ID instead of the assigned one
                            person_id_str = recognized_id
                            person_dir = persons_dir / person_id_str
                            person_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Update the person_id in metadata
                            person_metadata["person_id"] = person_id_str
                            
                            # TODO: Update the person_id in the database/detections
                    except Exception as e:
                        print(f"Recognition error: {e}")
                        # Continue with assigned ID
    '''
    
    return recognition_code

# The problem is that GPU detection:
# 1. Assigns IDs based on position tracking only (in assign_person_id function)
# 2. Never attempts recognition
# 3. Creates folders with these new IDs

# To fix GPU detection, you need to:

print("🔧 GPU Detection Recognition Fix\n")

print("The issue: GPU detection assigns new IDs without trying recognition\n")

print("To fix gpu_enhanced_detection.py:")
print("\n1. Add at the top (after imports):")
print("""
# Import recognition
try:
    from processing.simple_person_recognition_inference import SimplePersonRecognitionInference
    _recognizer = SimplePersonRecognitionInference()
    if _recognizer.inference is None:
        print("⚠️ Recognition model not loaded - will create new IDs for all persons")
        _recognizer = None
    else:
        print("✅ Recognition model loaded for GPU detection")
except Exception as e:
    print(f"⚠️ Recognition not available: {e}")
    _recognizer = None
""")

print("\n2. In extract_person_images_gpu function, after line 1036 (person_img = frame[y1:y2, x1:x2]):")
print("""
                # TRY RECOGNITION FIRST
                recognized = False
                if person_img.size > 0 and _recognizer is not None:
                    try:
                        result = _recognizer.predict_single(person_img)
                        if result and result.get('person_id') != 'unknown' and result.get('confidence', 0) > 0.7:
                            # Override the person_id with recognized one
                            person_id_str = result['person_id']
                            person_dir = persons_dir / person_id_str
                            person_dir.mkdir(parents=True, exist_ok=True)
                            recognized = True
                            print(f"🎯 Recognized {person_id_str} with confidence {result['confidence']:.2f}")
                    except Exception as e:
                        pass  # Continue with assigned ID
                
                if not recognized:
                    # Use the original assigned ID
                    # (existing code continues here)
""")

print("\n3. Alternative Quick Fix - Post-process recognition:")
print("   Run recognition AFTER GPU detection to fix IDs")
print("   This is easier than modifying GPU detection\n")

print("Would you like me to create a post-processing script that:")
print("1. Reads the extracted persons from GPU detection")
print("2. Runs recognition on each person")
print("3. Merges/renames folders for recognized persons?")
print("\nThis would fix the IDs without modifying GPU detection code.")