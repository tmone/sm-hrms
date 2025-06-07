#!/usr/bin/env python3
"""
Patch GPU detection to include recognition before assigning new PERSON IDs
"""

import sys
import os
from pathlib import Path

print("🔧 Patching GPU Detection with Recognition\n")

# Read the GPU detection file
gpu_file = Path('processing/gpu_enhanced_detection.py')

if not gpu_file.exists():
    print("❌ GPU detection file not found")
    sys.exit(1)

with open(gpu_file, 'r') as f:
    content = f.read()

# Check if already patched
if 'RECOGNITION PATCH' in content:
    print("✅ Already patched!")
    sys.exit(0)

# Find the imports section
import_section = """
# RECOGNITION PATCH - START
# Try to import recognition for person identification
try:
    from processing.simple_person_recognition_inference import SimplePersonRecognitionInference
    _recognizer = SimplePersonRecognitionInference()
    if _recognizer.inference is None:
        print("⚠️ Recognition model not loaded - will create new IDs for all persons")
        _recognizer = None
    else:
        print("✅ Recognition model loaded for GPU detection")
    RECOGNITION_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Recognition not available: {e}")
    _recognizer = None
    RECOGNITION_AVAILABLE = False
# RECOGNITION PATCH - END
"""

# Find where to insert (after other imports)
insert_pos = content.find('# Configure logging')
if insert_pos == -1:
    insert_pos = content.find('print("GPU Enhanced Detection")')

if insert_pos != -1:
    # Insert the import section
    content = content[:insert_pos] + import_section + "\n" + content[insert_pos:]
    print("✅ Added recognition imports")
else:
    print("❌ Could not find insertion point for imports")
    sys.exit(1)

# Now patch the assign_person_id function to try recognition first
recognition_code = '''
    # RECOGNITION PATCH - Try recognition first
    if frame is not None and _recognizer is not None:
        try:
            # Extract person region
            person_img = frame[int(y1):int(y2), int(x1):int(x2)]
            
            if person_img.size > 0 and person_img.shape[0] > 50 and person_img.shape[1] > 50:
                # Try recognition
                result = _recognizer.predict_single(person_img)
                
                if result and result.get('person_id') != 'unknown' and result.get('confidence', 0) > 0.7:
                    recognized_id = result['person_id']
                    confidence = result['confidence']
                    
                    print(f"🎯 Frame {frame_num}: Recognized {recognized_id} with confidence {confidence:.2f}")
                    
                    # Extract numeric ID from PERSON-XXXX format
                    try:
                        if recognized_id.startswith('PERSON-'):
                            person_id = int(recognized_id.replace('PERSON-', ''))
                            
                            # Update tracking info
                            person_tracks[person_id] = {
                                'last_frame': frame_num,
                                'last_center': ((x1 + x2) / 2, (y1 + y2) / 2),
                                'last_size': (x2 - x1, y2 - y1),
                                'recognized': True,
                                'confidence': confidence
                            }
                            
                            return person_id
                    except:
                        pass
        except Exception as e:
            pass  # Silent fail, continue with tracking
    # END RECOGNITION PATCH
'''

# Find the assign_person_id function
func_start = content.find('def assign_person_id(')
if func_start != -1:
    # Find where to insert (after the docstring)
    func_body_start = content.find('center_x = (x1 + x2) / 2', func_start)
    if func_body_start != -1:
        # Insert recognition code before center calculation
        content = content[:func_body_start] + recognition_code + "\n    " + content[func_body_start:]
        print("✅ Patched assign_person_id function")

# Now we need to modify detect_persons_batch to pass frames
# Find where person_id is assigned in detect_persons_batch
batch_func = content.find('def detect_persons_batch(')
if batch_func != -1:
    # This is more complex - need to modify the function signature
    print("⚠️ Note: detect_persons_batch needs manual modification to pass frames for recognition")
    print("   Currently it only passes frame numbers, not actual frames")

# Save the patched file
backup_file = gpu_file.with_suffix('.py.backup')
if not backup_file.exists():
    # Make backup
    with open(backup_file, 'w') as f:
        f.write(content)
    print(f"✅ Created backup: {backup_file}")

# Write patched content
with open(gpu_file, 'w') as f:
    f.write(content)

print("\n✅ GPU detection patched!")
print("\n⚠️ IMPORTANT: The patch is partial. To fully enable recognition:")
print("1. The detect_persons_batch function needs to pass actual frames, not just frame numbers")
print("2. The assign_person_id calls need to include the frame parameter")
print("\nFor now, recognition will not work until these changes are made manually.")
print("\n💡 Alternative: Use chunked_video_processor.py which already has recognition")