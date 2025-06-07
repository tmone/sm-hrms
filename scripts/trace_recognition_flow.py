#!/usr/bin/env python3
"""Trace the exact recognition flow to see where it fails"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from pathlib import Path

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

print("🔍 Tracing Recognition Flow\n")

# 1. Check what's actually being used in chunked processor
print("1. Checking chunked_video_processor.py imports...")
chunked_path = Path('processing/chunked_video_processor.py')
if chunked_path.exists():
    with open(chunked_path) as f:
        content = f.read()
    
    # Check recognition imports
    if 'SimplePersonRecognitionInference' in content:
        print("   ✓ Uses SimplePersonRecognitionInference")
    else:
        print("   ✗ Does NOT use SimplePersonRecognitionInference")
        
    if 'use_recognition=True' in content:
        print("   ✓ Recognition is enabled")
    else:
        print("   ✗ Recognition might be disabled")

# 2. Check SimplePersonRecognitionInference
print("\n2. Checking SimplePersonRecognitionInference...")
simple_rec_path = Path('processing/simple_person_recognition_inference.py')
if simple_rec_path.exists():
    print("   ✓ File exists")
    
    # Try to import it
    try:
        from processing.simple_person_recognition_inference import SimplePersonRecognitionInference
        print("   ✓ Can import")
        
        # Try to create instance
        recognizer = SimplePersonRecognitionInference()
        print(f"   Inference loaded: {recognizer.inference is not None}")
        
    except Exception as e:
        print(f"   ✗ Cannot use: {e}")
else:
    print("   ✗ File not found")

# 3. Check where recognition happens in chunk processor
print("\n3. Finding recognition calls in chunked_video_processor.py...")
if chunked_path.exists():
    with open(chunked_path) as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if 'recogniz' in line.lower() and 'recognize' in line.lower():
            print(f"   Line {i+1}: {line.strip()}")

# 4. Check model directory
print("\n4. Checking model files...")
model_dir = Path('models/person_recognition/refined_quick_20250606_054446')
if model_dir.exists():
    files = list(model_dir.iterdir())
    print(f"   Model files: {[f.name for f in files]}")
    
    # Check file sizes
    model_pkl = model_dir / 'model.pkl'
    if model_pkl.exists():
        size_mb = model_pkl.stat().st_size / (1024 * 1024)
        print(f"   model.pkl size: {size_mb:.1f} MB")

# 5. Try direct recognition test
print("\n5. Direct recognition test...")
try:
    # Try the hr_management import path
    from hr_management.processing.person_recognition_inference_simple import PersonRecognitionInferenceSimple
    
    inference = PersonRecognitionInferenceSimple('refined_quick_20250606_054446')
    print("   ✓ PersonRecognitionInferenceSimple loaded!")
    
except Exception as e:
    print(f"   ✗ Cannot load PersonRecognitionInferenceSimple: {e}")
    
    # Try alternative
    try:
        import pickle
        model_path = Path('models/person_recognition/refined_quick_20250606_054446/model.pkl')
        with open(model_path, 'rb') as f:
            # Just try to read first few bytes
            header = f.read(10)
            print(f"   Model file header: {header}")
            
    except Exception as e2:
        print(f"   ✗ Cannot read model file: {e2}")

print("\n💡 Summary:")
print("If recognition is not working, it's likely because:")
print("1. The model file is incompatible with current Python/library versions")
print("2. The recognition is not being called during processing")
print("3. The imports are failing silently")