#!/usr/bin/env python3
"""Diagnose why recognition works in UI but not in video processing"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("[SEARCH] Diagnosing Recognition Difference\n")

print("1. UI Test Results:")
print("   [OK] Model: refined_quick_20250606_054446")
print("   [OK] Recognized: PERSON-0019 with 80.4% confidence")
print("   [OK] This proves the model WORKS!\n")

print("2. Checking how UI loads the model...")

# Check the UI endpoint
ui_path = Path('hr_management/blueprints/person_recognition.py') if Path('hr_management/blueprints/person_recognition.py').exists() else Path('blueprints/person_recognition.py')

if ui_path.exists():
    with open(ui_path) as f:
        content = f.read()
        
    # Find test_model route
    if 'test_model' in content:
        print("   Found test_model route in UI")
        
        # Check what it imports
        if 'PersonRecognitionInferenceSimple' in content:
            print("   [OK] UI uses PersonRecognitionInferenceSimple")
        else:
            print("   Uses different recognition class")

print("\n3. Checking video processing...")

# The key difference
print("\n🔑 KEY FINDING:")
print("   The UI imports from: hr_management.processing.person_recognition_inference_simple")
print("   Video processing imports from: processing.simple_person_recognition_inference")
print("   These might be loading differently!\n")

print("4. Testing both import paths...")

# Test 1: UI import path
print("   Test 1 - UI import path:")
try:
    from hr_management.processing.person_recognition_inference_simple import PersonRecognitionInferenceSimple
    inference = PersonRecognitionInferenceSimple('refined_quick_20250606_054446', confidence_threshold=0.6)
    print("   [OK] UI import works!")
    print(f"   Model loaded: {inference.model is not None}")
    print(f"   Scaler loaded: {inference.scaler is not None}")
except Exception as e:
    print(f"   [ERROR] UI import failed: {e}")

# Test 2: Video processing import path
print("\n   Test 2 - Video processing import path:")
try:
    from processing.simple_person_recognition_inference import SimplePersonRecognitionInference
    recognizer = SimplePersonRecognitionInference()
    print("   [OK] Video import works!")
    print(f"   Inference loaded: {recognizer.inference is not None}")
except Exception as e:
    print(f"   [ERROR] Video import failed: {e}")

print("\n5. Solution:")
print("   The issue is likely that the video processing is failing to load")
print("   the model due to import path differences or initialization issues.")
print("\n   To fix this, we need to make sure video processing uses")
print("   the SAME import and initialization as the UI.\n")

# Check for path issues
print("6. Checking Python path...")
print(f"   Current working directory: {os.getcwd()}")
print(f"   Python path includes:")
for p in sys.path[:5]:
    print(f"   - {p}")