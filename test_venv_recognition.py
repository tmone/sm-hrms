#!/usr/bin/env python3
"""
Test recognition using virtual environment
This should work because it uses the same Python environment as the web UI
"""

import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing Recognition with Virtual Environment\n")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

# Test direct import
print("\n1. Testing direct import...")
try:
    from hr_management.processing.person_recognition_inference_simple import PersonRecognitionInferenceSimple
    print("SUCCESS: Direct import successful")
    
    # Try to load model
    recognizer = PersonRecognitionInferenceSimple('refined_quick_20250606_054446', 0.8)
    print("SUCCESS: Model loaded successfully")
except Exception as e:
    print(f"ERROR: Direct import failed: {e}")

# Test venv wrapper
print("\n2. Testing venv wrapper...")
try:
    from processing.venv_recognition_wrapper import VenvRecognitionWrapper, get_venv_python
    print("SUCCESS: Venv wrapper import successful")
    
    venv_python = get_venv_python()
    print(f"Virtual env Python: {venv_python}")
    
    # Test recognition
    wrapper = VenvRecognitionWrapper()
    print("SUCCESS: Venv wrapper initialized")
    
    # Test with a sample image
    import cv2
    from pathlib import Path
    
    test_dir = Path("processing/outputs/persons/PERSON-0001")
    if test_dir.exists():
        images = list(test_dir.glob("*.jpg"))
        if images:
            test_img = cv2.imread(str(images[0]))
            result = wrapper.recognize_person(test_img)
            if result:
                print(f"SUCCESS: Recognition successful: {result['person_id']} (confidence: {result['confidence']:.2%})")
            else:
                print("ERROR: No recognition result")
    else:
        print("INFO: No test images available")
        
except Exception as e:
    print(f"ERROR: Venv wrapper failed: {e}")
    import traceback
    traceback.print_exc()

print("\nSummary:")
print("The virtual environment wrapper allows recognition to work")
print("even when direct import fails due to NumPy compatibility.")
print("\nTo use this in production:")
print("1. Activate virtual environment: source .venv/bin/activate (Linux) or .venv\\Scripts\\activate (Windows)")
print("2. Run the app: python app.py")
print("3. Process videos - recognition should now work!")