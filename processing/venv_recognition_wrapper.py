#!/usr/bin/env python3
"""
Recognition wrapper that ensures we use the virtual environment
This solves the NumPy compatibility issue by using the correct Python environment
"""
import subprocess
import sys
import os
import json
import tempfile
import cv2
from pathlib import Path

def get_venv_python():
    """Get the Python executable from virtual environment"""
    venv_path = Path(__file__).parent.parent / '.venv'
    
    if sys.platform == 'win32':
        python_exe = venv_path / 'Scripts' / 'python.exe'
    else:
        python_exe = venv_path / 'bin' / 'python'
    
    if python_exe.exists():
        return str(python_exe)
    else:
        # Fallback to current Python
        return sys.executable

class VenvRecognitionWrapper:
    """Run recognition in the virtual environment where it works"""
    
    def __init__(self, model_name=None):
        # If no model specified, get default from config
        if model_name is None:
            config_path = Path("models/person_recognition/config.json")
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                model_name = config.get('default_model', 'person_model_svm_20250607_181818')
                print(f"📋 Using default model from config: {model_name}")
            else:
                model_name = 'person_model_svm_20250607_181818'
                print(f"📋 Using fallback model: {model_name}")
        
        self.model_name = model_name
        self.python_exe = get_venv_python()
        print(f"🐍 Using Python: {self.python_exe}")
        
    def recognize_person(self, person_img, confidence_threshold=0.8):
        """
        Recognize person by running inference in the virtual environment
        """
        try:
            # Save image to temp file
            temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            cv2.imwrite(temp_file.name, person_img)
            temp_file.close()
            
            # Create a simple recognition script
            script = f'''
import sys
sys.path.insert(0, r"{Path(__file__).parent.parent}")
from hr_management.processing.person_recognition_inference_simple import PersonRecognitionInferenceSimple

recognizer = PersonRecognitionInferenceSimple("{self.model_name}", {confidence_threshold})
result = recognizer.process_cropped_image(r"{temp_file.name}")

if result and result.get("persons"):
    person = result["persons"][0]
    print(person["person_id"], person["confidence"])
else:
    print("unknown", "0.0")
'''
            
            # Run in virtual environment
            result = subprocess.run(
                [self.python_exe, '-c', script],
                capture_output=True,
                text=True
            )
            
            # Clean up
            os.unlink(temp_file.name)
            
            if result.returncode == 0:
                output = result.stdout.strip()
                if output:
                    parts = output.split()
                    if len(parts) >= 2 and parts[0] != 'unknown':
                        person_id = parts[0]
                        confidence = float(parts[1])
                        if confidence >= confidence_threshold:
                            return {
                                'person_id': person_id,
                                'confidence': confidence
                            }
            else:
                print(f"Recognition error: {result.stderr}")
            
            return None
            
        except Exception as e:
            print(f"Venv recognition error: {e}")
            return None

# Global instance
_venv_recognizer = None

def get_venv_recognizer():
    """Get or create the virtual environment recognizer"""
    global _venv_recognizer
    if _venv_recognizer is None:
        _venv_recognizer = VenvRecognitionWrapper()
    return _venv_recognizer

def recognize_in_venv(person_img, confidence_threshold=0.8):
    """
    Recognize person using the virtual environment
    This avoids NumPy compatibility issues
    """
    recognizer = get_venv_recognizer()
    return recognizer.recognize_person(person_img, confidence_threshold)