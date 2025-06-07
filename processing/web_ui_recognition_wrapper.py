"""
Wrapper that calls the web UI recognition endpoint
Since the web UI works, we can use it for recognition during video processing
"""
import requests
import cv2
import base64
import json
import tempfile
import os
from pathlib import Path

class WebUIRecognitionWrapper:
    """Use the working web UI for recognition"""
    
    def __init__(self, base_url="http://localhost:5000", model_name='refined_quick_20250606_054446'):
        self.base_url = base_url
        self.model_name = model_name
        self.confidence_threshold = 0.8
        
    def recognize_person(self, person_img):
        """
        Recognize person by calling the web UI endpoint
        """
        try:
            # Save image to temp file (like UI does)
            temp_dir = tempfile.mkdtemp()
            temp_file = os.path.join(temp_dir, 'person.jpg')
            cv2.imwrite(temp_file, person_img)
            
            # Read and encode image
            with open(temp_file, 'rb') as f:
                img_data = f.read()
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            
            # Call web API
            response = requests.post(
                f"{self.base_url}/api/person_recognition/predict",
                json={
                    'image': img_base64,
                    'model_name': self.model_name,
                    'confidence_threshold': self.confidence_threshold
                }
            )
            
            # Clean up
            os.unlink(temp_file)
            os.rmdir(temp_dir)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('persons'):
                    person = result['persons'][0]
                    if person['confidence'] >= self.confidence_threshold:
                        return {
                            'person_id': person['person_id'],
                            'confidence': person['confidence']
                        }
            
            return None
            
        except Exception as e:
            print(f"Web UI recognition error: {e}")
            return None

# Alternative: Direct file-based approach
def recognize_using_web_method(person_img, model_name='refined_quick_20250606_054446'):
    """
    Use the exact same method as the web UI test
    This avoids the NumPy issue by using file-based processing
    """
    try:
        # Import here to avoid early failure
        from hr_management.blueprints.person_recognition import test_model_internal
        
        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        cv2.imwrite(temp_file.name, person_img)
        temp_file.close()
        
        # Use the internal test function that works
        result = test_model_internal(
            model_name=model_name,
            test_type='image',
            file_path=temp_file.name,
            confidence_threshold=0.8,
            is_cropped=True
        )
        
        # Clean up
        os.unlink(temp_file.name)
        
        if result and result.get('persons'):
            person = result['persons'][0]
            if person['confidence'] >= 0.8:
                return {
                    'person_id': person['person_id'],
                    'confidence': person['confidence']
                }
        
        return None
        
    except Exception as e:
        print(f"Direct method error: {e}")
        return None