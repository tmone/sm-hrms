"""
Use the EXACT same recognition approach as the working UI test
"""
import cv2
import os
import tempfile
import shutil
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class UIStyleRecognition:
    """Recognition that works exactly like the UI test"""
    
    def __init__(self, model_name='refined_quick_20250606_054446', confidence_threshold=0.6):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.inference = None
        self.loaded = False
        
        # Load exactly like UI does
        try:
            from hr_management.processing.person_recognition_inference_simple import PersonRecognitionInferenceSimple
            self.inference = PersonRecognitionInferenceSimple(model_name, confidence_threshold)
            self.loaded = True
            logger.info(f"[OK] Loaded recognition model UI-style: {model_name}")
        except Exception as e:
            logger.error(f"[ERROR] Failed to load model UI-style: {e}")
    
    def recognize_person_like_ui(self, person_img):
        """
        Recognize person using the EXACT same method as UI
        The UI saves image to file and uses process_cropped_image
        """
        if not self.loaded or self.inference is None:
            return None
            
        try:
            # Create temp file EXACTLY like UI does
            temp_dir = tempfile.mkdtemp()
            temp_file = os.path.join(temp_dir, 'test_image.jpg')
            
            # Save image to file
            cv2.imwrite(temp_file, person_img)
            
            # Process EXACTLY like UI - using process_cropped_image
            results = self.inference.process_cropped_image(temp_file)
            
            # Clean up
            shutil.rmtree(temp_dir)
            
            # Parse results like UI expects
            if results and results.get('persons'):
                person = results['persons'][0]
                if person['confidence'] >= self.confidence_threshold:
                    return {
                        'person_id': person['person_id'],
                        'confidence': person['confidence']
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Recognition error: {e}")
            return None

# Global instance
_ui_recognizer = None

def get_ui_recognizer():
    """Get recognizer that works like UI"""
    global _ui_recognizer
    if _ui_recognizer is None:
        _ui_recognizer = UIStyleRecognition()
    return _ui_recognizer

def recognize_person_ui_style(person_img, confidence_threshold=0.6):
    """
    Recognize person using the exact same method as the working UI test
    """
    recognizer = get_ui_recognizer()
    recognizer.confidence_threshold = confidence_threshold
    return recognizer.recognize_person_like_ui(person_img)

# Test function
if __name__ == "__main__":
    print("Testing UI-style recognition...")
    
    # Test with a person image
    test_person = "PERSON-0001"
    person_dir = Path(f"processing/outputs/persons/{test_person}")
    
    if person_dir.exists():
        images = list(person_dir.glob("*.jpg"))
        if images:
            test_img = cv2.imread(str(images[0]))
            
            result = recognize_person_ui_style(test_img)
            if result:
                print(f"[OK] Recognized: {result['person_id']} (confidence: {result['confidence']:.2f})")
            else:
                print("[ERROR] Not recognized")
    
    print("\nThis uses the EXACT same method as the UI test that works!")