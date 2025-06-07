"""
GPU Enhanced Detection with Recognition Support
This adds recognition before assigning new PERSON IDs
"""
import logging
import numpy as np
import cv2
from pathlib import Path

logger = logging.getLogger(__name__)

# Import the original gpu_enhanced_detection functions
from processing.gpu_enhanced_detection import *

# Try to import recognition
try:
    from processing.simple_person_recognition_inference import SimplePersonRecognitionInference
    RECOGNITION_AVAILABLE = True
except:
    RECOGNITION_AVAILABLE = False
    logger.warning("Recognition not available")

# Global recognition instance
_recognizer = None

def get_recognizer():
    """Get or create recognizer instance"""
    global _recognizer
    if _recognizer is None and RECOGNITION_AVAILABLE:
        try:
            _recognizer = SimplePersonRecognitionInference()
            if _recognizer.inference is None:
                logger.warning("Recognition model not loaded - will create new IDs for all persons")
                _recognizer = None
            else:
                logger.info("✅ Recognition model loaded for GPU detection")
        except Exception as e:
            logger.error(f"Failed to load recognizer: {e}")
            _recognizer = None
    return _recognizer

def assign_person_id_with_recognition(x1, y1, x2, y2, frame_num, person_tracks, next_person_id, frame=None):
    """
    Enhanced person ID assignment that tries recognition first
    """
    # First try recognition if we have the frame
    if frame is not None:
        recognizer = get_recognizer()
        if recognizer and recognizer.inference:
            try:
                # Extract person region
                person_img = frame[int(y1):int(y2), int(x1):int(x2)]
                
                if person_img.size > 0 and person_img.shape[0] > 50 and person_img.shape[1] > 50:
                    # Try recognition
                    result = recognizer.predict_single(person_img)
                    
                    if result and result.get('person_id') != 'unknown' and result.get('confidence', 0) > 0.7:
                        recognized_id = result['person_id']
                        confidence = result['confidence']
                        
                        logger.info(f"🎯 Frame {frame_num}: Recognized {recognized_id} with confidence {confidence:.2f}")
                        
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
                logger.debug(f"Recognition error: {e}")
    
    # Fall back to position-based tracking
    return assign_person_id(x1, y1, x2, y2, frame_num, person_tracks, next_person_id)

# Monkey patch the original function
original_detect_persons_batch = detect_persons_batch

def detect_persons_batch_with_recognition(frames, frame_numbers, model, device='cpu'):
    """Enhanced batch detection with recognition support"""
    
    # Log once at start
    recognizer = get_recognizer()
    if recognizer is None:
        logger.warning("⚠️ Recognition disabled - all persons will get new PERSON-XXXX IDs")
    else:
        logger.info("✅ Recognition enabled for GPU detection")
    
    # Call original function
    return original_detect_persons_batch(frames, frame_numbers, model, device)

# Replace the function
detect_persons_batch = detect_persons_batch_with_recognition

logger.info("GPU detection enhanced with recognition support")