"""Debug version of enhanced detection to track recognition process"""

import cv2
import numpy as np
import os
import json
import logging
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Also create a file logger
debug_log_dir = Path("processing/debug_logs")
debug_log_dir.mkdir(exist_ok=True)
debug_log_file = debug_log_dir / f"recognition_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

file_handler = logging.FileHandler(debug_log_file)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

logger.info(f"Debug log will be saved to: {debug_log_file}")


class DebugPersonTracker:
    """Debug version of PersonTracker with detailed logging"""
    
    def __init__(self):
        self.tracks = {}
        self.next_track_id = 1
        self.next_person_id = self._get_next_person_id()
        self.recognition_model = None
        
        logger.info("=== DebugPersonTracker initialized ===")
        logger.info(f"Starting person ID: PERSON-{self.next_person_id:04d}")
        
        # Try to load recognition model
        self._load_recognition_model()
        
    def _get_next_person_id(self):
        """Get next available person ID"""
        persons_dir = Path('processing/outputs/persons')
        persons_dir.mkdir(parents=True, exist_ok=True)
        
        existing_persons = list(persons_dir.glob('PERSON-*'))
        max_id = 0
        
        for person_folder in existing_persons:
            try:
                folder_name = person_folder.name
                if folder_name.startswith('PERSON-'):
                    person_id = int(folder_name.replace('PERSON-', ''))
                    max_id = max(max_id, person_id)
            except ValueError:
                continue
                
        logger.info(f"Found {len(existing_persons)} existing person folders, max ID: {max_id}")
        return max_id + 1
        
    def _load_recognition_model(self):
        """Load recognition model with detailed logging"""
        logger.info("Attempting to load recognition model...")
        
        # Check config
        config_path = Path('models/person_recognition/config.json')
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return
            
        with open(config_path) as f:
            config = json.load(f)
            
        default_model = config.get('default_model')
        logger.info(f"Default model from config: {default_model}")
        
        if not default_model:
            logger.error("No default model specified in config")
            return
            
        model_dir = Path('models/person_recognition') / default_model
        if not model_dir.exists():
            logger.error(f"Model directory not found: {model_dir}")
            return
            
        # List files in model directory
        model_files = list(model_dir.iterdir())
        logger.info(f"Files in model directory: {[f.name for f in model_files]}")
        
        # Try to load the model
        try:
            from hr_management.processing.person_recognition_inference_simple import PersonRecognitionInferenceSimple
            
            self.recognition_model = PersonRecognitionInferenceSimple(
                default_model,
                confidence_threshold=0.7
            )
            logger.info("✅ Recognition model loaded successfully")
            
            # Check what the model contains
            if hasattr(self.recognition_model, 'trainer'):
                trainer = self.recognition_model.trainer
                if hasattr(trainer, 'label_encoder') and trainer.label_encoder:
                    classes = trainer.label_encoder.classes_
                    logger.info(f"Model classes: {list(classes)}")
                else:
                    logger.warning("Model has no label encoder")
                    
        except Exception as e:
            logger.error(f"Failed to load recognition model: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
    def _recognize_person(self, person_image, frame_number):
        """Attempt recognition with detailed logging"""
        logger.debug(f"Frame {frame_number}: Attempting recognition...")
        
        if not self.recognition_model:
            logger.debug(f"Frame {frame_number}: No recognition model available")
            return None
            
        try:
            # Save image temporarily
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            temp_path = temp_file.name
            temp_file.close()
            
            cv2.imwrite(temp_path, person_image)
            logger.debug(f"Frame {frame_number}: Saved temp image to {temp_path}")
            
            # Process with recognition model
            result = self.recognition_model.process_cropped_image(temp_path)
            logger.debug(f"Frame {frame_number}: Recognition result: {result}")
            
            # Clean up
            try:
                os.unlink(temp_path)
            except:
                pass
                
            if result.get('persons') and len(result['persons']) > 0:
                person = result['persons'][0]
                logger.info(f"Frame {frame_number}: Recognition result - ID: {person['person_id']}, Confidence: {person['confidence']:.2f}")
                
                if person['confidence'] >= 0.7 and person['person_id'] != 'unknown':
                    return person['person_id'], person['confidence']
                else:
                    logger.debug(f"Frame {frame_number}: Recognition below threshold or unknown")
                    
            return None
            
        except Exception as e:
            logger.error(f"Frame {frame_number}: Recognition error: {type(e).__name__}: {str(e)}")
            return None
            
    def update_tracks(self, detections, frame_number, frame=None):
        """Update tracks with detailed logging"""
        logger.debug(f"Frame {frame_number}: Updating tracks with {len(detections)} detections")
        
        current_frame_tracks = []
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            
            # For first few detections, try recognition
            if frame is not None and i < 3:  # Only test first 3 to avoid spam
                x, y, w, h = bbox
                x1 = max(0, int(x))
                y1 = max(0, int(y))
                x2 = min(frame.shape[1], int(x + w))
                y2 = min(frame.shape[0], int(y + h))
                
                person_img = frame[y1:y2, x1:x2]
                
                if person_img.size > 0:
                    logger.debug(f"Frame {frame_number}, Detection {i}: Image size {person_img.shape}")
                    recognition_result = self._recognize_person(person_img, frame_number)
                    
                    if recognition_result:
                        recognized_id, confidence = recognition_result
                        logger.info(f"🎯 Frame {frame_number}: RECOGNIZED {recognized_id} with confidence {confidence:.2f}")
                    else:
                        logger.info(f"❓ Frame {frame_number}: NO RECOGNITION for detection {i}")
                        
            # Create new track (simplified for debugging)
            track_id = self.next_track_id
            self.next_track_id += 1
            
            person_id = f"PERSON-{self.next_person_id:04d}"
            self.next_person_id += 1
            
            logger.debug(f"Frame {frame_number}: Assigned track_id={track_id}, person_id={person_id}")
            
            self.tracks[track_id] = {
                'person_id': person_id,
                'first_frame': frame_number,
                'last_frame': frame_number
            }
            
            detection['track_id'] = track_id
            detection['person_id'] = person_id
            current_frame_tracks.append(detection)
            
        return current_frame_tracks


def debug_video_recognition(video_path, max_frames=100):
    """Debug video recognition process"""
    logger.info(f"=== Starting debug for video: {video_path} ===")
    
    # Load YOLO model
    model = YOLO('yolov8n.pt')
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video info: {total_frames} frames at {fps} FPS")
    
    tracker = DebugPersonTracker()
    frame_number = 0
    
    while cap.isOpened() and frame_number < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run YOLO detection
        results = model(frame, verbose=False)
        
        # Extract person detections
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    if int(box.cls[0]) == 0:  # Person class
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0])
                        
                        detection = {
                            'frame_number': frame_number,
                            'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                            'confidence': confidence
                        }
                        detections.append(detection)
                        
        # Update tracks
        if detections:
            tracked = tracker.update_tracks(detections, frame_number, frame)
            
        frame_number += 1
        
        if frame_number % 10 == 0:
            logger.info(f"Processed {frame_number} frames...")
            
    cap.release()
    logger.info(f"=== Debug completed. Processed {frame_number} frames ===")
    logger.info(f"Debug log saved to: {debug_log_file}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # Find a test video
        video_files = list(Path('static/uploads').glob('*.mp4'))
        if video_files:
            video_path = str(video_files[0])
        else:
            print("No video files found")
            sys.exit(1)
            
    debug_video_recognition(video_path, max_frames=50)