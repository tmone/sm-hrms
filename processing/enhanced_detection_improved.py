"""
Enhanced Detection with Improved Recognition
This module improves person detection by always checking recognition first
to reduce duplicate person codes for the same real person.
"""
import cv2
import numpy as np
import os
import json
import uuid
import tempfile
from datetime import datetime
from ultralytics import YOLO
import logging
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
from collections import defaultdict

# Import OCR extractor
try:
    from hr_management.processing.ocr_extractor import VideoOCRExtractor, OCR_AVAILABLE
except ImportError:
    OCR_AVAILABLE = False
    VideoOCRExtractor = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedPersonTracker:
    """Improved tracker that prioritizes recognition over new ID creation"""
    
    def __init__(self, max_distance=50, min_confidence=0.5, use_recognition=True, 
                 recognition_threshold=0.85, recognition_check_interval=5):
        self.tracks = {}  # track_id -> track_data
        self.next_track_id = 1
        self.max_distance = max_distance
        self.min_confidence = min_confidence
        self.next_person_id = self._get_next_person_id()
        self.use_recognition = use_recognition
        self.recognition_threshold = recognition_threshold
        self.recognition_check_interval = recognition_check_interval  # Check recognition every N frames
        self.recognition_model = None
        
        # Recognition cache to avoid redundant checks
        self.recognition_cache = {}  # track_id -> (person_id, confidence, last_check_frame)
        
        # Person ID to tracks mapping for merging
        self.person_to_tracks = defaultdict(list)  # person_id -> [track_ids]
        
        # Load default recognition model
        if self.use_recognition:
            self._load_default_model()
    
    def _load_default_model(self):
        """Load the default recognition model if available"""
        try:
            config_path = Path('models/person_recognition/config.json')
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                    default_model = config.get('default_model')
                    
                    if default_model:
                        model_dir = Path('models/person_recognition') / default_model
                        if model_dir.exists():
                            logger.info(f"Loading default recognition model: {default_model}")
                            try:
                                from hr_management.processing.person_recognition_inference_simple import PersonRecognitionInferenceSimple
                                self.recognition_model = PersonRecognitionInferenceSimple(
                                    default_model, 
                                    confidence_threshold=self.recognition_threshold
                                )
                                logger.info(f"[OK] Default model loaded successfully: {default_model}")
                            except Exception as e:
                                logger.error(f"Failed to load recognition model: {e}")
                                self.recognition_model = None
                        else:
                            logger.warning(f"Default model directory not found: {model_dir}")
                    else:
                        logger.info("No default model configured")
            else:
                logger.info("No recognition model config found")
        except Exception as e:
            logger.error(f"Error loading default model config: {e}")
            self.recognition_model = None
    
    def _get_next_person_id(self):
        """Get the next available person ID by checking existing person folders"""
        persons_dir = Path('processing/outputs/persons')
        persons_dir.mkdir(parents=True, exist_ok=True)
        
        counter_file = persons_dir / 'person_id_counter.json'
        
        # Check existing folders
        existing_persons = list(persons_dir.glob('PERSON-*'))
        max_folder_id = 0
        
        for person_folder in existing_persons:
            try:
                folder_name = person_folder.name
                if folder_name.startswith('PERSON-'):
                    person_id = int(folder_name.replace('PERSON-', ''))
                    max_folder_id = max(max_folder_id, person_id)
            except ValueError:
                continue
        
        # Check counter file
        max_counter_id = 0
        if counter_file.exists():
            try:
                with open(counter_file, 'r') as f:
                    data = json.load(f)
                    max_counter_id = data.get('last_person_id', 0)
            except Exception as e:
                logger.warning(f"Error reading counter file: {e}")
        
        max_id = max(max_folder_id, max_counter_id)
        next_id = max_id + 1
        
        logger.info(f"Found {len(existing_persons)} existing persons, starting from: PERSON-{next_id:04d}")
        return next_id
    
    def euclidean_distance(self, box1, box2):
        """Calculate distance between two bounding box centers"""
        center1 = (box1[0] + box1[2]/2, box1[1] + box1[3]/2)
        center2 = (box2[0] + box2[2]/2, box2[1] + box2[3]/2)
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def _recognize_person(self, person_image, frame_number):
        """Try to recognize a person using the default model"""
        if not self.recognition_model:
            return None
            
        try:
            # Save image temporarily
            temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            temp_path = temp_file.name
            temp_file.close()
            
            cv2.imwrite(temp_path, person_image)
            
            # Process with recognition model
            result = self.recognition_model.process_cropped_image(temp_path)
            
            # Clean up
            try:
                os.unlink(temp_path)
            except:
                pass
            
            if result.get('persons') and len(result['persons']) > 0:
                person = result['persons'][0]
                if person['confidence'] >= self.recognition_threshold and person['person_id'] != 'unknown':
                    logger.info(f"[TARGET] Frame {frame_number}: Recognized {person['person_id']} with confidence {person['confidence']:.2f}")
                    return person['person_id'], person['confidence']
            
            return None
        except Exception as e:
            logger.error(f"Error during person recognition: {e}")
            return None
    
    def _should_check_recognition(self, track_id, frame_number):
        """Check if we should run recognition for this track"""
        if track_id in self.recognition_cache:
            _, _, last_check = self.recognition_cache[track_id]
            # Check every N frames
            return (frame_number - last_check) >= self.recognition_check_interval
        return True  # Always check for new tracks
    
    def _merge_tracks_for_person(self, person_id, primary_track_id):
        """Merge all tracks that belong to the same recognized person"""
        if person_id not in self.person_to_tracks:
            return
            
        track_ids = self.person_to_tracks[person_id]
        if len(track_ids) <= 1:
            return
            
        # Keep the primary track and merge others into it
        primary_track = self.tracks[primary_track_id]
        
        for track_id in track_ids:
            if track_id != primary_track_id and track_id in self.tracks:
                track = self.tracks[track_id]
                
                # Merge detections
                primary_track['detections'].extend(track['detections'])
                
                # Update time range
                primary_track['first_frame'] = min(primary_track['first_frame'], track['first_frame'])
                primary_track['last_frame'] = max(primary_track['last_frame'], track['last_frame'])
                
                # Remove merged track
                del self.tracks[track_id]
                logger.info(f"Merged track {track_id} into {primary_track_id} for person {person_id}")
        
        # Update person_to_tracks
        self.person_to_tracks[person_id] = [primary_track_id]
    
    def update_tracks(self, detections, frame_number, frame=None):
        """Update person tracks with new detections, prioritizing recognition"""
        current_frame_tracks = []
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            if confidence < self.min_confidence:
                continue
            
            # Try to recognize person FIRST (before track assignment)
            recognized_person_id = None
            recognition_confidence = 0.0
            
            if frame is not None and self.recognition_model:
                x, y, w, h = bbox
                x1 = max(0, int(x))
                y1 = max(0, int(y))
                x2 = min(frame.shape[1], int(x + w))
                y2 = min(frame.shape[0], int(y + h))
                
                person_img = frame[y1:y2, x1:x2]
                
                if person_img.size > 0 and person_img.shape[0] > 50 and person_img.shape[1] > 50:
                    recognition_result = self._recognize_person(person_img, frame_number)
                    if recognition_result:
                        recognized_person_id, recognition_confidence = recognition_result
            
            # Find best matching track
            best_track_id = None
            min_distance = float('inf')
            
            for track_id, track_data in self.tracks.items():
                if track_data['last_frame'] >= frame_number - 10:  # Active in last 10 frames
                    distance = self.euclidean_distance(bbox, track_data['last_bbox'])
                    if distance < min_distance and distance < self.max_distance:
                        min_distance = distance
                        best_track_id = track_id
            
            # If we recognized a person, check if we already have tracks for them
            if recognized_person_id:
                existing_tracks = self.person_to_tracks.get(recognized_person_id, [])
                
                # If we have existing tracks for this person
                if existing_tracks:
                    # Find the closest existing track for this person
                    best_person_track = None
                    min_person_distance = float('inf')
                    
                    for track_id in existing_tracks:
                        if track_id in self.tracks:
                            track_data = self.tracks[track_id]
                            distance = self.euclidean_distance(bbox, track_data['last_bbox'])
                            if distance < min_person_distance:
                                min_person_distance = distance
                                best_person_track = track_id
                    
                    # Use the person's track if it's reasonably close
                    if best_person_track and min_person_distance < self.max_distance * 2:
                        best_track_id = best_person_track
                        logger.info(f"Using existing track {best_track_id} for recognized person {recognized_person_id}")
            
            if best_track_id is not None:
                # Update existing track
                track = self.tracks[best_track_id]
                track['last_bbox'] = bbox
                track['last_frame'] = frame_number
                track['detections'].append(detection)
                
                # Update recognition if we have a better result
                if recognized_person_id and recognized_person_id != track.get('person_id'):
                    old_person_id = track.get('person_id')
                    track['person_id'] = recognized_person_id
                    track['is_recognized'] = True
                    track['recognition_confidence'] = recognition_confidence
                    
                    # Update person_to_tracks mapping
                    if old_person_id and old_person_id in self.person_to_tracks:
                        self.person_to_tracks[old_person_id].remove(best_track_id)
                    self.person_to_tracks[recognized_person_id].append(best_track_id)
                    
                    logger.info(f"Updated track {best_track_id} from {old_person_id} to {recognized_person_id}")
                    
                    # Merge tracks if needed
                    self._merge_tracks_for_person(recognized_person_id, best_track_id)
                
                track_id = best_track_id
            else:
                # Create new track
                track_id = self.next_track_id
                self.next_track_id += 1
                
                # Determine person ID
                if recognized_person_id:
                    person_id_str = recognized_person_id
                    logger.info(f"[OK] Creating track with recognized person ID: {person_id_str} (confidence: {recognition_confidence:.2f})")
                else:
                    # Only create new person ID if not recognized
                    person_id = self.next_person_id
                    self.next_person_id += 1
                    person_id_str = f"PERSON-{person_id:04d}"
                    logger.info(f"[NEW] Creating new person ID: {person_id_str}")
                
                self.tracks[track_id] = {
                    'first_frame': frame_number,
                    'last_frame': frame_number,
                    'last_bbox': bbox,
                    'detections': [detection],
                    'person_id': person_id_str,
                    'is_recognized': recognized_person_id is not None,
                    'recognition_confidence': recognition_confidence
                }
                
                # Update person_to_tracks
                self.person_to_tracks[person_id_str].append(track_id)
            
            detection['track_id'] = track_id
            detection['person_id'] = self.tracks[track_id]['person_id']
            detection['is_recognized'] = self.tracks[track_id].get('is_recognized', False)
            current_frame_tracks.append(detection)
        
        return current_frame_tracks
    
    def save_person_id_counter(self):
        """Save the updated person ID counter"""
        persons_dir = Path('processing/outputs/persons')
        counter_file = persons_dir / 'person_id_counter.json'
        
        try:
            with open(counter_file, 'w') as f:
                json.dump({
                    'last_person_id': self.next_person_id - 1,
                    'updated_at': datetime.now().isoformat()
                }, f, indent=2)
            logger.info(f"Updated person ID counter to: {self.next_person_id - 1}")
        except Exception as e:
            logger.error(f"Error updating person ID counter: {e}")


def detect_and_track_persons_improved(video_path):
    """Improved detection that prioritizes recognition to reduce duplicate person codes"""
    logger.info(f"Starting improved person detection and tracking for: {video_path}")
    
    # Load YOLO model
    try:
        model = YOLO('yolov8n.pt')
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        return {}, None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return {}, None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Use improved tracker
    tracker = ImprovedPersonTracker()
    all_tracks = {}
    frame_number = 0
    
    logger.info(f"Processing {total_frames} frames at {fps} FPS with recognition enabled")
    
    while cap.isOpened():
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
                            'timestamp': frame_number / fps,
                            'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                            'confidence': confidence,
                            'class': 'person'
                        }
                        detections.append(detection)
        
        # Update tracks with recognition
        tracked_detections = tracker.update_tracks(detections, frame_number, frame)
        
        # Store in all_tracks by frame
        all_tracks[frame_number] = tracked_detections
        
        frame_number += 1
        
        # Progress logging
        if frame_number % 100 == 0:
            recognized_count = sum(1 for track in tracker.tracks.values() if track.get('is_recognized'))
            logger.info(f"Processed {frame_number}/{total_frames} frames - {recognized_count} recognized persons")
    
    cap.release()
    
    # Save the updated person ID counter
    tracker.save_person_id_counter()
    
    # Organize tracks by person_id
    person_tracks = {}
    for frame_data in all_tracks.values():
        for detection in frame_data:
            person_id = detection['person_id']
            if person_id not in person_tracks:
                person_tracks[person_id] = []
            person_tracks[person_id].append(detection)
    
    # Log recognition statistics
    recognized_persons = [pid for pid in person_tracks if any(d.get('is_recognized') for d in person_tracks[pid])]
    logger.info(f"Detection complete: {len(person_tracks)} unique persons ({len(recognized_persons)} recognized from model)")
    
    return person_tracks, tracker