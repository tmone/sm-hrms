import cv2
import numpy as np
import os
import json
import uuid
import tempfile
from datetime import datetime
from ultralytics import YOLO
import logging
from typing import List, Dict, Tuple, Any
from pathlib import Path

# Import OCR extractor
try:
    from hr_management.processing.ocr_extractor import VideoOCRExtractor, OCR_AVAILABLE
except ImportError:
    OCR_AVAILABLE = False
    VideoOCRExtractor = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonTracker:
    """Track persons across video frames to solve duplicate detection problem"""
    
    def __init__(self, max_distance=50, min_confidence=0.5, use_recognition=True, recognition_threshold=0.85):
        self.tracks = {}  # track_id -> track_data
        self.next_track_id = 1
        self.max_distance = max_distance
        self.min_confidence = min_confidence
        # Track global person IDs to ensure uniqueness across videos
        self.next_person_id = self._get_next_person_id()
        self.use_recognition = use_recognition
        self.recognition_threshold = recognition_threshold
        self.recognition_model = None
        self.recognized_persons = {}  # Cache for recognized persons
        
        # Try to load default recognition model
        if self.use_recognition:
            self._load_default_model()
    
    def _load_default_model(self):
        """Load the default recognition model if available"""
        try:
            # Get default model from config
            config_path = Path('models/person_recognition/config.json')
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                    default_model = config.get('default_model')
                    
                    if default_model:
                        # Check if model exists
                        model_dir = Path('models/person_recognition') / default_model
                        if model_dir.exists():
                            logger.info(f"Loading default recognition model: {default_model}")
                            # Import inference class
                            try:
                                from hr_management.processing.person_recognition_inference_simple import PersonRecognitionInferenceSimple
                                self.recognition_model = PersonRecognitionInferenceSimple(
                                    default_model, 
                                    confidence_threshold=self.recognition_threshold
                                )
                                logger.info(f"✅ Default model loaded successfully: {default_model}")
                            except Exception as e:
                                logger.error(f"Failed to load recognition model: {e}")
                                self.recognition_model = None
        except Exception as e:
            logger.error(f"Error loading default model config: {e}")
            self.recognition_model = None
    
    def _get_next_person_id(self):
        """Get the next available person ID by checking existing person folders"""
        persons_dir = Path('processing/outputs/persons')
        persons_dir.mkdir(parents=True, exist_ok=True)
        
        # Path to the ID counter file
        counter_file = persons_dir / 'person_id_counter.json'
        
        # First, check existing folders to find the maximum ID
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
        
        # Check the counter file
        max_counter_id = 0
        if counter_file.exists():
            try:
                with open(counter_file, 'r') as f:
                    data = json.load(f)
                    max_counter_id = data.get('last_person_id', 0)
            except Exception as e:
                logger.warning(f"Error reading counter file: {e}")
        
        # Use the maximum of both
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
            import tempfile
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
                    logger.info(f"🎯 Frame {frame_number}: Recognized {person['person_id']} with confidence {person['confidence']:.2f}")
                    return person['person_id'], person['confidence']
            
            return None
        except Exception as e:
            logger.error(f"Error during person recognition: {e}")
            return None
    
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
    
    def update_tracks(self, detections, frame_number, frame=None):
        """Update person tracks with new detections"""
        current_frame_tracks = []
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            if confidence < self.min_confidence:
                continue
                
            # Find closest existing track
            best_track_id = None
            min_distance = float('inf')
            
            for track_id, track_data in self.tracks.items():
                if track_data['last_frame'] >= frame_number - 10:  # Track active in last 10 frames
                    distance = self.euclidean_distance(bbox, track_data['last_bbox'])
                    if distance < min_distance and distance < self.max_distance:
                        min_distance = distance
                        best_track_id = track_id
            
            if best_track_id is not None:
                # Update existing track
                self.tracks[best_track_id].update({
                    'last_bbox': bbox,
                    'last_frame': frame_number,
                    'detections': self.tracks[best_track_id]['detections'] + [detection]
                })
                track_id = best_track_id
            else:
                # Create new track
                track_id = self.next_track_id
                self.next_track_id += 1
                
                # Try to recognize person if frame is available
                recognized_person_id = None
                recognition_confidence = 0.0
                
                if frame is not None and self.recognition_model:
                    # Extract person region
                    x, y, w, h = bbox
                    x1 = max(0, int(x))
                    y1 = max(0, int(y))
                    x2 = min(frame.shape[1], int(x + w))
                    y2 = min(frame.shape[0], int(y + h))
                    
                    person_img = frame[y1:y2, x1:x2]
                    
                    if person_img.size > 0:
                        recognition_result = self._recognize_person(person_img, frame_number)
                        if recognition_result:
                            recognized_person_id, recognition_confidence = recognition_result
                
                # Use recognized person ID or create new one
                if recognized_person_id:
                    person_id_str = recognized_person_id
                    logger.info(f"✅ Using recognized person ID: {person_id_str} (confidence: {recognition_confidence:.2f})")
                else:
                    # Use global person ID instead of track ID
                    person_id = self.next_person_id
                    self.next_person_id += 1
                    person_id_str = f"PERSON-{person_id:04d}"
                    logger.info(f"🆕 Creating new person ID: {person_id_str}")
                
                self.tracks[track_id] = {
                    'first_frame': frame_number,
                    'last_frame': frame_number,
                    'last_bbox': bbox,
                    'detections': [detection],
                    'person_id': person_id_str,
                    'is_recognized': recognized_person_id is not None,
                    'recognition_confidence': recognition_confidence
                }
            
            detection['track_id'] = track_id
            detection['person_id'] = self.tracks[track_id]['person_id']
            current_frame_tracks.append(detection)
            
        return current_frame_tracks

def detect_and_track_persons(video_path):
    """Detect and track persons across all video frames"""
    logger.info(f"Starting person detection and tracking for: {video_path}")
    
    # Load YOLO model
    try:
        model = YOLO('yolov8n.pt')  # Use nano model for speed
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        return {}, None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return {}, None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    tracker = PersonTracker()
    all_tracks = {}
    frame_number = 0
    
    logger.info(f"Processing {total_frames} frames at {fps} FPS")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run YOLO detection
        results = model(frame, verbose=False)
        
        # Extract person detections (class 0 is 'person' in COCO)
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
                            'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],  # [x, y, width, height]
                            'confidence': confidence,
                            'class': 'person'
                        }
                        detections.append(detection)
        
        # Update tracks with current frame detections
        tracked_detections = tracker.update_tracks(detections, frame_number, frame)
        
        # Store in all_tracks by frame
        all_tracks[frame_number] = tracked_detections
        
        frame_number += 1
        
        # Progress logging
        if frame_number % 100 == 0:
            logger.info(f"Processed {frame_number}/{total_frames} frames")
    
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
    
    logger.info(f"Detected {len(person_tracks)} unique persons across {frame_number} frames")
    
    # Include tracker info for recognized persons
    for person_id in person_tracks:
        for track_id, track_data in tracker.tracks.items():
            if track_data['person_id'] == person_id:
                if track_data.get('is_recognized', False):
                    # Add recognition info to first detection
                    if person_tracks[person_id]:
                        person_tracks[person_id][0]['is_recognized'] = True
                        person_tracks[person_id][0]['recognition_confidence'] = track_data.get('recognition_confidence', 0.0)
                break
    
    return person_tracks, tracker

def create_annotated_video(video_path, person_tracks, output_dir, tracker=None):
    """Create video with bounding boxes drawn directly on frames"""
    logger.info("Creating annotated video with bounding boxes")
    
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video for annotation: {video_path}")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup output video writer with timestamp
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_path = os.path.join(output_dir, f"{video_name}_annotated_{timestamp}.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Create frame-to-detections mapping for fast lookup
    frame_detections = {}
    for person_id, detections in person_tracks.items():
        for detection in detections:
            frame_num = detection['frame_number']
            if frame_num not in frame_detections:
                frame_detections[frame_num] = []
            frame_detections[frame_num].append(detection)
    
    frame_number = 0
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw bounding boxes for current frame
        if frame_number in frame_detections:
            for detection in frame_detections[frame_number]:
                x, y, w, h = detection['bbox']
                track_id = detection.get('track_id', 0)
                person_id = detection.get('person_id', 'UNKNOWN')
                confidence = detection['confidence']
                
                # Choose color based on track_id
                color = colors[track_id % len(colors)]
                
                # Draw bounding box
                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
                
                # Check if this person was recognized
                is_recognized = False
                recognition_confidence = 0.0
                if tracker:
                    for track_id, track_data in tracker.tracks.items():
                        if track_data['person_id'] == person_id:
                            is_recognized = track_data.get('is_recognized', False)
                            recognition_confidence = track_data.get('recognition_confidence', 0.0)
                            break
                
                # Draw label with recognition indicator
                if is_recognized:
                    label = f"{person_id} [R] ({confidence:.2f})"
                    # Use green for recognized persons
                    color = (0, 255, 0)
                else:
                    label = f"{person_id} ({confidence:.2f})"
                
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(frame, (int(x), int(y) - label_size[1] - 10), 
                             (int(x) + label_size[0], int(y)), color, -1)
                cv2.putText(frame, label, (int(x), int(y) - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        out.write(frame)
        frame_number += 1
        
        if frame_number % 100 == 0:
            logger.info(f"Annotated {frame_number} frames")
    
    cap.release()
    out.release()
    
    logger.info(f"Annotated video saved: {output_video_path}")
    return output_video_path

def extract_persons_data(video_path, person_tracks, persons_dir):
    """Extract person images and metadata to PERSON-XXXX folders"""
    logger.info("Extracting person data and creating PERSON-XXXX folders")
    
    os.makedirs(persons_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video for person extraction: {video_path}")
        return
    
    for person_id, detections in person_tracks.items():
        person_dir = os.path.join(persons_dir, person_id)
        os.makedirs(person_dir, exist_ok=True)
        
        # Extract all detection frames with intelligent sampling
        # Sample every N frames to avoid storing redundant consecutive frames
        # This reduces storage while maintaining diversity
        FRAME_SAMPLE_INTERVAL = 5  # Extract every 5th frame (approx 6 images per second at 30fps)
        
        # If person appears briefly, extract all frames
        if len(detections) <= 30:  # Less than 1 second of appearance
            sample_detections = detections
        else:
            # Sample frames at regular intervals
            sample_detections = detections[::FRAME_SAMPLE_INTERVAL]
            # Always include first and last detection
            if detections[0] not in sample_detections:
                sample_detections.insert(0, detections[0])
            if detections[-1] not in sample_detections:
                sample_detections.append(detections[-1])
        
        person_metadata = {
            'person_id': person_id,
            'total_detections': len(detections),
            'first_appearance': detections[0]['timestamp'],
            'last_appearance': detections[-1]['timestamp'],
            'avg_confidence': sum(d['confidence'] for d in detections) / len(detections),
            'images': [],
            'created_at': datetime.now().isoformat()
        }
        
        for i, detection in enumerate(sample_detections):
            frame_number = detection['frame_number']
            
            # Seek to specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret:
                x, y, w, h = detection['bbox']
                
                # QUALITY FILTER: Skip persons with bounding box width < 128 pixels
                # Small bounding boxes typically contain low-quality person images
                # that are not suitable for face recognition training
                MIN_BBOX_WIDTH = 128
                
                if w < MIN_BBOX_WIDTH:
                    logger.info(f"⚠️ Skipping {person_id} frame {frame_number}: bbox width {w:.0f}px < {MIN_BBOX_WIDTH}px (too small for quality face recognition)")
                    continue
                
                # Extract person region with some padding
                padding = 10
                x1 = max(0, int(x - padding))
                y1 = max(0, int(y - padding))
                x2 = min(frame.shape[1], int(x + w + padding))
                y2 = min(frame.shape[0], int(y + h + padding))
                
                person_img = frame[y1:y2, x1:x2]
                
                if person_img.size > 0:
                    # Use simple UUID for filename
                    img_filename = f"{uuid.uuid4()}.jpg"
                    img_path = os.path.join(person_dir, img_filename)
                    cv2.imwrite(img_path, person_img)
                    
                    person_metadata['images'].append({
                        'filename': img_filename,
                        'frame_number': frame_number,
                        'timestamp': detection['timestamp'],
                        'confidence': detection['confidence'],
                        'bbox': detection['bbox']
                    })
        
        # Save metadata
        metadata_path = os.path.join(person_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(person_metadata, f, indent=2)
        
        logger.info(f"Created {person_id} folder with {len(person_metadata['images'])} images (from {len(detections)} detections)")
    
    cap.release()

def process_video_with_enhanced_detection(video_path, output_base_dir="static/uploads", extract_ocr=True):
    """Main function to process video with enhanced person detection and tracking"""
    logger.info(f"Starting enhanced video processing: {video_path}")
    
    # Get file size for logging
    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    logger.info(f"Video file size: {file_size_mb:.1f} MB")
    
    # Setup output directories
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = output_base_dir  # Use base dir directly, not a subdirectory
    # Keep persons in a separate directory for organization
    persons_dir = "processing/outputs/persons"
    
    # Initialize OCR data
    ocr_data = None
    
    try:
        # Step 0: Extract OCR data (timestamp and location)
        if extract_ocr and OCR_AVAILABLE:
            logger.info("Step 0: Extracting OCR data from video...")
            try:
                ocr_extractor = VideoOCRExtractor(ocr_engine='easyocr')
                ocr_data = ocr_extractor.extract_video_info(video_path, sample_interval=300)  # Sample every 10 seconds
                logger.info(f"OCR extraction complete. Location: {ocr_data.get('location')}, Date: {ocr_data.get('video_date')}")
            except Exception as e:
                logger.error(f"OCR extraction failed: {e}")
                ocr_data = None
        
        # Always use chunked processing for all videos
        logger.info(f"Processing video using chunked approach (30s chunks)...")
        
        # Import chunked processor
        from processing.chunked_video_processor import ChunkedVideoProcessor
        
        # Create chunked processor
        processor = ChunkedVideoProcessor(max_workers=4, chunk_duration=30)
        
        # Process video with chunking
        result = processor.process_video(video_path, output_dir)
        
        if result['status'] != 'success':
            logger.error(f"Chunked processing failed: {result.get('error')}")
            return {
                'success': False,
                'error': result.get('error', 'Chunked processing failed')
            }
        
        # Convert chunked results to standard format
        detections = result['detections']
        
        # Group by person_id to create person_tracks
        # Only include PERSON IDs, filter out any UNKNOWN IDs
        person_tracks = {}
        for det in detections:
            person_id = det.get('person_id', '')
            
            # Skip if not a valid PERSON ID
            if not person_id.startswith('PERSON-'):
                continue
                
            if person_id not in person_tracks:
                person_tracks[person_id] = []
            
            # Convert to standard detection format
            detection = {
                'frame_number': det['global_frame_num'],
                'timestamp': det['timestamp'],
                'bbox': det['bbox'],
                'confidence': det['confidence'],
                'person_id': person_id,
                'track_id': det.get('track_id', 0),
                'is_recognized': det.get('recognized_id') is not None,
                'recognition_confidence': det.get('confidence', 0.0) if det.get('recognized_id') else 0.0
            }
            person_tracks[person_id].append(detection)
        
        # Use the annotated video from chunked processing
        annotated_video_path = result['annotated_video']
        
        # Extract person data is already done by chunked processor
        # Just organize the extracted crops for PERSON IDs only
        for person_id in person_tracks:
            # Skip if not a valid PERSON ID (extra safety check)
            if not person_id.startswith('PERSON-'):
                continue
                
            person_dir = os.path.join(persons_dir, person_id)
            
            # Find person crops in chunk directories
            import glob
            import shutil
            
            os.makedirs(person_dir, exist_ok=True)
            
            # Search for this person's crops in all chunk directories
            chunk_dirs = glob.glob(os.path.join(output_dir, "chunk_*"))
            moved_count = 0
            
            for chunk_dir in chunk_dirs:
                person_crops = glob.glob(os.path.join(chunk_dir, f"{person_id}_*.jpg"))
                for crop_file in person_crops:
                    if os.path.exists(crop_file):
                        dest_file = os.path.join(person_dir, os.path.basename(crop_file))
                        shutil.move(crop_file, dest_file)
                        moved_count += 1
                        
            if moved_count > 0:
                logger.info(f"Moved {moved_count} crops for {person_id}")
                    
            # Create metadata for each person
            detections = person_tracks[person_id]
            person_metadata = {
                'person_id': person_id,
                'total_detections': len(detections),
                'first_appearance': detections[0]['timestamp'],
                'last_appearance': detections[-1]['timestamp'],
                'avg_confidence': sum(d['confidence'] for d in detections) / len(detections),
                'created_at': datetime.now().isoformat()
            }
            
            metadata_path = os.path.join(person_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(person_metadata, f, indent=2)
        
        # Create summary report
        summary = {
            'video_path': video_path,
            'file_size_mb': file_size_mb,
            'processing_method': 'chunked',  # Always chunked now
            'annotated_video_path': annotated_video_path,
            'output_directory': output_dir,
            'persons_directory': persons_dir,
            'total_persons': len(person_tracks),
            'processing_completed': datetime.now().isoformat(),
            'ocr_data': ocr_data,  # Add OCR extracted data
            'person_summary': {
                person_id: {
                    'total_detections': len(detections),
                    'duration': detections[-1]['timestamp'] - detections[0]['timestamp'],
                    'avg_confidence': sum(d['confidence'] for d in detections) / len(detections)
                }
                for person_id, detections in person_tracks.items()
            }
        }
        
        summary_path = os.path.join(output_dir, 'processing_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Enhanced processing completed. Output: {output_dir}")
        return {
            'success': True,
            'output_dir': output_dir,
            'annotated_video': annotated_video_path,
            'person_tracks': person_tracks,
            'summary': summary
        }
        
    except Exception as e:
        logger.error(f"Enhanced processing failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def enhanced_person_detection_task(video_path):
    """Task wrapper for enhanced detection to integrate with existing processing system"""
    try:
        result = process_video_with_enhanced_detection(video_path)
        
        if result and result['success']:
            # Convert person_tracks to database format
            detections_for_db = []
            
            for person_id, detections in result['person_tracks'].items():
                for detection in detections:
                    db_detection = {
                        'frame_number': detection['frame_number'],
                        'timestamp': detection['timestamp'],
                        'x': detection['bbox'][0],
                        'y': detection['bbox'][1],
                        'width': detection['bbox'][2],
                        'height': detection['bbox'][3],
                        'confidence': detection['confidence'],
                        'class_name': 'person',
                        'person_id': detection['person_id'],
                        'track_id': detection['track_id']
                    }
                    detections_for_db.append(db_detection)
            
            return {
                'detections': detections_for_db,
                'annotated_video_path': result['annotated_video'],
                'processing_summary': result['summary']
            }
        else:
            return {'error': result.get('error', 'Unknown error')}
            
    except Exception as e:
        logger.error(f"Enhanced detection task failed: {e}")
        return {'error': str(e)}