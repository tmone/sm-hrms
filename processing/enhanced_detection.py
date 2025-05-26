import cv2
import numpy as np
import os
import json
import uuid
from datetime import datetime
from ultralytics import YOLO
import logging
from typing import List, Dict, Tuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonTracker:
    """Track persons across video frames to solve duplicate detection problem"""
    
    def __init__(self, max_distance=50, min_confidence=0.5):
        self.tracks = {}  # track_id -> track_data
        self.next_track_id = 1
        self.max_distance = max_distance
        self.min_confidence = min_confidence
        
    def euclidean_distance(self, box1, box2):
        """Calculate distance between two bounding box centers"""
        center1 = (box1[0] + box1[2]/2, box1[1] + box1[3]/2)
        center2 = (box2[0] + box2[2]/2, box2[1] + box2[3]/2)
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def update_tracks(self, detections, frame_number):
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
                self.tracks[track_id] = {
                    'first_frame': frame_number,
                    'last_frame': frame_number,
                    'last_bbox': bbox,
                    'detections': [detection],
                    'person_id': f"PERSON-{track_id:04d}"
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
        return {}
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return {}
    
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
        tracked_detections = tracker.update_tracks(detections, frame_number)
        
        # Store in all_tracks by frame
        all_tracks[frame_number] = tracked_detections
        
        frame_number += 1
        
        # Progress logging
        if frame_number % 100 == 0:
            logger.info(f"Processed {frame_number}/{total_frames} frames")
    
    cap.release()
    
    # Organize tracks by person_id
    person_tracks = {}
    for frame_data in all_tracks.values():
        for detection in frame_data:
            person_id = detection['person_id']
            if person_id not in person_tracks:
                person_tracks[person_id] = []
            person_tracks[person_id].append(detection)
    
    logger.info(f"Detected {len(person_tracks)} unique persons across {frame_number} frames")
    return person_tracks

def create_annotated_video(video_path, person_tracks, output_dir):
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
    
    # Setup output video writer
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_video_path = os.path.join(output_dir, f"detected_{video_name}.mp4")
    
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
                
                # Draw label
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
                    img_filename = f"{person_id}_frame_{frame_number:06d}.jpg"
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

def process_video_with_enhanced_detection(video_path, output_base_dir="processing/outputs"):
    """Main function to process video with enhanced person detection and tracking"""
    logger.info(f"Starting enhanced video processing: {video_path}")
    
    # Setup output directories
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(output_base_dir, f"detected_{video_name}")
    persons_dir = os.path.join(output_dir, "persons")
    
    try:
        # Step 1: Detect and track persons across frames
        logger.info("Step 1: Detecting and tracking persons...")
        person_tracks = detect_and_track_persons(video_path)
        
        if not person_tracks:
            logger.warning("No persons detected in video")
            return None
        
        # Step 2: Create annotated video with bounding boxes
        logger.info("Step 2: Creating annotated video...")
        annotated_video_path = create_annotated_video(video_path, person_tracks, output_dir)
        
        # Step 3: Extract person images and metadata
        logger.info("Step 3: Extracting person data...")
        extract_persons_data(video_path, person_tracks, persons_dir)
        
        # Create summary report
        summary = {
            'video_path': video_path,
            'annotated_video_path': annotated_video_path,
            'output_directory': output_dir,
            'persons_directory': persons_dir,
            'total_persons': len(person_tracks),
            'processing_completed': datetime.now().isoformat(),
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