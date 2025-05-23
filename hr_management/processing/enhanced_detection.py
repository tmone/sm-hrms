"""
Enhanced Person Detection and Tracking System
- Detects persons in video frames
- Tracks same person across multiple frames
- Draws bounding boxes directly on video frames
- Creates annotated video with detection overlays
- Extracts person images and metadata for face recognition
"""

import os
import cv2
import numpy as np
from datetime import datetime
import json
import tempfile
import shutil

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLO not available")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available")

def process_video_with_enhanced_detection(video_path, output_base_dir="processing/outputs"):
    """
    Process video with enhanced person detection and tracking
    Creates annotated video and person datasets
    """
    print(f"🎬 Starting enhanced video processing: {video_path}")
    
    # Setup output directories
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(output_base_dir, f"detected_{video_name}")
    persons_dir = os.path.join(output_dir, "persons")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(persons_dir, exist_ok=True)
    
    # Process video
    try:
        # Step 1: Detect and track persons across frames
        person_tracks = detect_and_track_persons(video_path)
        print(f"👥 Found {len(person_tracks)} unique persons in video")
        
        # Step 2: Create annotated video with bounding boxes
        annotated_video_path = create_annotated_video(video_path, person_tracks, output_dir)
        print(f"🎥 Created annotated video: {annotated_video_path}")
        
        # Step 3: Extract person images and metadata
        extract_persons_data(video_path, person_tracks, persons_dir)
        print(f"👤 Extracted person data to: {persons_dir}")
        
        # Step 4: Generate processing summary
        summary = generate_processing_summary(person_tracks, output_dir)
        
        return {
            'success': True,
            'person_tracks': person_tracks,
            'annotated_video': annotated_video_path,
            'persons_directory': persons_dir,
            'summary': summary,
            'output_directory': output_dir
        }
        
    except Exception as e:
        print(f"❌ Enhanced detection failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'person_tracks': [],
            'output_directory': output_dir
        }

def detect_and_track_persons(video_path):
    """
    Detect persons and track them across frames to identify unique individuals
    """
    print(f"🔍 Detecting and tracking persons in: {video_path}")
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Video info: {total_frames} frames at {fps} FPS")
    
    # Initialize person tracker
    person_tracker = PersonTracker()
    
    frame_number = 0
    process_every_n_frames = max(1, int(fps // 2))  # Process 2 times per second
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every N frames for efficiency
        if frame_number % process_every_n_frames == 0:
            print(f"🔄 Processing frame {frame_number}/{total_frames} ({frame_number/total_frames*100:.1f}%)")
            
            # Detect persons in current frame
            detections = detect_persons_in_frame(frame, frame_number, fps)
            
            # Update person tracker
            person_tracker.update(detections, frame_number)
        
        frame_number += 1
        
        # Limit processing for demo (remove in production)
        if frame_number > fps * 60:  # Process max 1 minute
            print("⏹️ Stopping at 1 minute limit (demo)")
            break
    
    cap.release()
    
    # Get final person tracks
    person_tracks = person_tracker.get_tracks()
    print(f"🎯 Tracking completed: {len(person_tracks)} unique persons identified")
    
    return person_tracks

def detect_persons_in_frame(frame, frame_number, fps):
    """
    Detect all persons in a single frame using YOLO
    """
    detections = []
    
    if YOLO_AVAILABLE:
        try:
            # Load YOLO model (cached after first load)
            if not hasattr(detect_persons_in_frame, 'model'):
                print("📥 Loading YOLO model for person detection...")
                detect_persons_in_frame.model = YOLO('yolov8n.pt')
            
            model = detect_persons_in_frame.model
            
            # Run detection
            results = model(frame, classes=[0], verbose=False)  # Class 0 = person
            
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        
                        if confidence > 0.5:  # Higher confidence for tracking
                            # Convert to our detection format
                            detection = {
                                'frame_number': frame_number,
                                'timestamp': frame_number / fps,
                                'bbox': {
                                    'x': int(x1),
                                    'y': int(y1),
                                    'width': int(x2 - x1),
                                    'height': int(y2 - y1)
                                },
                                'confidence': float(confidence),
                                'center': (int((x1 + x2) / 2), int((y1 + y2) / 2))
                            }
                            detections.append(detection)
            
        except Exception as e:
            print(f"⚠️ YOLO detection failed: {e}")
    
    # Fallback to mock detections if YOLO unavailable
    if not detections and not YOLO_AVAILABLE:
        detections = create_mock_detections(frame_number, fps)
    
    return detections

def create_mock_detections(frame_number, fps):
    """Create mock detections for demo purposes"""
    detections = []
    
    # Simulate 2-3 persons moving around
    if frame_number % 30 < 20:  # Person 1 appears for 20 out of every 30 frames
        detections.append({
            'frame_number': frame_number,
            'timestamp': frame_number / fps,
            'bbox': {
                'x': 20 + (frame_number % 100) * 2,  # Moving across screen
                'y': 30,
                'width': 80,
                'height': 150
            },
            'confidence': 0.85,
            'center': (60 + (frame_number % 100) * 2, 105)
        })
    
    if frame_number % 45 < 30:  # Person 2 appears less frequently
        detections.append({
            'frame_number': frame_number,
            'timestamp': frame_number / fps,
            'bbox': {
                'x': 200,
                'y': 40 + (frame_number % 50),  # Moving vertically
                'width': 70,
                'height': 140
            },
            'confidence': 0.78,
            'center': (235, 110 + (frame_number % 50))
        })
    
    return detections

class PersonTracker:
    """
    Track persons across frames to identify unique individuals
    Solves the problem of one person being detected as multiple persons
    """
    
    def __init__(self, max_distance=100, max_frames_lost=30):
        self.tracks = {}  # track_id -> track_data
        self.next_track_id = 1
        self.max_distance = max_distance
        self.max_frames_lost = max_frames_lost
    
    def update(self, detections, frame_number):
        """Update tracker with new detections from current frame"""
        
        # Match detections to existing tracks
        matched_tracks = set()
        unmatched_detections = []
        
        for detection in detections:
            best_track_id = None
            best_distance = float('inf')
            
            # Find closest existing track
            for track_id, track in self.tracks.items():
                if not track.get('active', True):
                    continue
                
                # Calculate distance from last known position
                last_center = track['last_center']
                current_center = detection['center']
                distance = self.calculate_distance(last_center, current_center)
                
                if distance < best_distance and distance < self.max_distance:
                    best_distance = distance
                    best_track_id = track_id
            
            if best_track_id is not None:
                # Update existing track
                self.tracks[best_track_id]['frames'].append(detection)
                self.tracks[best_track_id]['last_center'] = detection['center']
                self.tracks[best_track_id]['last_frame'] = frame_number
                self.tracks[best_track_id]['active'] = True
                matched_tracks.add(best_track_id)
            else:
                # Create new track
                unmatched_detections.append(detection)
        
        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            track_id = self.next_track_id
            self.next_track_id += 1
            
            self.tracks[track_id] = {
                'person_id': f"PERSON-{track_id:04d}",
                'frames': [detection],
                'last_center': detection['center'],
                'last_frame': frame_number,
                'first_frame': frame_number,
                'active': True
            }
        
        # Deactivate tracks that haven't been updated recently
        for track_id, track in self.tracks.items():
            if track['last_frame'] < frame_number - self.max_frames_lost:
                track['active'] = False
    
    def calculate_distance(self, center1, center2):
        """Calculate Euclidean distance between two centers"""
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def get_tracks(self):
        """Get all person tracks with at least minimum frames"""
        min_frames = 5  # Minimum frames to consider a valid person
        
        valid_tracks = []
        for track in self.tracks.values():
            if len(track['frames']) >= min_frames:
                valid_tracks.append({
                    'person_id': track['person_id'],
                    'frames': track['frames'],
                    'frame_count': len(track['frames']),
                    'first_appearance': track['first_frame'],
                    'last_appearance': track['last_frame']
                })
        
        return valid_tracks

def create_annotated_video(video_path, person_tracks, output_dir):
    """
    Create new video with bounding boxes drawn on frames
    """
    print(f"🎨 Creating annotated video with {len(person_tracks)} person tracks")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Output video path
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"detected_{video_name}.mp4")
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create frame-to-detections mapping for fast lookup
    frame_detections = {}
    for track in person_tracks:
        for frame_data in track['frames']:
            frame_num = frame_data['frame_number']
            if frame_num not in frame_detections:
                frame_detections[frame_num] = []
            frame_detections[frame_num].append({
                'bbox': frame_data['bbox'],
                'person_id': track['person_id'],
                'confidence': frame_data['confidence']
            })
    
    # Process video frame by frame
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw bounding boxes if detections exist for this frame
        if frame_number in frame_detections:
            for detection in frame_detections[frame_number]:
                draw_detection_box(frame, detection)
        
        # Write annotated frame
        out.write(frame)
        frame_number += 1
        
        if frame_number % 100 == 0:
            print(f"📝 Annotated {frame_number} frames")
    
    cap.release()
    out.release()
    
    print(f"✅ Annotated video created: {output_path}")
    return output_path

def draw_detection_box(frame, detection):
    """
    Draw bounding box and label on frame
    """
    bbox = detection['bbox']
    person_id = detection['person_id']
    confidence = detection['confidence']
    
    # Box coordinates
    x1, y1 = bbox['x'], bbox['y']
    x2, y2 = x1 + bbox['width'], y1 + bbox['height']
    
    # Colors for different persons (cycle through colors)
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green  
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]
    person_num = int(person_id.split('-')[1]) if '-' in person_id else 0
    color = colors[person_num % len(colors)]
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Draw label background
    label = f"{person_id} ({confidence:.2f})"
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
    
    # Draw label text
    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def extract_persons_data(video_path, person_tracks, persons_dir):
    """
    Extract person images and metadata for each tracked person
    """
    print(f"👤 Extracting person data for {len(person_tracks)} persons")
    
    cap = cv2.VideoCapture(video_path)
    
    for track in person_tracks:
        person_id = track['person_id']
        person_dir = os.path.join(persons_dir, person_id)
        os.makedirs(person_dir, exist_ok=True)
        
        # Extract sample images from different frames
        sample_frames = track['frames'][::max(1, len(track['frames']) // 10)]  # Max 10 samples
        
        images_extracted = 0
        for i, frame_data in enumerate(sample_frames):
            frame_num = frame_data['frame_number']
            bbox = frame_data['bbox']
            
            # Jump to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if ret:
                # Extract person region
                x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
                person_image = frame[y:y+h, x:x+w]
                
                if person_image.size > 0:
                    # Save person image
                    image_path = os.path.join(person_dir, f"{person_id}_frame_{frame_num:06d}.jpg")
                    cv2.imwrite(image_path, person_image)
                    images_extracted += 1
        
        # Save metadata
        metadata = {
            'person_id': person_id,
            'total_frames': len(track['frames']),
            'first_appearance_frame': track['first_appearance'],
            'last_appearance_frame': track['last_appearance'],
            'duration_seconds': (track['last_appearance'] - track['first_appearance']) / 30.0,  # Assuming 30fps
            'images_extracted': images_extracted,
            'detection_timestamps': [f['timestamp'] for f in track['frames']],
            'avg_confidence': sum(f['confidence'] for f in track['frames']) / len(track['frames']),
            'bounding_boxes': [f['bbox'] for f in track['frames']]
        }
        
        metadata_path = os.path.join(person_dir, f"{person_id}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   📁 {person_id}: {images_extracted} images, {len(track['frames'])} frames")
    
    cap.release()

def generate_processing_summary(person_tracks, output_dir):
    """
    Generate processing summary and statistics
    """
    summary = {
        'processing_timestamp': datetime.now().isoformat(),
        'total_persons_detected': len(person_tracks),
        'person_details': []
    }
    
    for track in person_tracks:
        person_summary = {
            'person_id': track['person_id'],
            'frame_count': track['frame_count'],
            'first_appearance': track['first_appearance'],
            'last_appearance': track['last_appearance'],
            'avg_confidence': sum(f['confidence'] for f in track['frames']) / len(track['frames'])
        }
        summary['person_details'].append(person_summary)
    
    # Save summary
    summary_path = os.path.join(output_dir, 'processing_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📋 Processing summary saved: {summary_path}")
    return summary

# Integration function for the existing system
def enhanced_person_detection_task(video_path):
    """
    Enhanced person detection task for integration with existing workflow
    """
    try:
        print(f"🚀 Starting enhanced person detection for: {video_path}")
        
        result = process_video_with_enhanced_detection(video_path)
        
        if result['success']:
            # Convert to format expected by existing system
            detections_for_db = []
            
            for track in result['person_tracks']:
                for frame_data in track['frames']:
                    detection = {
                        'person_id': track['person_id'],
                        'frame_number': frame_data['frame_number'],
                        'timestamp': frame_data['timestamp'],
                        'bbox': frame_data['bbox'],
                        'confidence': frame_data['confidence']
                    }
                    detections_for_db.append(detection)
            
            result['detections_for_db'] = detections_for_db
            print(f"✅ Enhanced detection completed: {len(detections_for_db)} detections")
        
        return result
        
    except Exception as e:
        print(f"❌ Enhanced detection failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'detections_for_db': []
        }