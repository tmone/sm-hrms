from celery import current_task
from celery_app import celery
from app import create_app
from models.base import db
from models.video import Video, DetectedPerson
from models.face_recognition import FaceDataset, TrainedModel
import os
import json
from datetime import datetime
import uuid

# Optional CV2 import
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV not available. Video processing will be limited.")

app = create_app()

@celery.task(bind=True)
def process_video(self, video_id):
    """Process a video for person detection and face extraction"""
    with app.app_context():
        video = Video.query.get(video_id)
        if not video:
            return {'error': 'Video not found'}
        
        try:
            video.status = 'processing'
            video.processing_progress = 0
            db.session.commit()
            
            # Step 1: Extract video metadata
            self.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Extracting metadata'})
            metadata = extract_video_metadata(video.filepath)
            
            video.duration = metadata.get('duration')
            video.fps = metadata.get('fps')
            video.resolution = metadata.get('resolution')
            video.processing_progress = 20
            db.session.commit()
            
            # Step 2: Detect persons
            self.update_state(state='PROGRESS', meta={'progress': 30, 'status': 'Detecting persons'})
            detections = detect_persons_in_video(video.filepath)
            video.processing_progress = 60
            db.session.commit()
            
            # Step 3: Save detections to database
            self.update_state(state='PROGRESS', meta={'progress': 80, 'status': 'Saving detections'})
            save_detections_to_db(video_id, detections, metadata.get('fps', 25))
            
            # Step 4: Complete processing
            video.status = 'completed'
            video.processing_progress = 100
            db.session.commit()
            
            self.update_state(state='SUCCESS', meta={'progress': 100, 'status': 'Completed'})
            
            return {
                'video_id': video_id,
                'detections_count': len(detections),
                'status': 'completed'
            }
            
        except Exception as e:
            video.status = 'failed'
            video.error_message = str(e)
            db.session.commit()
            
            self.update_state(state='FAILURE', meta={'error': str(e)})
            raise

@celery.task(bind=True)
def extract_faces_from_detections(self, dataset_id, video_ids=None):
    """Extract face images from detected persons"""
    with app.app_context():
        dataset = FaceDataset.query.get(dataset_id)
        if not dataset:
            return {'error': 'Dataset not found'}
        
        try:
            dataset.status = 'processing'
            db.session.commit()
            
            # Get videos to process
            if video_ids:
                videos = Video.query.filter(Video.id.in_(video_ids), Video.status == 'completed').all()
            else:
                videos = Video.query.filter_by(status='completed').all()
            
            total_faces = 0
            person_codes = set()
            
            for i, video in enumerate(videos):
                self.update_state(state='PROGRESS', meta={
                    'progress': int((i / len(videos)) * 100),
                    'status': f'Processing video {i+1}/{len(videos)}'
                })
                
                detections = DetectedPerson.query.filter_by(video_id=video.id).all()
                
                for detection in detections:
                    faces_extracted = extract_faces_from_detection(
                        video.filepath, 
                        detection, 
                        dataset.dataset_path
                    )
                    total_faces += faces_extracted
                    person_codes.add(detection.person_code)
                    
                    detection.face_count = faces_extracted
                    db.session.commit()
            
            # Update dataset statistics
            dataset.image_count = total_faces
            dataset.person_count = len(person_codes)
            dataset.status = 'ready'
            db.session.commit()
            
            return {
                'dataset_id': dataset_id,
                'total_faces': total_faces,
                'person_count': len(person_codes),
                'status': 'completed'
            }
            
        except Exception as e:
            dataset.status = 'error'
            db.session.commit()
            raise

@celery.task(bind=True)
def train_face_recognition_model(self, model_id):
    """Train a face recognition model"""
    with app.app_context():
        model = TrainedModel.query.get(model_id)
        if not model:
            return {'error': 'Model not found'}
        
        try:
            model.status = 'training'
            model.training_progress = 0
            db.session.commit()
            
            # Load dataset
            dataset = FaceDataset.query.get(model.dataset_id)
            if not dataset or dataset.status != 'ready':
                raise Exception('Dataset not ready for training')
            
            # Training simulation (replace with actual training logic)
            for epoch in range(model.epochs):
                self.update_state(state='PROGRESS', meta={
                    'progress': int((epoch / model.epochs) * 100),
                    'status': f'Epoch {epoch+1}/{model.epochs}'
                })
                
                # Simulate training progress
                import time
                time.sleep(1)  # Remove in production
                
                model.training_progress = int((epoch / model.epochs) * 100)
                db.session.commit()
            
            # Set final metrics (these would come from actual training)
            model.accuracy = 0.95  # Simulated
            model.loss = 0.05
            model.validation_accuracy = 0.92
            model.validation_loss = 0.08
            model.status = 'completed'
            model.training_progress = 100
            db.session.commit()
            
            return {
                'model_id': model_id,
                'accuracy': model.accuracy,
                'status': 'completed'
            }
            
        except Exception as e:
            model.status = 'failed'
            model.error_message = str(e)
            db.session.commit()
            raise

def extract_video_metadata(filepath):
    """Extract metadata from video file"""
    if not CV2_AVAILABLE:
        # Return mock metadata when OpenCV is not available
        file_size = os.path.getsize(filepath)
        return {
            'duration': 60.0,  # Mock 1 minute duration
            'fps': 25.0,
            'frame_count': 1500,
            'resolution': "1920x1080",
            'width': 1920,
            'height': 1080
        }
    
    cap = cv2.VideoCapture(filepath)
    
    if not cap.isOpened():
        raise Exception(f"Could not open video file: {filepath}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        'duration': duration,
        'fps': fps,
        'frame_count': frame_count,
        'resolution': f"{width}x{height}",
        'width': width,
        'height': height
    }

def detect_persons_in_video(filepath):
    """Detect persons in video using YOLO or similar"""
    # This is a simplified implementation
    # In production, you would use actual person detection models
    
    cap = cv2.VideoCapture(filepath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    detections = []
    current_persons = {}
    person_counter = 1
    
    frame_number = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Simulate person detection (replace with actual detection logic)
        # This would use YOLO, SAM, or other detection models
        
        # For demo purposes, create some fake detections
        if frame_number % 30 == 0:  # Every 30 frames
            person_code = f"PERSON-{person_counter:04d}"
            start_frame = frame_number
            start_time = frame_number / fps
            
            # Simulate person appearing for 5 seconds
            end_frame = start_frame + int(5 * fps)
            end_time = start_time + 5
            
            detections.append({
                'person_code': person_code,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'start_time': start_time,
                'end_time': end_time,
                'confidence': 0.85,
                'bbox_data': [{
                    'frame': start_frame,
                    'x': 20,  # Percentage
                    'y': 30,
                    'width': 15,
                    'height': 40
                }]
            })
            
            person_counter += 1
        
        frame_number += 1
        
        # Limit processing for demo
        if frame_number > 300:  # Process only first 300 frames
            break
    
    cap.release()
    return detections

def save_detections_to_db(video_id, detections, fps):
    """Save detection results to database"""
    for detection in detections:
        db_detection = DetectedPerson(
            video_id=video_id,
            person_code=detection['person_code'],
            start_frame=detection['start_frame'],
            end_frame=detection['end_frame'],
            start_time=detection['start_time'],
            end_time=detection['end_time'],
            confidence=detection['confidence'],
            bbox_data=detection['bbox_data']
        )
        
        db.session.add(db_detection)
    
    db.session.commit()

def extract_faces_from_detection(video_filepath, detection, dataset_path):
    """Extract face images from a specific detection"""
    # Create person directory
    person_dir = os.path.join(dataset_path, detection.person_code)
    os.makedirs(person_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_filepath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    faces_extracted = 0
    
    # Jump to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, detection.start_frame)
    
    for frame_num in range(detection.start_frame, detection.end_frame, int(fps)):  # Extract every second
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Extract face from frame (simplified)
        # In production, you would use face detection and cropping
        
        # For demo, save a small crop from the frame
        height, width = frame.shape[:2]
        
        # Use bbox data if available
        if detection.bbox_data:
            bbox = detection.bbox_data[0]  # Use first bbox
            x = int(bbox['x'] / 100 * width)
            y = int(bbox['y'] / 100 * height)
            w = int(bbox['width'] / 100 * width)
            h = int(bbox['height'] / 100 * height)
            
            face_crop = frame[y:y+h, x:x+w]
        else:
            # Fallback to center crop
            crop_size = min(width, height) // 4
            start_x = (width - crop_size) // 2
            start_y = (height - crop_size) // 2
            face_crop = frame[start_y:start_y+crop_size, start_x:start_x+crop_size]
        
        # Resize to standard face size
        face_resized = cv2.resize(face_crop, (128, 128))
        
        # Save face image
        face_filename = f"face_{detection.person_code}_frame_{frame_num:06d}.jpg"
        face_path = os.path.join(person_dir, face_filename)
        cv2.imwrite(face_path, face_resized)
        
        faces_extracted += 1
    
    cap.release()
    return faces_extracted