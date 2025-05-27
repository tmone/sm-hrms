"""
Person Recognition Inference
Uses trained models to recognize persons in videos
"""

import cv2
import numpy as np
import face_recognition
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import time
from .person_recognition_trainer import PersonRecognitionTrainer

class PersonRecognitionInference:
    def __init__(self, model_name: str, confidence_threshold: float = 0.6):
        self.trainer = PersonRecognitionTrainer()
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        
        # Load model
        self.model, self.scaler, self.metadata, self.person_id_mapping = \
            self.trainer.load_model(model_name)
        
        # Track recognition results
        self.recognition_results = defaultdict(list)
        
    def process_video(self, video_path: str, output_path: str = None,
                     skip_frames: int = 5, show_preview: bool = False) -> Dict:
        """
        Process video and recognize persons
        
        Args:
            video_path: Path to input video
            output_path: Path for annotated output video (optional)
            skip_frames: Process every N frames to speed up
            show_preview: Show live preview during processing
            
        Returns:
            Recognition results
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer if output path specified
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process video
        frame_count = 0
        processed_frames = 0
        start_time = time.time()
        
        # Recognition tracking
        person_tracks = defaultdict(lambda: {
            'detections': [],
            'first_seen': None,
            'last_seen': None,
            'confidence_scores': []
        })
        
        print(f"🎥 Processing video: {video_path}")
        print(f"   Total frames: {total_frames}, FPS: {fps}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = frame_count / fps
            
            # Process frame if not skipping
            if frame_count % skip_frames == 0:
                # Detect faces
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame, model='hog')
                
                if face_locations:
                    # Get face encodings
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    
                    # Recognize each face
                    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                        # Predict person
                        result = self.trainer.predict_person(
                            face_encoding, 
                            self.model_name,
                            self.confidence_threshold
                        )
                        
                        person_id = result['person_id']
                        confidence = result['confidence']
                        
                        # Update tracking
                        if person_id != 'unknown':
                            detection = {
                                'frame': frame_count,
                                'time': current_time,
                                'bbox': (left, top, right, bottom),
                                'confidence': confidence
                            }
                            
                            person_tracks[person_id]['detections'].append(detection)
                            person_tracks[person_id]['confidence_scores'].append(confidence)
                            
                            if person_tracks[person_id]['first_seen'] is None:
                                person_tracks[person_id]['first_seen'] = current_time
                            person_tracks[person_id]['last_seen'] = current_time
                        
                        # Draw on frame
                        if output_path or show_preview:
                            self._draw_recognition(frame, (left, top, right, bottom), 
                                                 person_id, confidence)
                
                processed_frames += 1
                
                # Show progress
                if processed_frames % 100 == 0:
                    elapsed = time.time() - start_time
                    fps_actual = processed_frames / elapsed
                    progress = (frame_count / total_frames) * 100
                    print(f"   Progress: {progress:.1f}%, FPS: {fps_actual:.1f}")
            
            # Write frame if output specified
            if out:
                out.write(frame)
            
            # Show preview if requested
            if show_preview:
                cv2.imshow('Person Recognition', cv2.resize(frame, (960, 540)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
        
        # Clean up
        cap.release()
        if out:
            out.release()
        if show_preview:
            cv2.destroyAllWindows()
        
        # Compile results
        results = {
            'video_path': video_path,
            'model_used': self.model_name,
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'persons_detected': {}
        }
        
        # Summarize person detections
        for person_id, track_data in person_tracks.items():
            avg_confidence = np.mean(track_data['confidence_scores'])
            duration = track_data['last_seen'] - track_data['first_seen']
            
            results['persons_detected'][person_id] = {
                'detection_count': len(track_data['detections']),
                'first_seen': track_data['first_seen'],
                'last_seen': track_data['last_seen'],
                'duration': duration,
                'avg_confidence': float(avg_confidence),
                'min_confidence': float(min(track_data['confidence_scores'])),
                'max_confidence': float(max(track_data['confidence_scores']))
            }
        
        # Processing stats
        total_time = time.time() - start_time
        results['processing_time'] = total_time
        results['processing_fps'] = processed_frames / total_time
        
        print(f"✅ Video processing complete!")
        print(f"   Time: {total_time:.1f}s")
        print(f"   Persons found: {len(results['persons_detected'])}")
        
        return results
    
    def _draw_recognition(self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
                         person_id: str, confidence: float):
        """Draw recognition results on frame"""
        left, top, right, bottom = bbox
        
        # Choose color based on confidence
        if person_id == 'unknown':
            color = (128, 128, 128)  # Gray for unknown
        elif confidence > 0.8:
            color = (0, 255, 0)  # Green for high confidence
        elif confidence > 0.6:
            color = (0, 165, 255)  # Orange for medium confidence
        else:
            color = (0, 0, 255)  # Red for low confidence
        
        # Draw bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Draw label
        label = f"{person_id}: {confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        
        # Draw label background
        cv2.rectangle(frame, (left, top - label_size[1] - 4), 
                     (left + label_size[0], top), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (left, top - 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def process_image(self, image_path: str) -> Dict:
        """Process a single image for person recognition"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {'error': 'Failed to load image'}
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_image)
        
        if not face_locations:
            return {'persons': [], 'message': 'No faces detected'}
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        results = {'persons': []}
        
        # Recognize each face
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Predict person
            prediction = self.trainer.predict_person(
                face_encoding,
                self.model_name,
                self.confidence_threshold
            )
            
            results['persons'].append({
                'person_id': prediction['person_id'],
                'confidence': prediction['confidence'],
                'bbox': {
                    'left': left,
                    'top': top,
                    'right': right,
                    'bottom': bottom
                },
                'all_probabilities': prediction['all_probabilities']
            })
        
        return results
    
    def batch_process_images(self, image_paths: List[str]) -> Dict:
        """Process multiple images"""
        results = {
            'images': {},
            'summary': defaultdict(lambda: {
                'count': 0,
                'confidences': []
            })
        }
        
        for image_path in image_paths:
            image_results = self.process_image(image_path)
            results['images'][image_path] = image_results
            
            # Update summary
            for person in image_results.get('persons', []):
                person_id = person['person_id']
                results['summary'][person_id]['count'] += 1
                results['summary'][person_id]['confidences'].append(person['confidence'])
        
        # Calculate average confidences
        for person_id, data in results['summary'].items():
            data['avg_confidence'] = np.mean(data['confidences'])
            data['confidences'] = data['confidences']  # Keep raw data
        
        return results