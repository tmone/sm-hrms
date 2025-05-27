"""
Person Recognition Inference (Simple Version)
Uses trained models to recognize persons without face_recognition library
"""

import cv2
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import time
from .person_recognition_trainer import PersonRecognitionTrainer
from .person_dataset_creator_simple import PersonDatasetCreatorSimple

class PersonRecognitionInferenceSimple:
    def __init__(self, model_name: str, confidence_threshold: float = 0.6):
        self.trainer = PersonRecognitionTrainer()
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        
        # Load model
        self.model, self.scaler, self.metadata, self.person_id_mapping = \
            self.trainer.load_model(model_name)
        
        # Feature extractor
        self.feature_extractor = PersonDatasetCreatorSimple()
        
        # Track recognition results
        self.recognition_results = defaultdict(list)
        
    def process_video(self, video_path: str, output_path: str = None,
                     skip_frames: int = 5, show_preview: bool = False) -> Dict:
        """
        Process video and recognize persons using simple features
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
        
        # For simple version, we'll process the entire frame
        # In a real scenario, you'd want person detection first
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = frame_count / fps
            
            # Process frame if not skipping
            if frame_count % skip_frames == 0:
                # Extract features from frame
                features = self._extract_frame_features(frame)
                
                if features is not None:
                    # Predict person
                    result = self._predict_person(features)
                    
                    person_id = result['person_id']
                    confidence = result['confidence']
                    
                    # Update tracking
                    if person_id != 'unknown':
                        detection = {
                            'frame': frame_count,
                            'time': current_time,
                            'confidence': confidence
                        }
                        
                        person_tracks[person_id]['detections'].append(detection)
                        person_tracks[person_id]['confidence_scores'].append(confidence)
                        
                        if person_tracks[person_id]['first_seen'] is None:
                            person_tracks[person_id]['first_seen'] = current_time
                        person_tracks[person_id]['last_seen'] = current_time
                    
                    # Draw on frame
                    if output_path or show_preview:
                        self._draw_recognition(frame, person_id, confidence)
                
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
    
    def _extract_frame_features(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract features from a frame"""
        try:
            # Save frame temporarily
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            cv2.imwrite(temp_file.name, frame)
            
            # Extract features
            features = self.feature_extractor._extract_simple_features(
                temp_file.name, 'temp', 'temp.jpg'
            )
            
            # Clean up
            Path(temp_file.name).unlink()
            
            return features
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def _predict_person(self, features: np.ndarray) -> Dict:
        """Predict person from features"""
        # Prepare features
        features_array = np.array(features).reshape(1, -1)
        features_scaled = self.scaler.transform(features_array)
        
        # Get prediction and probabilities
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Get confidence for predicted class
        confidence = probabilities[prediction]
        
        # Check threshold
        if confidence < self.confidence_threshold:
            return {
                'person_id': 'unknown',
                'confidence': float(confidence),
                'all_probabilities': {}
            }
        
        # Get person ID
        person_id = self.person_id_mapping[prediction]
        
        # Get all probabilities
        all_probs = {
            self.person_id_mapping[i]: float(prob) 
            for i, prob in enumerate(probabilities)
        }
        
        return {
            'person_id': person_id,
            'confidence': float(confidence),
            'all_probabilities': all_probs
        }
    
    def _draw_recognition(self, frame: np.ndarray, person_id: str, confidence: float):
        """Draw recognition results on frame"""
        # Draw text at top of frame
        label = f"Person: {person_id} ({confidence:.2f})"
        
        # Choose color based on confidence
        if person_id == 'unknown':
            color = (128, 128, 128)  # Gray for unknown
        elif confidence > 0.8:
            color = (0, 255, 0)  # Green for high confidence
        elif confidence > 0.6:
            color = (0, 165, 255)  # Orange for medium confidence
        else:
            color = (0, 0, 255)  # Red for low confidence
        
        # Draw background rectangle
        cv2.rectangle(frame, (10, 10), (400, 40), (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(frame, label, (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    def process_image(self, image_path: str) -> Dict:
        """Process a single image for person recognition"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {'error': 'Failed to load image'}
        
        # Extract features
        features = self.feature_extractor._extract_simple_features(
            image_path, 'temp', Path(image_path).name
        )
        
        if features is None:
            return {'persons': [], 'message': 'Failed to extract features'}
        
        # Predict person
        prediction = self._predict_person(features)
        
        return {
            'persons': [{
                'person_id': prediction['person_id'],
                'confidence': prediction['confidence'],
                'all_probabilities': prediction['all_probabilities']
            }]
        }