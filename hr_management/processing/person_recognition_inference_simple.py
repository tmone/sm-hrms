"""
Person Recognition Inference (Simple Version)
Uses trained models to recognize persons without face_recognition library
"""

import cv2
import numpy as np
from pathlib import Path
import json
import os
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import time
from ultralytics import YOLO
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
        
        # Load YOLO for person detection
        try:
            self.yolo_model = YOLO('yolov8n.pt')
        except Exception as e:
            print(f"Warning: Failed to load YOLO model: {e}")
        
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
        
        print(f"[MOVIE] Processing video: {video_path}")
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
                # Detect persons in the frame first
                detected_persons = self._detect_persons_in_image(frame)
                
                if detected_persons:
                    if processed_frames == 0:
                        print(f"[OK] Detected {len(detected_persons)} persons in first processed frame")
                    # Process each detected person
                    for x1, y1, x2, y2, detection_confidence in detected_persons:
                        # Crop the person from the frame
                        person_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                        
                        # Skip if crop is too small
                        if person_crop.shape[0] < 50 or person_crop.shape[1] < 20:
                            continue
                        
                        # Save cropped image temporarily
                        import tempfile
                        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                        temp_path = temp_file.name
                        temp_file.close()  # Close the file handle immediately
                        
                        cv2.imwrite(temp_path, person_crop)
                        
                        try:
                            # Extract features from cropped person
                            features = self.feature_extractor._extract_simple_features(
                                temp_path, 'temp', f'frame_{frame_count}.jpg'
                            )
                            
                            if features is not None:
                                # Predict person
                                result = self._predict_person(features)
                                
                                person_id = result['person_id']
                                confidence = result['confidence']
                                
                                # Update tracking only if confidence is high enough
                                if confidence >= self.confidence_threshold:
                                    detection = {
                                        'frame': frame_count,
                                        'time': current_time,
                                        'confidence': confidence,
                                        'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                                        'detection_confidence': detection_confidence
                                    }
                                    
                                    person_tracks[person_id]['detections'].append(detection)
                                    person_tracks[person_id]['confidence_scores'].append(confidence)
                                    
                                    if person_tracks[person_id]['first_seen'] is None:
                                        person_tracks[person_id]['first_seen'] = current_time
                                    person_tracks[person_id]['last_seen'] = current_time
                                    
                                    # Draw on frame
                                    if output_path or show_preview:
                                        self._draw_recognition_box(frame, x1, y1, x2, y2, person_id, confidence)
                        finally:
                            # Clean up temp file with retry for Windows
                            try:
                                os.unlink(temp_path)
                            except PermissionError:
                                # Windows sometimes holds file locks, ignore cleanup
                                # Temp files will be cleaned up by OS
                                pass
                            except Exception:
                                # Ignore any other cleanup errors
                                pass
                else:
                    # No persons detected in this frame
                    if processed_frames == 0:
                        print(f"[WARNING]  No persons detected in first processed frame")
                
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
        
        print(f"[OK] Video processing complete!")
        print(f"   Time: {total_time:.1f}s")
        print(f"   Persons found: {len(results['persons_detected'])}")
        
        return results
    
    def _extract_frame_features(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract features from a frame"""
        try:
            # Save frame temporarily
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            temp_path = temp_file.name
            temp_file.close()  # Close file handle immediately
            
            cv2.imwrite(temp_path, frame)
            
            # Extract features
            features = self.feature_extractor._extract_simple_features(
                temp_path, 'temp', 'temp.jpg'
            )
            
            # Clean up with Windows-friendly approach
            try:
                os.unlink(temp_path)
            except:
                pass  # Ignore cleanup errors
            
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
    
    def _draw_recognition_box(self, frame: np.ndarray, x1: float, y1: float, x2: float, y2: float,
                              person_id: str, confidence: float):
        """Draw recognition bounding box and label on frame"""
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
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Prepare label
        label = f"{person_id} ({confidence:.2f})"
        
        # Get label size
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Draw label background
        cv2.rectangle(frame, (int(x1), int(y1) - label_h - 10), 
                     (int(x1) + label_w, int(y1)), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (int(x1), int(y1) - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def process_image(self, image_path: str) -> Dict:
        """Process a single image for person recognition"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {'error': 'Failed to load image'}
        
        # Check if this might be a pre-cropped person image
        h, w = image.shape[:2]
        aspect_ratio = h / w if w > 0 else 0
        
        # Heuristic: If image has person-like aspect ratio and reasonable size,
        # it might be pre-cropped
        is_likely_cropped = (
            1.5 <= aspect_ratio <= 3.0 and  # Person images are typically taller
            100 <= w <= 500 and             # Reasonable width for cropped person
            150 <= h <= 800                 # Reasonable height for cropped person
        )
        
        # Detect persons in the image first
        detected_persons = self._detect_persons_in_image(image)
        
        if not detected_persons:
            # If no persons detected but image looks like it might be pre-cropped,
            # try processing it as a cropped image
            if is_likely_cropped:
                print(f"No persons detected, but image appears to be pre-cropped (size: {w}x{h}, aspect: {aspect_ratio:.2f})")
                return self.process_cropped_image(image_path)
            else:
                return {'persons': [], 'message': 'No persons detected in the image'}
        
        # Process each detected person
        recognized_persons = []
        
        for i, (x1, y1, x2, y2, detection_confidence) in enumerate(detected_persons):
            # Crop the person from the image
            person_crop = image[int(y1):int(y2), int(x1):int(x2)]
            
            # Save cropped image temporarily with proper cleanup
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            temp_path = temp_file.name
            temp_file.close()  # Close the file handle before cv2 uses it
            
            cv2.imwrite(temp_path, person_crop)
            
            try:
                # Extract features from cropped person
                features = self.feature_extractor._extract_simple_features(
                    temp_path, 'temp', f'person_{i}.jpg'
                )
                
                if features is not None:
                    # Predict person
                    prediction = self._predict_person(features)
                    
                    recognized_persons.append({
                        'person_id': prediction['person_id'],
                        'confidence': prediction['confidence'],
                        'detection_confidence': float(detection_confidence),
                        'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                        'all_probabilities': prediction['all_probabilities']
                    })
            finally:
                # Clean up temp file - more robust for Windows
                try:
                    os.unlink(temp_path)
                except PermissionError:
                    # On Windows, sometimes the file is still in use
                    # Try again after a short delay
                    import time
                    time.sleep(0.1)
                    try:
                        os.unlink(temp_path)
                    except:
                        # If still fails, ignore - temp files will be cleaned up later
                        pass
        
        return {
            'persons': recognized_persons,
            'total_detections': len(detected_persons)
        }
    
    def process_cropped_image(self, image_path: str) -> Dict:
        """Process a pre-cropped person image for recognition"""
        print(f"[SEARCH] process_cropped_image called with: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Failed to load image: {image_path}")
            return {'error': 'Failed to load image'}
        
        print(f"[OK] Image loaded successfully: {image.shape}")
        
        # Since this is already a cropped person image, process it directly
        features = self.feature_extractor._extract_simple_features(
            image_path, 'temp', Path(image_path).name
        )
        
        if features is None:
            print(f"[ERROR] Failed to extract features from image")
            return {'persons': [], 'message': 'Failed to extract features from image'}
        
        print(f"[OK] Features extracted: shape {features.shape}")
        
        # Predict person
        prediction = self._predict_person(features)
        print(f"[TARGET] Prediction: {prediction}")
        
        # Get image dimensions for bbox
        h, w = image.shape[:2]
        
        return {
            'persons': [{
                'person_id': prediction['person_id'],
                'confidence': prediction['confidence'],
                'detection_confidence': 1.0,  # Already cropped, so detection confidence is 1.0
                'bbox': [0, 0, w, h],
                'all_probabilities': prediction['all_probabilities']
            }],
            'total_detections': 1,
            'is_cropped': True
        }
    
    def _detect_persons_in_image(self, image: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        """Detect persons in image using YOLO"""
        if not hasattr(self, 'yolo_model'):
            return []
        
        # Run YOLO detection
        results = self.yolo_model(image, verbose=False)
        
        persons = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    # Check if it's a person (class 0 in COCO)
                    if int(box.cls[0]) == 0:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0])
                        
                        # Filter by confidence
                        if confidence > 0.2:  # Lower threshold for detection in videos
                            persons.append((x1, y1, x2, y2, confidence))
        
        return persons