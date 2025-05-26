"""
GPU-Accelerated Person Tracking with Deep Appearance Features

This module implements state-of-the-art person tracking using:
1. ResNet18 for extracting deep appearance features
2. GPU-accelerated batch processing
3. Hybrid tracking combining appearance and position
4. Efficient similarity computation using matrix operations
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from typing import List, Dict, Tuple, Optional
import cv2
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AppearanceFeatureExtractor:
    """
    Extracts deep appearance features using ResNet18 on GPU
    """
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load ResNet18 pretrained on ImageNet
        self.model = models.resnet18(pretrained=True)
        
        # Remove the final classification layer to get feature vectors
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 64)),  # Standard person crop size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"✅ Appearance feature extractor initialized on {self.device}")
    
    def extract_batch(self, person_crops: List[np.ndarray]) -> torch.Tensor:
        """
        Extract features from a batch of person crops
        
        Args:
            person_crops: List of person image crops (BGR format)
            
        Returns:
            Normalized feature vectors of shape (N, 512)
        """
        if not person_crops:
            return torch.empty(0, 512).to(self.device)
        
        # Preprocess all crops
        batch = []
        for crop in person_crops:
            # Convert BGR to RGB
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = self.transform(crop_rgb)
            batch.append(tensor)
        
        # Stack into batch tensor
        batch_tensor = torch.stack(batch).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(batch_tensor)
            features = features.squeeze(-1).squeeze(-1)  # Remove spatial dimensions
            
            # L2 normalize features
            features = torch.nn.functional.normalize(features, p=2, dim=1)
        
        return features


class GPUPersonTracker:
    """
    GPU-accelerated person tracker with appearance features
    """
    
    def __init__(self, 
                 appearance_weight: float = 0.7,
                 position_weight: float = 0.3,
                 max_distance: float = 100.0,
                 max_frames_lost: int = 30,
                 min_similarity: float = 0.5,
                 device: str = 'cuda'):
        """
        Initialize tracker
        
        Args:
            appearance_weight: Weight for appearance similarity (0-1)
            position_weight: Weight for position similarity (0-1)
            max_distance: Maximum pixel distance for position matching
            max_frames_lost: Frames before a track is considered lost
            min_similarity: Minimum similarity threshold for matching
            device: Device to run on ('cuda' or 'cpu')
        """
        self.appearance_weight = appearance_weight
        self.position_weight = position_weight
        self.max_distance = max_distance
        self.max_frames_lost = max_frames_lost
        self.min_similarity = min_similarity
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize feature extractor
        self.feature_extractor = AppearanceFeatureExtractor(device)
        
        # Track storage
        self.tracks = {}  # track_id -> track_data
        self.next_track_id = 1
        
        # Feature storage for active tracks
        self.track_features = {}  # track_id -> feature tensor
        
        logger.info(f"✅ GPU Person Tracker initialized on {self.device}")
        logger.info(f"   Appearance weight: {appearance_weight}")
        logger.info(f"   Position weight: {position_weight}")
    
    def update(self, detections: List[Dict], frame: np.ndarray, frame_number: int) -> List[Dict]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of detection dicts with bbox info
            frame: Current frame (for extracting appearance)
            frame_number: Current frame number
            
        Returns:
            List of detections with track IDs assigned
        """
        if not detections:
            self._update_lost_tracks(frame_number)
            return []
        
        # Extract person crops
        person_crops = []
        detection_centers = []
        
        for det in detections:
            x, y, w, h = det['bbox']
            
            # Extract person crop with padding
            padding = 5
            x1 = max(0, int(x - padding))
            y1 = max(0, int(y - padding))
            x2 = min(frame.shape[1], int(x + w + padding))
            y2 = min(frame.shape[0], int(y + h + padding))
            
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                person_crops.append(crop)
                detection_centers.append(torch.tensor([x + w/2, y + h/2], device=self.device))
        
        if not person_crops:
            return detections
        
        # Extract appearance features for all detections in batch
        detection_features = self.feature_extractor.extract_batch(person_crops)
        detection_centers_tensor = torch.stack(detection_centers)
        
        # Get active tracks
        active_track_ids = [tid for tid, track in self.tracks.items() 
                           if track['last_frame'] >= frame_number - self.max_frames_lost]
        
        if active_track_ids:
            # Compute similarities with all active tracks
            track_assignments = self._compute_similarities_batch(
                detection_features, 
                detection_centers_tensor,
                active_track_ids
            )
        else:
            track_assignments = [-1] * len(detections)
        
        # Update tracks and assign IDs
        tracked_detections = []
        for i, (det, assignment) in enumerate(zip(detections, track_assignments)):
            if assignment >= 0:
                # Update existing track
                track_id = assignment
                self._update_track(track_id, det, detection_features[i], frame_number)
            else:
                # Create new track
                track_id = self._create_track(det, detection_features[i], frame_number)
            
            # Add track info to detection
            det['track_id'] = track_id
            det['person_id'] = f"PERSON-{track_id:04d}"
            tracked_detections.append(det)
        
        # Update lost tracks
        self._update_lost_tracks(frame_number)
        
        return tracked_detections
    
    def _compute_similarities_batch(self, 
                                   detection_features: torch.Tensor,
                                   detection_centers: torch.Tensor,
                                   active_track_ids: List[int]) -> List[int]:
        """
        Compute similarities between all detections and tracks in batch
        
        Returns:
            List of track assignments (-1 for new track)
        """
        num_detections = len(detection_features)
        num_tracks = len(active_track_ids)
        
        # Stack track features and centers
        track_features_list = []
        track_centers_list = []
        
        for track_id in active_track_ids:
            if track_id in self.track_features:
                track_features_list.append(self.track_features[track_id])
                track = self.tracks[track_id]
                center = torch.tensor([track['last_center'][0], track['last_center'][1]], 
                                    device=self.device)
                track_centers_list.append(center)
        
        if not track_features_list:
            return [-1] * num_detections
        
        track_features_tensor = torch.stack(track_features_list)
        track_centers_tensor = torch.stack(track_centers_list)
        
        # Compute appearance similarities using matrix multiplication
        # Shape: (num_detections, num_tracks)
        appearance_similarities = torch.mm(detection_features, track_features_tensor.t())
        
        # Compute position distances
        # Shape: (num_detections, 1, 2) - (1, num_tracks, 2) = (num_detections, num_tracks, 2)
        position_distances = torch.cdist(detection_centers.unsqueeze(0), 
                                       track_centers_tensor.unsqueeze(0)).squeeze(0)
        
        # Convert distances to similarities (0-1 range)
        position_similarities = 1.0 - torch.clamp(position_distances / self.max_distance, 0, 1)
        
        # Combine similarities
        combined_similarities = (self.appearance_weight * appearance_similarities + 
                               self.position_weight * position_similarities)
        
        # Find best matches using Hungarian algorithm simulation
        assignments = [-1] * num_detections
        used_tracks = set()
        
        # Convert to numpy for easier manipulation
        sim_matrix = combined_similarities.cpu().numpy()
        
        # Greedy assignment (can be replaced with scipy.optimize.linear_sum_assignment)
        for _ in range(min(num_detections, num_tracks)):
            # Find best remaining match
            best_val = -1
            best_det = -1
            best_track = -1
            
            for i in range(num_detections):
                if assignments[i] >= 0:
                    continue
                for j in range(num_tracks):
                    if j in used_tracks:
                        continue
                    if sim_matrix[i, j] > best_val and sim_matrix[i, j] > self.min_similarity:
                        best_val = sim_matrix[i, j]
                        best_det = i
                        best_track = j
            
            if best_det >= 0:
                assignments[best_det] = active_track_ids[best_track]
                used_tracks.add(best_track)
        
        return assignments
    
    def _create_track(self, detection: Dict, features: torch.Tensor, frame_number: int) -> int:
        """Create a new track"""
        track_id = self.next_track_id
        self.next_track_id += 1
        
        x, y, w, h = detection['bbox']
        center = (x + w/2, y + h/2)
        
        self.tracks[track_id] = {
            'first_frame': frame_number,
            'last_frame': frame_number,
            'last_center': center,
            'last_bbox': detection['bbox'],
            'detections': [detection],
            'active': True
        }
        
        self.track_features[track_id] = features
        
        return track_id
    
    def _update_track(self, track_id: int, detection: Dict, features: torch.Tensor, frame_number: int):
        """Update existing track"""
        track = self.tracks[track_id]
        
        x, y, w, h = detection['bbox']
        center = (x + w/2, y + h/2)
        
        track['last_frame'] = frame_number
        track['last_center'] = center
        track['last_bbox'] = detection['bbox']
        track['detections'].append(detection)
        
        # Update features with exponential moving average
        alpha = 0.1  # Learning rate
        self.track_features[track_id] = (1 - alpha) * self.track_features[track_id] + alpha * features
        self.track_features[track_id] = torch.nn.functional.normalize(self.track_features[track_id], p=2, dim=0)
    
    def _update_lost_tracks(self, frame_number: int):
        """Mark tracks as inactive if lost for too long"""
        for track_id, track in self.tracks.items():
            if track['active'] and track['last_frame'] < frame_number - self.max_frames_lost:
                track['active'] = False
                # Remove features to save memory
                if track_id in self.track_features:
                    del self.track_features[track_id]


def process_video_with_gpu_tracking(video_path: str, output_path: str = None):
    """
    Process video with GPU-accelerated tracking
    
    Args:
        video_path: Path to input video
        output_path: Path to save annotated video (optional)
    """
    try:
        from ultralytics import YOLO
        
        # Initialize models
        logger.info("🚀 Initializing GPU-accelerated tracking...")
        detector = YOLO('yolov8n.pt')
        tracker = GPUPersonTracker()
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"📹 Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup video writer if output path provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        frame_number = 0
        all_tracks = {}
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect persons
            results = detector(frame, classes=[0], verbose=False)  # Class 0 is person
            
            # Convert to our detection format
            detections = []
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0])
                        
                        detections.append({
                            'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                            'confidence': confidence,
                            'frame_number': frame_number,
                            'timestamp': frame_number / fps
                        })
            
            # Update tracker
            tracked_detections = tracker.update(detections, frame, frame_number)
            
            # Store tracks
            if frame_number not in all_tracks:
                all_tracks[frame_number] = []
            all_tracks[frame_number].extend(tracked_detections)
            
            # Draw on frame if output requested
            if out:
                annotated_frame = draw_tracked_persons(frame, tracked_detections)
                out.write(annotated_frame)
            
            frame_number += 1
            
            # Progress update
            if frame_number % 100 == 0:
                logger.info(f"Processed {frame_number}/{total_frames} frames...")
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        
        # Summary
        unique_persons = len(tracker.tracks)
        logger.info(f"✅ Tracking complete: {unique_persons} unique persons tracked")
        
        return all_tracks, tracker.tracks
        
    except Exception as e:
        logger.error(f"❌ GPU tracking failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def draw_tracked_persons(frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Draw bounding boxes with track IDs"""
    annotated = frame.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for det in detections:
        x, y, w, h = det['bbox']
        track_id = det.get('track_id', 0)
        person_id = det.get('person_id', 'UNKNOWN')
        
        # Choose color based on track ID
        color = colors[track_id % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(annotated, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
        
        # Draw label
        label = f"{person_id} ({det['confidence']:.2f})"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        cv2.rectangle(annotated, (int(x), int(y) - label_size[1] - 10),
                     (int(x) + label_size[0], int(y)), color, -1)
        cv2.putText(annotated, label, (int(x), int(y) - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return annotated


if __name__ == "__main__":
    # Test the tracker
    import sys
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else "tracked_output.mp4"
        
        tracks, track_data = process_video_with_gpu_tracking(video_path, output_path)
        
        if tracks:
            print(f"\n✅ Successfully tracked {len(track_data)} unique persons")
            for tid, data in track_data.items():
                duration = (data['last_frame'] - data['first_frame']) / 30.0  # Assuming 30fps
                print(f"   Person {tid}: {len(data['detections'])} detections, {duration:.1f}s duration")