"""
Enhanced Person Tracker - Tracks persons across frames with recognition support
"""
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class EnhancedPersonTracker:
    """Enhanced tracker with recognition support"""
    
    def __init__(self, max_distance: float = 100.0, max_frames_missing: int = 30,
                 use_recognition: bool = True):
        """
        Initialize enhanced person tracker
        
        Args:
            max_distance: Maximum distance to associate detections between frames
            max_frames_missing: Maximum frames a track can be missing before removal
            use_recognition: Whether to use recognition results
        """
        self.max_distance = max_distance
        self.max_frames_missing = max_frames_missing
        self.use_recognition = use_recognition
        
        # Track storage
        self.tracks = {}  # track_id -> track_info
        self.next_track_id = 1
        self.frame_count = 0
        
        # Recognition storage
        self.track_to_person = {}  # track_id -> recognized_person_id
        self.person_to_tracks = defaultdict(list)  # person_id -> [track_ids]
        
    def update(self, boxes: List[List[int]], frame: np.ndarray = None) -> List[int]:
        """
        Update tracks with new detections
        
        Args:
            boxes: List of [x1, y1, x2, y2] bounding boxes
            frame: Current frame (optional, for future enhancements)
            
        Returns:
            List of track IDs corresponding to each box
        """
        self.frame_count += 1
        track_ids = []
        
        # Convert boxes to centers for tracking
        centers = []
        for box in boxes:
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            centers.append((cx, cy))
            
        # Match detections to existing tracks
        matched_tracks = set()
        
        for i, (cx, cy) in enumerate(centers):
            best_track_id = None
            min_distance = float('inf')
            
            # Find closest track
            for track_id, track_info in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                    
                # Calculate distance to last position
                last_cx, last_cy = track_info['last_center']
                distance = np.sqrt((cx - last_cx)**2 + (cy - last_cy)**2)
                
                if distance < min_distance and distance < self.max_distance:
                    min_distance = distance
                    best_track_id = track_id
                    
            if best_track_id is not None:
                # Update existing track
                matched_tracks.add(best_track_id)
                self.tracks[best_track_id]['last_center'] = (cx, cy)
                self.tracks[best_track_id]['last_frame'] = self.frame_count
                self.tracks[best_track_id]['bbox_history'].append(boxes[i])
                track_ids.append(best_track_id)
            else:
                # Create new track
                new_track_id = self.next_track_id
                self.next_track_id += 1
                
                self.tracks[new_track_id] = {
                    'first_frame': self.frame_count,
                    'last_frame': self.frame_count,
                    'last_center': (cx, cy),
                    'bbox_history': [boxes[i]],
                    'recognized_person_id': None
                }
                track_ids.append(new_track_id)
                
        # Remove old tracks
        tracks_to_remove = []
        for track_id, track_info in self.tracks.items():
            if self.frame_count - track_info['last_frame'] > self.max_frames_missing:
                tracks_to_remove.append(track_id)
                
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
            if track_id in self.track_to_person:
                person_id = self.track_to_person[track_id]
                self.person_to_tracks[person_id].remove(track_id)
                del self.track_to_person[track_id]
                
        return track_ids
        
    def set_recognized_person_id(self, track_id: int, person_id: str):
        """
        Set the recognized person ID for a track
        
        Args:
            track_id: Track ID
            person_id: Recognized person ID
        """
        if track_id in self.tracks:
            self.tracks[track_id]['recognized_person_id'] = person_id
            self.track_to_person[track_id] = person_id
            self.person_to_tracks[person_id].append(track_id)
            
    def get_recognized_person_id(self, track_id: int) -> Optional[str]:
        """
        Get the recognized person ID for a track
        
        Args:
            track_id: Track ID
            
        Returns:
            Recognized person ID or None
        """
        return self.track_to_person.get(track_id)
        
    def get_track_info(self, track_id: int) -> Optional[Dict]:
        """
        Get information about a track
        
        Args:
            track_id: Track ID
            
        Returns:
            Track information dictionary or None
        """
        return self.tracks.get(track_id)
        
    def get_active_tracks(self) -> List[int]:
        """
        Get list of currently active track IDs
        
        Returns:
            List of active track IDs
        """
        return list(self.tracks.keys())
        
    def get_track_duration(self, track_id: int) -> int:
        """
        Get the duration of a track in frames
        
        Args:
            track_id: Track ID
            
        Returns:
            Duration in frames or 0 if track not found
        """
        if track_id in self.tracks:
            track = self.tracks[track_id]
            return track['last_frame'] - track['first_frame'] + 1
        return 0
        
    def merge_tracks(self, track_id1: int, track_id2: int):
        """
        Merge two tracks (useful when same person is detected as multiple tracks)
        
        Args:
            track_id1: First track ID (will be kept)
            track_id2: Second track ID (will be merged into first)
        """
        if track_id1 not in self.tracks or track_id2 not in self.tracks:
            return
            
        # Merge bbox history
        track1 = self.tracks[track_id1]
        track2 = self.tracks[track_id2]
        
        # Update time range
        track1['first_frame'] = min(track1['first_frame'], track2['first_frame'])
        track1['last_frame'] = max(track1['last_frame'], track2['last_frame'])
        
        # Merge bbox history (sorted by frame)
        track1['bbox_history'].extend(track2['bbox_history'])
        
        # Update recognition if needed
        if track2.get('recognized_person_id') and not track1.get('recognized_person_id'):
            self.set_recognized_person_id(track_id1, track2['recognized_person_id'])
            
        # Remove second track
        del self.tracks[track_id2]
        if track_id2 in self.track_to_person:
            person_id = self.track_to_person[track_id2]
            self.person_to_tracks[person_id].remove(track_id2)
            del self.track_to_person[track_id2]