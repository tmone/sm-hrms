import os
import sys
import logging
from threading import Lock
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class FrameViolation:
    """Represents a violation where same person appears multiple times in frame"""
    frame_num: int
    chunk_idx: int
    person_id: str
    track_ids: List[int]
    positions: List[Tuple[int, int, int, int]]  # List of bboxes
    timestamp: float


@dataclass
class DetectionInfo:
    """Information about a single detection"""
    unknown_id: str
    track_id: int
    chunk_idx: int
    frame_num: int
    bbox: Tuple[int, int, int, int]
    recognized_id: Optional[str]
    confidence: float
    timestamp: float


class ImprovedSharedStateManager:
    """Enhanced shared state manager with violation detection"""
    
    def __init__(self):
        self.lock = Lock()
        
        # ID counters
        self.unknown_counter = 0
        self.person_counter = 0
        
        # Tracking mappings
        self.track_to_unknown_id = {}  # track_id -> unknown_id
        self.unknown_to_detection_info = {}  # unknown_id -> DetectionInfo
        self.unknown_to_final_id = {}  # unknown_id -> final person_id
        
        # Frame-level tracking for violation detection
        self.frame_detections = defaultdict(list)  # (chunk_idx, frame_num) -> [DetectionInfo]
        self.recognized_id_frames = defaultdict(set)  # recognized_id -> set of (chunk_idx, frame_num)
        
        # Violation tracking
        self.violations = []
        self.violation_resolutions = {}  # frame_key -> resolution strategy
        
    def get_next_unknown_id(self) -> str:
        """Get next available unknown ID"""
        with self.lock:
            self.unknown_counter += 1
            return f"UNKNOWN-{self.unknown_counter:04d}"
            
    def get_next_person_id(self) -> str:
        """Get next available person ID"""
        with self.lock:
            self.person_counter += 1
            return f"PERSON-{self.person_counter:04d}"
            
    def assign_temporary_id(self, recognized_id: Optional[str], track_id: int,
                           chunk_idx: int, frame_num: int, bbox: Tuple[int, int, int, int],
                           confidence: float, timestamp: float) -> str:
        """Assign temporary UNKNOWN ID with enhanced tracking"""
        with self.lock:
            # Check if this track already has an unknown ID
            if track_id in self.track_to_unknown_id:
                unknown_id = self.track_to_unknown_id[track_id]
            else:
                # Create new unknown ID
                unknown_id = self.get_next_unknown_id()
                self.track_to_unknown_id[track_id] = unknown_id
            
            # Create detection info
            detection_info = DetectionInfo(
                unknown_id=unknown_id,
                track_id=track_id,
                chunk_idx=chunk_idx,
                frame_num=frame_num,
                bbox=bbox,
                recognized_id=recognized_id,
                confidence=confidence,
                timestamp=timestamp
            )
            
            # Store detection info
            self.unknown_to_detection_info[unknown_id] = detection_info
            
            # Track at frame level
            frame_key = (chunk_idx, frame_num)
            self.frame_detections[frame_key].append(detection_info)
            
            # Track recognized person appearances
            if recognized_id:
                self.recognized_id_frames[recognized_id].add(frame_key)
            
            return unknown_id
            
    def detect_violations(self) -> List[FrameViolation]:
        """Detect frames where same person appears multiple times"""
        with self.lock:
            violations = []
            
            # Check each frame for duplicate recognized persons
            for frame_key, detections in self.frame_detections.items():
                chunk_idx, frame_num = frame_key
                
                # Group detections by recognized_id
                recognized_groups = defaultdict(list)
                for det in detections:
                    if det.recognized_id:
                        recognized_groups[det.recognized_id].append(det)
                
                # Check for violations (same person multiple times)
                for recognized_id, dets in recognized_groups.items():
                    if len(dets) > 1:
                        violation = FrameViolation(
                            frame_num=frame_num,
                            chunk_idx=chunk_idx,
                            person_id=recognized_id,
                            track_ids=[d.track_id for d in dets],
                            positions=[d.bbox for d in dets],
                            timestamp=dets[0].timestamp
                        )
                        violations.append(violation)
                        
            self.violations = violations
            return violations
            
    def resolve_violations(self, strategy: str = 'confidence') -> Dict[str, str]:
        """Resolve violations using specified strategy"""
        with self.lock:
            resolutions = {}
            
            for violation in self.violations:
                frame_key = (violation.chunk_idx, violation.frame_num)
                
                if strategy == 'confidence':
                    # Keep detection with highest confidence
                    frame_dets = self.frame_detections[frame_key]
                    recognized_dets = [d for d in frame_dets 
                                     if d.recognized_id == violation.person_id]
                    
                    if recognized_dets:
                        # Sort by confidence descending
                        recognized_dets.sort(key=lambda x: x.confidence, reverse=True)
                        
                        # Keep first (highest confidence), mark others for removal
                        keep_det = recognized_dets[0]
                        for det in recognized_dets[1:]:
                            resolutions[det.unknown_id] = 'remove'
                            logger.warning(f"Violation resolved: Removing {det.unknown_id} "
                                         f"(duplicate {violation.person_id} in frame {violation.frame_num})")
                                         
                elif strategy == 'center':
                    # Keep detection closest to frame center
                    frame_dets = self.frame_detections[frame_key]
                    recognized_dets = [d for d in frame_dets 
                                     if d.recognized_id == violation.person_id]
                    
                    if recognized_dets:
                        # Assuming frame dimensions, calculate distance to center
                        # This is simplified - in real use, pass frame dimensions
                        frame_center = (960, 540)  # Assume 1920x1080
                        
                        def distance_to_center(det):
                            x, y, w, h = det.bbox
                            det_center = (x + w/2, y + h/2)
                            return ((det_center[0] - frame_center[0])**2 + 
                                   (det_center[1] - frame_center[1])**2)**0.5
                        
                        recognized_dets.sort(key=distance_to_center)
                        
                        # Keep first (closest to center), mark others for removal
                        keep_det = recognized_dets[0]
                        for det in recognized_dets[1:]:
                            resolutions[det.unknown_id] = 'remove'
                            
            self.violation_resolutions = resolutions
            return resolutions
            
    def finalize_person_ids(self, auto_resolve_violations: bool = False) -> Dict[str, str]:
        """Convert UNKNOWN IDs to final PERSON IDs with violation handling"""
        with self.lock:
            # First detect violations
            violations = self.detect_violations()
            
            if violations:
                logger.warning(f"Detected {len(violations)} frame violations that need review")
                
                # Store violations for review instead of auto-resolving
                self.violations_need_review = violations
                
                # Don't auto-resolve by default - violations need manual review
                if auto_resolve_violations:
                    logger.warning("Auto-resolution is not recommended. Violations should be reviewed.")
                    
                # Log violations for review
                for v in violations:
                    logger.warning(f"REVIEW NEEDED - Frame {v.frame_num}: {v.person_id} appears "
                                 f"{len(v.track_ids)} times at positions {v.positions}")
            
            # Group unknowns by recognized ID
            recognized_groups = defaultdict(list)
            unrecognized_unknowns = []
            violation_unknowns = []  # Track IDs involved in violations
            
            # First, identify all unknowns involved in violations
            for violation in violations:
                for track_id in violation.track_ids:
                    # Find unknown ID for this track
                    for unknown_id, det_info in self.unknown_to_detection_info.items():
                        if det_info.track_id == track_id:
                            violation_unknowns.append(unknown_id)
            
            for unknown_id, det_info in self.unknown_to_detection_info.items():
                # Mark violation cases differently
                if unknown_id in violation_unknowns:
                    # Don't assign regular PERSON ID yet - needs review
                    self.unknown_to_final_id[unknown_id] = f"REVIEW-{unknown_id}"
                    continue
                    
                if det_info.recognized_id:
                    recognized_groups[det_info.recognized_id].append(unknown_id)
                else:
                    unrecognized_unknowns.append(unknown_id)
            
            # Assign PERSON IDs to recognized groups (non-violation cases)
            for recognized_id, unknown_ids in recognized_groups.items():
                # Check if this group has any violations
                has_violations = any(uid in violation_unknowns for uid in unknown_ids)
                
                if has_violations:
                    # This entire group needs review
                    for unknown_id in unknown_ids:
                        self.unknown_to_final_id[unknown_id] = f"REVIEW-{unknown_id}"
                else:
                    # Normal assignment
                    person_id = self.get_next_person_id()
                    for unknown_id in unknown_ids:
                        self.unknown_to_final_id[unknown_id] = person_id
                    
            # Assign unique PERSON IDs to unrecognized unknowns
            for unknown_id in unrecognized_unknowns:
                if unknown_id not in violation_unknowns:
                    person_id = self.get_next_person_id()
                    self.unknown_to_final_id[unknown_id] = person_id
                
            return self.unknown_to_final_id
            
    def get_violations_for_review(self) -> List[FrameViolation]:
        """Get violations that need manual review"""
        return getattr(self, 'violations_need_review', [])
            
    def get_frame_summary(self, chunk_idx: int, frame_num: int) -> Dict:
        """Get summary of detections in a specific frame"""
        with self.lock:
            frame_key = (chunk_idx, frame_num)
            detections = self.frame_detections.get(frame_key, [])
            
            summary = {
                'frame_num': frame_num,
                'chunk_idx': chunk_idx,
                'total_detections': len(detections),
                'recognized_persons': defaultdict(int),
                'unrecognized_count': 0,
                'violations': []
            }
            
            for det in detections:
                if det.recognized_id:
                    summary['recognized_persons'][det.recognized_id] += 1
                else:
                    summary['unrecognized_count'] += 1
                    
            # Check for violations in this frame
            for person_id, count in summary['recognized_persons'].items():
                if count > 1:
                    summary['violations'].append({
                        'person_id': person_id,
                        'count': count
                    })
                    
            return summary
            
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        with self.lock:
            stats = {
                'total_unknown_ids': self.unknown_counter,
                'total_detections': len(self.unknown_to_detection_info),
                'total_frames_with_detections': len(self.frame_detections),
                'recognized_persons': len(self.recognized_id_frames),
                'violations_detected': len(self.violations),
                'violations_resolved': len(self.violation_resolutions),
                'unique_tracks': len(self.track_to_unknown_id)
            }
            
            # Calculate average detections per frame
            if stats['total_frames_with_detections'] > 0:
                stats['avg_detections_per_frame'] = (
                    stats['total_detections'] / stats['total_frames_with_detections']
                )
            else:
                stats['avg_detections_per_frame'] = 0
                
            return stats
            
    def export_violations_report(self, filepath: str):
        """Export detailed violations report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'violations': [],
            'resolutions': self.violation_resolutions,
            'statistics': self.get_statistics()
        }
        
        for v in self.violations:
            report['violations'].append({
                'frame_num': v.frame_num,
                'chunk_idx': v.chunk_idx,
                'person_id': v.person_id,
                'track_ids': v.track_ids,
                'positions': v.positions,
                'timestamp': v.timestamp
            })
            
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Violations report exported to {filepath}")