import os
import sys
import logging
from threading import Lock
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

# Import the centralized PersonIDManager
try:
    from processing.person_id_manager import get_person_id_manager
    PERSON_ID_MANAGER_AVAILABLE = True
except ImportError:
    PERSON_ID_MANAGER_AVAILABLE = False
    get_person_id_manager = None

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


class ImprovedSharedStateManagerV3:
    """Enhanced shared state manager that prioritizes recognition and reuses existing person IDs"""
    
    def __init__(self):
        self.lock = Lock()
        
        # ID counters
        self.unknown_counter = 0
        self.person_counter = self._get_next_person_id_from_folders()
        
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
        
        # Recognition mapping cache
        self.recognized_to_person_id = {}  # recognized_id -> existing person_id
        self._load_existing_person_mappings()
        
    def _get_next_person_id_from_folders(self) -> int:
        """Get the next available person ID by checking existing person folders"""
        persons_dir = Path('processing/outputs/persons')
        persons_dir.mkdir(parents=True, exist_ok=True)
        
        # Check existing folders to find the maximum ID
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
        
        # Start from the next available ID
        return max_folder_id
        
    def _load_existing_person_mappings(self):
        """Load existing person mappings from the database or recognition model"""
        try:
            # Try to load from the default recognition model
            config_path = Path('models/person_recognition/config.json')
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                    default_model = config.get('default_model')
                    
                    if default_model:
                        model_dir = Path('models/person_recognition') / default_model
                        persons_file = model_dir / 'persons.json'
                        
                        if persons_file.exists():
                            with open(persons_file) as f:
                                persons_data = json.load(f)
                                
                                # Build mapping from person name to their PERSON ID
                                for person_name, person_info in persons_data.items():
                                    if 'person_id' in person_info:
                                        # Map the person name (as used in recognition) to their PERSON ID
                                        self.recognized_to_person_id[person_name] = person_info['person_id']
                                        logger.info(f"Loaded existing mapping: {person_name} -> {person_info['person_id']}")
                                        
                                logger.info(f"Loaded {len(self.recognized_to_person_id)} existing person mappings")
                                
        except Exception as e:
            logger.warning(f"Could not load existing person mappings: {e}")
            
    def get_next_unknown_id(self) -> str:
        """Get next available unknown ID"""
        with self.lock:
            self.unknown_counter += 1
            return f"UNKNOWN-{self.unknown_counter:04d}"
            
    def get_next_person_id(self) -> str:
        """Get next available person ID"""
        with self.lock:
            if PERSON_ID_MANAGER_AVAILABLE:
                # Use PersonIDManager to ensure globally unique IDs
                person_id_manager = get_person_id_manager()
                return person_id_manager.get_or_create_person_id(None)
            else:
                # Fallback to local counter
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
            
    def finalize_person_ids(self, auto_resolve_violations: bool = False) -> Dict[str, str]:
        """Convert UNKNOWN IDs to final PERSON IDs with improved recognition handling"""
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
            
            # IMPROVED: Assign PERSON IDs to recognized groups
            for recognized_id, unknown_ids in recognized_groups.items():
                # Check if this group has any violations
                has_violations = any(uid in violation_unknowns for uid in unknown_ids)
                
                if has_violations:
                    # This entire group needs review
                    for unknown_id in unknown_ids:
                        self.unknown_to_final_id[unknown_id] = f"REVIEW-{unknown_id}"
                else:
                    # Use PersonIDManager if available
                    if PERSON_ID_MANAGER_AVAILABLE:
                        person_id_manager = get_person_id_manager()
                        person_id = person_id_manager.get_or_create_person_id(recognized_id)
                        logger.info(f"PersonIDManager assigned {person_id} for recognized person {recognized_id}")
                        
                        for unknown_id in unknown_ids:
                            self.unknown_to_final_id[unknown_id] = person_id
                    else:
                        # Fallback to local mapping
                        if recognized_id in self.recognized_to_person_id:
                            # Reuse existing PERSON ID
                            existing_person_id = self.recognized_to_person_id[recognized_id]
                            logger.info(f"Reusing existing PERSON ID for recognized person {recognized_id}: {existing_person_id}")
                            for unknown_id in unknown_ids:
                                self.unknown_to_final_id[unknown_id] = existing_person_id
                        else:
                            # This is a new recognized person, assign new PERSON ID
                            person_id = self.get_next_person_id()
                            logger.info(f"Creating new PERSON ID for recognized person {recognized_id}: {person_id}")
                            
                            # Store this mapping for future use
                            self.recognized_to_person_id[recognized_id] = person_id
                            
                            for unknown_id in unknown_ids:
                                self.unknown_to_final_id[unknown_id] = person_id
                    
            # Assign unique PERSON IDs to unrecognized unknowns
            logger.info(f"[INFO] Resolving IDs: {len(recognized_groups)} recognized groups, "
                       f"{len(unrecognized_unknowns)} unrecognized persons")
            
            for unknown_id in unrecognized_unknowns:
                if unknown_id not in violation_unknowns:
                    person_id = self.get_next_person_id()
                    self.unknown_to_final_id[unknown_id] = person_id
                    logger.info(f"[NEW] Creating new PERSON ID: {person_id} for unrecognized person")
                
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
                'unique_tracks': len(self.track_to_unknown_id),
                'reused_person_ids': sum(1 for uid, pid in self.unknown_to_final_id.items() 
                                       if pid in self.recognized_to_person_id.values())
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
            'statistics': self.get_statistics(),
            'recognized_mappings': self.recognized_to_person_id
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