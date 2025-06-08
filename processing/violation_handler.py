import os
import sys
import logging
import json
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ViolationCase:
    """Represents a violation case that needs review"""
    case_id: str
    frame_num: int
    chunk_idx: int
    timestamp: float
    violation_type: str  # 'duplicate_recognition', 'tracking_conflict', etc.
    detections: List[Dict]  # List of conflicting detections
    tracking_analysis: Dict  # Analysis of tracking data
    recommended_action: str  # 'assign_new_id', 'merge_tracks', 'use_tracking', etc.
    confidence_scores: List[float]
    spatial_distance: float  # Distance between detections
    
    
class TrackingAnalyzer:
    """Analyzes tracking data to help resolve violations"""
    
    def __init__(self):
        self.track_history = defaultdict(list)  # track_id -> list of frames
        self.track_positions = defaultdict(list)  # track_id -> list of positions
        self.track_velocities = {}  # track_id -> velocity vector
        
    def update_track(self, track_id: int, frame_num: int, position: Tuple[int, int, int, int]):
        """Update tracking history"""
        self.track_history[track_id].append(frame_num)
        center = ((position[0] + position[2]) / 2, (position[1] + position[3]) / 2)
        self.track_positions[track_id].append(center)
        
        # Calculate velocity if we have enough points
        if len(self.track_positions[track_id]) >= 2:
            prev_pos = self.track_positions[track_id][-2]
            curr_pos = self.track_positions[track_id][-1]
            prev_frame = self.track_history[track_id][-2]
            curr_frame = self.track_history[track_id][-1]
            
            if curr_frame > prev_frame:
                velocity = (
                    (curr_pos[0] - prev_pos[0]) / (curr_frame - prev_frame),
                    (curr_pos[1] - prev_pos[1]) / (curr_frame - prev_frame)
                )
                self.track_velocities[track_id] = velocity
                
    def predict_position(self, track_id: int, target_frame: int) -> Optional[Tuple[float, float]]:
        """Predict position based on tracking history"""
        if track_id not in self.track_velocities or track_id not in self.track_history:
            return None
            
        last_frame = self.track_history[track_id][-1]
        last_pos = self.track_positions[track_id][-1]
        velocity = self.track_velocities[track_id]
        
        # Simple linear prediction
        frame_diff = target_frame - last_frame
        predicted_pos = (
            last_pos[0] + velocity[0] * frame_diff,
            last_pos[1] + velocity[1] * frame_diff
        )
        
        return predicted_pos
        
    def analyze_tracks_similarity(self, track1: int, track2: int) -> Dict:
        """Analyze if two tracks might be the same person"""
        analysis = {
            'likely_same_person': False,
            'confidence': 0.0,
            'reasoning': []
        }
        
        if track1 not in self.track_history or track2 not in self.track_history:
            analysis['reasoning'].append("Insufficient tracking data")
            return analysis
            
        # Check temporal overlap
        frames1 = set(self.track_history[track1])
        frames2 = set(self.track_history[track2])
        overlap = frames1.intersection(frames2)
        
        if overlap:
            analysis['reasoning'].append(f"Tracks overlap in {len(overlap)} frames")
            analysis['likely_same_person'] = False
            return analysis
            
        # Check spatial continuity
        # If track2 starts near where track1 ends
        if max(self.track_history[track1]) < min(self.track_history[track2]):
            last_pos_track1 = self.track_positions[track1][-1]
            first_pos_track2 = self.track_positions[track2][0]
            
            distance = np.sqrt(
                (last_pos_track1[0] - first_pos_track2[0])**2 +
                (last_pos_track1[1] - first_pos_track2[1])**2
            )
            
            # If tracks are close in space and time, likely same person
            time_gap = min(self.track_history[track2]) - max(self.track_history[track1])
            
            if distance < 100 and time_gap < 30:  # Within 100 pixels and 1 second (at 30fps)
                analysis['likely_same_person'] = True
                analysis['confidence'] = max(0.9 - distance/200 - time_gap/60, 0.5)
                analysis['reasoning'].append(
                    f"Tracks are spatially close (distance: {distance:.1f}px) "
                    f"and temporally close (gap: {time_gap} frames)"
                )
            else:
                analysis['reasoning'].append(
                    f"Tracks are too far apart: distance={distance:.1f}px, time_gap={time_gap} frames"
                )
                
        return analysis


class ViolationHandler:
    """Handles violations with manual review capabilities"""
    
    def __init__(self):
        self.violations = []
        self.tracking_analyzer = TrackingAnalyzer()
        self.review_decisions = {}  # case_id -> decision
        self.case_counter = 0
        
    def add_violation(self, frame_num: int, chunk_idx: int, timestamp: float,
                     detections: List[Dict]) -> str:
        """Add a violation case for review"""
        self.case_counter += 1
        case_id = f"CASE-{self.case_counter:04d}"
        
        # Analyze the violation
        violation_type = self._determine_violation_type(detections)
        tracking_analysis = self._analyze_tracking(detections)
        recommended_action = self._recommend_action(detections, tracking_analysis)
        
        # Calculate spatial distance between detections
        positions = [d['bbox'] for d in detections]
        spatial_distance = self._calculate_spatial_distance(positions)
        
        # Create violation case
        case = ViolationCase(
            case_id=case_id,
            frame_num=frame_num,
            chunk_idx=chunk_idx,
            timestamp=timestamp,
            violation_type=violation_type,
            detections=detections,
            tracking_analysis=tracking_analysis,
            recommended_action=recommended_action,
            confidence_scores=[d.get('confidence', 0) for d in detections],
            spatial_distance=spatial_distance
        )
        
        self.violations.append(case)
        logger.warning(f"Violation {case_id} added: {violation_type} in frame {frame_num}")
        
        return case_id
        
    def _determine_violation_type(self, detections: List[Dict]) -> str:
        """Determine the type of violation"""
        recognized_ids = [d.get('recognized_id') for d in detections if d.get('recognized_id')]
        
        if len(set(recognized_ids)) == 1 and len(recognized_ids) > 1:
            return 'duplicate_recognition'
        elif len(set([d.get('track_id') for d in detections])) == len(detections):
            return 'multiple_tracks_same_person'
        else:
            return 'tracking_conflict'
            
    def _analyze_tracking(self, detections: List[Dict]) -> Dict:
        """Analyze tracking data for the detections"""
        analysis = {
            'track_continuity': {},
            'spatial_consistency': True,
            'temporal_consistency': True
        }
        
        # Check each track's history
        for det in detections:
            track_id = det.get('track_id')
            if track_id:
                # Simple continuity check
                analysis['track_continuity'][track_id] = {
                    'frames_tracked': len(self.tracking_analyzer.track_history.get(track_id, [])),
                    'has_velocity': track_id in self.tracking_analyzer.track_velocities
                }
                
        return analysis
        
    def _recommend_action(self, detections: List[Dict], tracking_analysis: Dict) -> str:
        """Recommend action based on analysis"""
        # If detections are far apart, likely different people
        positions = [d['bbox'] for d in detections]
        distance = self._calculate_spatial_distance(positions)
        
        if distance > 300:  # More than 300 pixels apart
            return 'assign_new_ids'
            
        # If one track has much longer history, trust it
        track_lengths = [
            tracking_analysis['track_continuity'].get(d.get('track_id', -1), {}).get('frames_tracked', 0)
            for d in detections
        ]
        
        if max(track_lengths) > 2 * min(track_lengths) and max(track_lengths) > 10:
            return 'use_longest_track'
            
        # If tracks might be the same person split across time
        if len(detections) == 2:
            track_ids = [d.get('track_id') for d in detections]
            if all(track_ids):
                similarity = self.tracking_analyzer.analyze_tracks_similarity(track_ids[0], track_ids[1])
                if similarity['likely_same_person']:
                    return 'merge_tracks'
                    
        return 'manual_review_required'
        
    def _calculate_spatial_distance(self, positions: List[Tuple]) -> float:
        """Calculate average distance between detection positions"""
        if len(positions) < 2:
            return 0.0
            
        centers = [((p[0] + p[2])/2, (p[1] + p[3])/2) for p in positions]
        
        total_distance = 0
        count = 0
        
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                distance = np.sqrt(
                    (centers[i][0] - centers[j][0])**2 +
                    (centers[i][1] - centers[j][1])**2
                )
                total_distance += distance
                count += 1
                
        return total_distance / count if count > 0 else 0
        
    def make_decision(self, case_id: str, decision: str, params: Dict = None):
        """Record a decision for a violation case"""
        self.review_decisions[case_id] = {
            'decision': decision,
            'params': params or {},
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Decision recorded for {case_id}: {decision}")
        
    def apply_decisions(self, detections: List[Dict]) -> List[Dict]:
        """Apply review decisions to detections"""
        updated_detections = []
        
        for case in self.violations:
            if case.case_id not in self.review_decisions:
                logger.warning(f"No decision for {case.case_id}, keeping original detections")
                updated_detections.extend(case.detections)
                continue
                
            decision = self.review_decisions[case.case_id]
            
            if decision['decision'] == 'assign_new_ids':
                # Each detection gets a unique ID
                for i, det in enumerate(case.detections):
                    det_copy = det.copy()
                    det_copy['person_id'] = f"PERSON-NEW-{case.case_id}-{i+1}"
                    det_copy['violation_resolved'] = True
                    updated_detections.append(det_copy)
                    
            elif decision['decision'] == 'use_longest_track':
                # Keep only the detection with longest track
                track_lengths = [
                    len(self.tracking_analyzer.track_history.get(d.get('track_id', -1), []))
                    for d in case.detections
                ]
                longest_idx = track_lengths.index(max(track_lengths))
                det_copy = case.detections[longest_idx].copy()
                det_copy['violation_resolved'] = True
                updated_detections.append(det_copy)
                
            elif decision['decision'] == 'merge_tracks':
                # Use first detection but note the merge
                det_copy = case.detections[0].copy()
                det_copy['merged_tracks'] = [d.get('track_id') for d in case.detections]
                det_copy['violation_resolved'] = True
                updated_detections.append(det_copy)
                
            elif decision['decision'] == 'manual_assignment':
                # Use manual assignment from params
                person_id = decision['params'].get('person_id')
                keep_idx = decision['params'].get('keep_detection_index', 0)
                
                if keep_idx < len(case.detections):
                    det_copy = case.detections[keep_idx].copy()
                    det_copy['person_id'] = person_id
                    det_copy['violation_resolved'] = True
                    updated_detections.append(det_copy)
                    
        return updated_detections
        
    def export_review_report(self, filepath: str):
        """Export violations for manual review"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_violations': len(self.violations),
            'reviewed': len(self.review_decisions),
            'pending_review': len(self.violations) - len(self.review_decisions),
            'cases': []
        }
        
        for case in self.violations:
            case_dict = asdict(case)
            case_dict['decision'] = self.review_decisions.get(case.case_id, {'status': 'pending'})
            report['cases'].append(case_dict)
            
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Review report exported to {filepath}")
        
    def get_pending_cases(self) -> List[ViolationCase]:
        """Get cases that need review"""
        return [case for case in self.violations 
                if case.case_id not in self.review_decisions]
                
    def get_case_summary(self, case_id: str) -> Dict:
        """Get detailed summary of a specific case"""
        case = next((c for c in self.violations if c.case_id == case_id), None)
        if not case:
            return {}
            
        summary = {
            'case_id': case.case_id,
            'frame': case.frame_num,
            'type': case.violation_type,
            'num_detections': len(case.detections),
            'recognized_persons': list(set(d.get('recognized_id', 'Unknown') 
                                          for d in case.detections)),
            'spatial_distance': case.spatial_distance,
            'recommended_action': case.recommended_action,
            'confidence_range': (min(case.confidence_scores), max(case.confidence_scores)),
            'decision': self.review_decisions.get(case_id, {'status': 'pending'})
        }
        
        return summary