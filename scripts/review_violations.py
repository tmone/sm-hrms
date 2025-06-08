#!/usr/bin/env python3
"""Script to review and resolve violations from video processing"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from processing.violation_handler import ViolationHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_violations_report(filepath: str) -> dict:
    """Load violations report from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def interactive_review(violations_report: dict):
    """Interactive review of violations"""
    violations = violations_report.get('violations', [])
    
    if not violations:
        logger.info("No violations to review!")
        return {}
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Found {len(violations)} violations to review")
    logger.info(f"{'='*60}\n")
    
    decisions = {}
    
    for i, violation in enumerate(violations):
        logger.info(f"\nViolation {i+1}/{len(violations)}")
        logger.info(f"Frame: {violation['frame_num']}")
        logger.info(f"Person ID: {violation['person_id']}")
        logger.info(f"Track IDs: {violation['track_ids']}")
        logger.info(f"Positions: {violation['positions']}")
        logger.info(f"Number of detections: {len(violation['track_ids'])}")
        
        # Calculate distance between positions
        if len(violation['positions']) == 2:
            pos1, pos2 = violation['positions']
            center1 = ((pos1[0] + pos1[2])/2, (pos1[1] + pos1[3])/2)
            center2 = ((pos2[0] + pos2[2])/2, (pos2[1] + pos2[3])/2)
            distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
            logger.info(f"Distance between detections: {distance:.1f} pixels")
        
        logger.info("\nOptions:")
        logger.info("1. Keep all as same person (merge tracks)")
        logger.info("2. Assign new IDs (different people)")
        logger.info("3. Keep detection with highest confidence")
        logger.info("4. Keep detection with longest track history")
        logger.info("5. Manual assignment")
        logger.info("6. Skip (review later)")
        
        while True:
            choice = input("\nYour choice (1-6): ").strip()
            
            if choice == '1':
                decisions[f"violation_{i}"] = {
                    'action': 'merge_tracks',
                    'person_id': violation['person_id']
                }
                logger.info("[OK] Will merge all tracks as same person")
                break
                
            elif choice == '2':
                decisions[f"violation_{i}"] = {
                    'action': 'assign_new_ids',
                    'track_mapping': {}
                }
                for j, track_id in enumerate(violation['track_ids']):
                    new_id = f"{violation['person_id']}_v{j+1}"
                    decisions[f"violation_{i}"]['track_mapping'][track_id] = new_id
                logger.info("[OK] Will assign new IDs to each detection")
                break
                
            elif choice == '3':
                decisions[f"violation_{i}"] = {
                    'action': 'use_highest_confidence'
                }
                logger.info("[OK] Will keep detection with highest confidence")
                break
                
            elif choice == '4':
                decisions[f"violation_{i}"] = {
                    'action': 'use_longest_track'
                }
                logger.info("[OK] Will keep detection with longest track history")
                break
                
            elif choice == '5':
                # Manual assignment
                logger.info("\nManual assignment:")
                track_assignments = {}
                
                for track_id in violation['track_ids']:
                    person_id = input(f"  Person ID for track {track_id}: ").strip()
                    if person_id:
                        track_assignments[track_id] = person_id
                        
                decisions[f"violation_{i}"] = {
                    'action': 'manual_assignment',
                    'track_assignments': track_assignments
                }
                logger.info("[OK] Manual assignments recorded")
                break
                
            elif choice == '6':
                logger.info("⏭️  Skipping for later review")
                break
                
            else:
                logger.warning("Invalid choice. Please enter 1-6.")
    
    return decisions


def apply_decisions(violations_report: dict, decisions: dict, output_dir: str):
    """Apply decisions to create final ID mappings"""
    logger.info("\nApplying decisions...")
    
    # Create ID remapping based on decisions
    id_remapping = {}
    
    for i, violation in enumerate(violations_report.get('violations', [])):
        decision_key = f"violation_{i}"
        if decision_key not in decisions:
            continue
            
        decision = decisions[decision_key]
        action = decision['action']
        
        if action == 'merge_tracks':
            # All tracks get the same person ID
            person_id = decision['person_id']
            for track_id in violation['track_ids']:
                # Find UNKNOWN IDs for these tracks
                # This would need the original detection data
                logger.info(f"Track {track_id} -> {person_id}")
                
        elif action == 'assign_new_ids':
            # Each track gets a new ID
            track_mapping = decision.get('track_mapping', {})
            for track_id, new_id in track_mapping.items():
                logger.info(f"Track {track_id} -> {new_id}")
                
        # ... handle other actions
    
    # Save decisions
    decisions_file = os.path.join(output_dir, 'violation_decisions.json')
    with open(decisions_file, 'w') as f:
        json.dump({
            'decisions': decisions,
            'timestamp': str(Path(violations_report.get('timestamp', '')))
        }, f, indent=2)
    
    logger.info(f"\n[OK] Decisions saved to {decisions_file}")
    logger.info("Next step: Re-run processing with these decisions to apply final IDs")


def main():
    parser = argparse.ArgumentParser(description='Review violations from video processing')
    parser.add_argument('violations_file', help='Path to violations JSON file')
    parser.add_argument('--output-dir', default='.', help='Output directory for decisions')
    parser.add_argument('--auto', action='store_true', help='Auto-resolve using recommendations')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.violations_file):
        logger.error(f"Violations file not found: {args.violations_file}")
        return 1
    
    # Load violations
    violations_report = load_violations_report(args.violations_file)
    
    if args.auto:
        logger.info("Auto-resolving violations based on recommendations...")
        # Implement auto-resolution logic
        decisions = {}
    else:
        # Interactive review
        decisions = interactive_review(violations_report)
    
    # Apply decisions
    if decisions:
        apply_decisions(violations_report, decisions, args.output_dir)
    else:
        logger.info("No decisions made.")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())