#!/usr/bin/env python3
"""Test UNKNOWN to PERSON ID conversion"""

import os
import sys
import logging

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from processing.chunked_video_processor import SharedStateManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_basic_conversion():
    """Test basic UNKNOWN to PERSON conversion"""
    logger.info("Testing basic UNKNOWN to PERSON conversion...")
    
    manager = SharedStateManager()
    
    # Simulate detections from multiple chunks
    # Chunk 0: Person A (recognized as "john") appears in frames 0-100
    for frame in range(0, 100, 10):
        unknown_id = manager.assign_temporary_id("john", track_id=1, 
                                               chunk_idx=0, frame_num=frame)
    
    # Chunk 1: Person A (recognized as "john") appears again in frames 900-1000
    for frame in range(900, 1000, 10):
        unknown_id = manager.assign_temporary_id("john", track_id=15, 
                                               chunk_idx=1, frame_num=frame)
    
    # Chunk 0: Person B (not recognized) appears in frames 200-300
    for frame in range(200, 300, 10):
        unknown_id = manager.assign_temporary_id(None, track_id=2, 
                                               chunk_idx=0, frame_num=frame)
    
    # Chunk 2: Person C (recognized as "mary") appears in frames 1500-1600
    for frame in range(1500, 1600, 10):
        unknown_id = manager.assign_temporary_id("mary", track_id=25, 
                                               chunk_idx=2, frame_num=frame)
    
    logger.info(f"Created {manager.unknown_counter} UNKNOWN IDs")
    logger.info(f"Track to UNKNOWN mapping: {manager.track_to_unknown_id}")
    
    # Finalize person IDs
    mapping = manager.finalize_person_ids()
    
    logger.info(f"\nFinal mapping ({len(mapping)} entries):")
    for unknown_id, person_id in sorted(mapping.items()):
        logger.info(f"  {unknown_id} -> {person_id}")
    
    # Verify results
    # All "john" detections should map to same PERSON ID
    john_person_ids = set()
    for unknown_id, person_id in mapping.items():
        if unknown_id in ["UNKNOWN-0001", "UNKNOWN-0002"]:  # john's tracks
            john_person_ids.add(person_id)
    
    assert len(john_person_ids) == 1, f"John should have 1 person ID, got {len(john_person_ids)}"
    logger.info(f"[OK] All 'john' detections correctly mapped to {john_person_ids.pop()}")
    
    # Unrecognized person should have unique ID
    unrecognized_id = mapping.get("UNKNOWN-0003")  # Person B's track
    assert unrecognized_id is not None
    assert unrecognized_id.startswith("PERSON-")
    logger.info(f"[OK] Unrecognized person correctly mapped to {unrecognized_id}")
    
    # Mary should have unique ID
    mary_id = mapping.get("UNKNOWN-0004")  # Mary's track
    assert mary_id is not None
    assert mary_id.startswith("PERSON-")
    assert mary_id != unrecognized_id
    logger.info(f"[OK] Mary correctly mapped to {mary_id}")
    
    return True


def test_multiple_tracks_same_person():
    """Test multiple tracks for same recognized person"""
    logger.info("\nTesting multiple tracks for same recognized person...")
    
    manager = SharedStateManager()
    
    # Simulate person appearing and disappearing multiple times
    # Track 1: frames 0-50
    for frame in range(0, 50, 10):
        manager.assign_temporary_id("employee_001", track_id=1, 
                                  chunk_idx=0, frame_num=frame)
    
    # Track 5: frames 200-250 (same person reappears)
    for frame in range(200, 250, 10):
        manager.assign_temporary_id("employee_001", track_id=5, 
                                  chunk_idx=0, frame_num=frame)
    
    # Track 12: frames 500-550 (same person again)
    for frame in range(500, 550, 10):
        manager.assign_temporary_id("employee_001", track_id=12, 
                                  chunk_idx=1, frame_num=frame)
    
    # Different person
    for frame in range(100, 150, 10):
        manager.assign_temporary_id("employee_002", track_id=3, 
                                  chunk_idx=0, frame_num=frame)
    
    # Finalize
    mapping = manager.finalize_person_ids()
    
    # All employee_001 tracks should map to same PERSON ID
    emp1_ids = set()
    emp2_ids = set()
    
    for unknown_id, person_id in mapping.items():
        info = manager.unknown_recognition_info.get(unknown_id, {})
        if info.get('recognized_id') == 'employee_001':
            emp1_ids.add(person_id)
        elif info.get('recognized_id') == 'employee_002':
            emp2_ids.add(person_id)
    
    assert len(emp1_ids) == 1, f"Employee 001 should have 1 ID, got {len(emp1_ids)}"
    assert len(emp2_ids) == 1, f"Employee 002 should have 1 ID, got {len(emp2_ids)}"
    assert emp1_ids != emp2_ids, "Different employees should have different IDs"
    
    logger.info(f"[OK] Employee 001 (3 tracks) -> {emp1_ids.pop()}")
    logger.info(f"[OK] Employee 002 (1 track) -> {emp2_ids.pop()}")
    
    return True


def test_no_duplicates_in_frame():
    """Test that no duplicate PERSON IDs appear in same frame"""
    logger.info("\nTesting no duplicate PERSON IDs in same frame...")
    
    manager = SharedStateManager()
    
    # Simulate multiple people in same frame
    frame_num = 100
    
    # Person 1 at position A
    id1 = manager.assign_temporary_id("alice", track_id=1, 
                                    chunk_idx=0, frame_num=frame_num)
    
    # Person 2 at position B (same frame)
    id2 = manager.assign_temporary_id("bob", track_id=2, 
                                    chunk_idx=0, frame_num=frame_num)
    
    # Person 3 at position C (same frame, unrecognized)
    id3 = manager.assign_temporary_id(None, track_id=3, 
                                    chunk_idx=0, frame_num=frame_num)
    
    # Person 4 at position D (same frame, another alice - shouldn't happen but test it)
    id4 = manager.assign_temporary_id("alice", track_id=4, 
                                    chunk_idx=0, frame_num=frame_num)
    
    logger.info(f"Frame {frame_num} temporary IDs: {[id1, id2, id3, id4]}")
    
    # Finalize
    mapping = manager.finalize_person_ids()
    
    # Get final person IDs for this frame
    person_ids = [mapping[uid] for uid in [id1, id2, id3, id4]]
    
    logger.info(f"Frame {frame_num} final IDs: {person_ids}")
    
    # Check uniqueness (except for recognized duplicates)
    # id1 and id4 are both "alice" so they should have same PERSON ID
    assert mapping[id1] == mapping[id4], "Same recognized person should have same ID"
    assert mapping[id2] != mapping[id1], "Different people should have different IDs"
    assert mapping[id3] != mapping[id1], "Unrecognized person should have unique ID"
    assert mapping[id3] != mapping[id2], "Unrecognized person should have unique ID"
    
    logger.info(f"[OK] Alice appears twice: {mapping[id1]} (consistent)")
    logger.info(f"[OK] Bob has unique ID: {mapping[id2]}")
    logger.info(f"[OK] Unknown has unique ID: {mapping[id3]}")
    
    return True


def main():
    """Run all tests"""
    tests = [
        test_basic_conversion,
        test_multiple_tracks_same_person,
        test_no_duplicates_in_frame
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
                logger.info(f"[OK] {test_func.__name__} PASSED\n")
            else:
                failed += 1
                logger.error(f"[ERROR] {test_func.__name__} FAILED\n")
        except Exception as e:
            failed += 1
            logger.error(f"[ERROR] {test_func.__name__} FAILED with exception: {e}\n")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Test Results: {passed}/{len(tests)} passed")
    logger.info(f"{'='*60}")
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())