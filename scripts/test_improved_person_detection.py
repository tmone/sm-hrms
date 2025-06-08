#!/usr/bin/env python3
"""
Test the improved person detection system that prioritizes recognition
"""

import sys
import os
import json
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processing.person_id_manager import get_person_id_manager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_person_id_manager():
    """Test the PersonIDManager functionality"""
    logger.info("Testing PersonIDManager...")
    
    # Get the manager instance
    manager = get_person_id_manager()
    
    # Show current mappings
    logger.info(f"Current mappings: {len(manager.get_all_mappings())} persons")
    for recognized_name, person_id in manager.get_all_mappings().items():
        logger.info(f"  {recognized_name} -> {person_id}")
        
    # Test getting existing person ID
    test_names = ["John Doe", "Jane Smith", "Unknown Person"]
    
    for name in test_names:
        person_id = manager.get_or_create_person_id(name)
        logger.info(f"Person ID for '{name}': {person_id}")
        
    # Test getting same person again
    for name in test_names[:2]:
        person_id = manager.get_or_create_person_id(name)
        logger.info(f"Person ID for '{name}' (second time): {person_id}")
        
    # Save mappings
    manager.save_mappings()
    logger.info("Mappings saved")
    
    # Show final state
    logger.info(f"Final mappings: {len(manager.get_all_mappings())} persons")
    logger.info(f"Next person ID will be: PERSON-{manager.next_person_id:04d}")


def test_recognition_model_integration():
    """Test integration with recognition model"""
    logger.info("\nTesting recognition model integration...")
    
    # Check if default model exists
    config_path = Path('models/person_recognition/config.json')
    if not config_path.exists():
        logger.warning("No recognition model config found")
        return
        
    with open(config_path) as f:
        config = json.load(f)
        default_model = config.get('default_model')
        
    if not default_model:
        logger.warning("No default model configured")
        return
        
    logger.info(f"Default model: {default_model}")
    
    # Check persons.json
    model_dir = Path('models/person_recognition') / default_model
    persons_file = model_dir / 'persons.json'
    
    if persons_file.exists():
        with open(persons_file) as f:
            persons_data = json.load(f)
            
        logger.info(f"Model has {len(persons_data)} persons:")
        for person_name, person_info in persons_data.items():
            person_id = person_info.get('person_id', 'Not assigned')
            logger.info(f"  {person_name}: {person_id}")
    else:
        logger.warning("No persons.json found in model")


def test_shared_state_manager():
    """Test the improved shared state manager"""
    logger.info("\nTesting ImprovedSharedStateManagerV3...")
    
    try:
        from processing.shared_state_manager_improved import ImprovedSharedStateManagerV3
        
        manager = ImprovedSharedStateManagerV3()
        
        # Simulate some detections
        logger.info("Simulating detections...")
        
        # Detection 1: Recognized person
        temp_id1 = manager.assign_temporary_id(
            recognized_id="John Doe",
            track_id=1,
            chunk_idx=0,
            frame_num=100,
            bbox=(100, 100, 200, 200),
            confidence=0.95,
            timestamp=3.33
        )
        logger.info(f"Assigned temporary ID for recognized 'John Doe': {temp_id1}")
        
        # Detection 2: Same recognized person
        temp_id2 = manager.assign_temporary_id(
            recognized_id="John Doe",
            track_id=2,
            chunk_idx=1,
            frame_num=200,
            bbox=(150, 150, 250, 250),
            confidence=0.92,
            timestamp=6.66
        )
        logger.info(f"Assigned temporary ID for recognized 'John Doe' (2nd): {temp_id2}")
        
        # Detection 3: Unrecognized person
        temp_id3 = manager.assign_temporary_id(
            recognized_id=None,
            track_id=3,
            chunk_idx=0,
            frame_num=150,
            bbox=(300, 300, 400, 400),
            confidence=0.88,
            timestamp=5.0
        )
        logger.info(f"Assigned temporary ID for unrecognized person: {temp_id3}")
        
        # Finalize IDs
        logger.info("\nFinalizing person IDs...")
        final_mappings = manager.finalize_person_ids()
        
        for temp_id, final_id in final_mappings.items():
            logger.info(f"  {temp_id} -> {final_id}")
            
        # Show statistics
        stats = manager.get_statistics()
        logger.info(f"\nStatistics: {json.dumps(stats, indent=2)}")
        
    except Exception as e:
        logger.error(f"Error testing shared state manager: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_person_id_manager()
    test_recognition_model_integration()
    test_shared_state_manager()