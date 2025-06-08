"""
Centralized Person ID Manager
Manages the mapping between recognized persons and their PERSON IDs
"""
import json
import logging
from pathlib import Path
from typing import Dict, Optional
from threading import Lock

logger = logging.getLogger(__name__)


class PersonIDManager:
    """Manages person ID assignments and mappings"""
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.recognized_to_person_id = {}  # recognized_name -> PERSON-XXXX
        self.person_id_to_recognized = {}  # PERSON-XXXX -> recognized_name
        self.next_person_id = 1
        
        # Load existing mappings
        self._load_mappings()
        
    def _load_mappings(self):
        """Load existing person mappings from various sources"""
        
        # 1. Load from recognition model's persons.json
        try:
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
                                
                                for person_name, person_info in persons_data.items():
                                    # Handle both person_name as key and person_id in info
                                    if isinstance(person_info, dict) and 'person_id' in person_info:
                                        person_id = person_info['person_id']
                                        # Map both ways for flexibility
                                        self.recognized_to_person_id[person_name] = person_id
                                        self.person_id_to_recognized[person_id] = person_name
                                        # Also map person_id to itself for when model returns person_id
                                        self.recognized_to_person_id[person_id] = person_id
                                        
                                        # Extract numeric ID for counter
                                        try:
                                            num_id = int(person_id.replace('PERSON-', ''))
                                            self.next_person_id = max(self.next_person_id, num_id + 1)
                                        except:
                                            pass
                                            
                                logger.info(f"Loaded {len(self.recognized_to_person_id)} person mappings from model")
                                
        except Exception as e:
            logger.warning(f"Could not load model person mappings: {e}")
            
        # 2. Scan existing PERSON folders
        try:
            persons_dir = Path('processing/outputs/persons')
            if persons_dir.exists():
                for person_folder in persons_dir.glob('PERSON-*'):
                    try:
                        person_id = person_folder.name
                        num_id = int(person_id.replace('PERSON-', ''))
                        self.next_person_id = max(self.next_person_id, num_id + 1)
                        
                        # Check if this folder has recognition info
                        metadata_file = person_folder / 'metadata.json'
                        if metadata_file.exists():
                            with open(metadata_file) as f:
                                metadata = json.load(f)
                                if 'recognized_name' in metadata:
                                    recognized_name = metadata['recognized_name']
                                    self.recognized_to_person_id[recognized_name] = person_id
                                    self.person_id_to_recognized[person_id] = recognized_name
                                    
                    except Exception as e:
                        logger.warning(f"Error processing folder {person_folder}: {e}")
                        
                logger.info(f"Next person ID will be: PERSON-{self.next_person_id:04d}")
                
        except Exception as e:
            logger.warning(f"Could not scan person folders: {e}")
            
        # 3. Load from a persistent mapping file
        mapping_file = Path('processing/outputs/person_id_mappings.json')
        if mapping_file.exists():
            try:
                with open(mapping_file) as f:
                    saved_mappings = json.load(f)
                    
                    # Merge with existing mappings
                    for recognized_name, person_id in saved_mappings.get('recognized_to_person_id', {}).items():
                        if recognized_name not in self.recognized_to_person_id:
                            self.recognized_to_person_id[recognized_name] = person_id
                            self.person_id_to_recognized[person_id] = recognized_name
                            
                    logger.info(f"Loaded additional mappings from {mapping_file}")
                    
            except Exception as e:
                logger.warning(f"Could not load mapping file: {e}")
                
    def save_mappings(self):
        """Save current mappings to persistent file"""
        mapping_file = Path('processing/outputs/person_id_mappings.json')
        mapping_file.parent.mkdir(parents=True, exist_ok=True)
        
        mappings = {
            'recognized_to_person_id': self.recognized_to_person_id,
            'person_id_to_recognized': self.person_id_to_recognized,
            'next_person_id': self.next_person_id
        }
        
        try:
            with open(mapping_file, 'w') as f:
                json.dump(mappings, f, indent=2)
            logger.info(f"Saved {len(self.recognized_to_person_id)} mappings")
        except Exception as e:
            logger.error(f"Failed to save mappings: {e}")
            
    def get_or_create_person_id(self, recognized_name: Optional[str]) -> str:
        """Get existing PERSON ID or create new one"""
        with self._lock:
            if recognized_name and recognized_name in self.recognized_to_person_id:
                # Return existing ID
                return self.recognized_to_person_id[recognized_name]
            else:
                # Create new ID
                person_id = f"PERSON-{self.next_person_id:04d}"
                self.next_person_id += 1
                
                if recognized_name:
                    # Store mapping
                    self.recognized_to_person_id[recognized_name] = person_id
                    self.person_id_to_recognized[person_id] = recognized_name
                    
                    # Save mappings periodically
                    if self.next_person_id % 10 == 0:
                        self.save_mappings()
                        
                return person_id
                
    def get_recognized_name(self, person_id: str) -> Optional[str]:
        """Get recognized name for a PERSON ID"""
        return self.person_id_to_recognized.get(person_id)
        
    def update_mapping(self, person_id: str, recognized_name: str):
        """Update mapping between PERSON ID and recognized name"""
        with self._lock:
            # Remove old mapping if exists
            old_name = self.person_id_to_recognized.get(person_id)
            if old_name and old_name in self.recognized_to_person_id:
                del self.recognized_to_person_id[old_name]
                
            # Add new mapping
            self.recognized_to_person_id[recognized_name] = person_id
            self.person_id_to_recognized[person_id] = recognized_name
            
            # Save immediately for important updates
            self.save_mappings()
            
    def get_all_mappings(self) -> Dict[str, str]:
        """Get all recognized name to PERSON ID mappings"""
        return self.recognized_to_person_id.copy()


# Global instance
_person_id_manager = None

def get_person_id_manager() -> PersonIDManager:
    """Get the global PersonIDManager instance"""
    global _person_id_manager
    if _person_id_manager is None:
        _person_id_manager = PersonIDManager()
    return _person_id_manager