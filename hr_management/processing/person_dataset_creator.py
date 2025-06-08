"""
Person Dataset Creator
Creates training datasets from reviewed person detections
"""

import os
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import face_recognition
from collections import defaultdict
import pickle

class PersonDatasetCreator:
    def __init__(self, persons_dir: str = "processing/outputs/persons", 
                 dataset_dir: str = "datasets/person_recognition"):
        self.persons_dir = Path(persons_dir)
        self.dataset_dir = Path(dataset_dir)
        self.min_face_size = 40  # Minimum face size in pixels
        self.face_padding = 0.2  # Padding around face (20%)
        
    def create_dataset_from_persons(self, person_ids: List[str], dataset_name: str) -> Dict:
        """
        Create a training dataset from selected persons
        
        Args:
            person_ids: List of person IDs to include in dataset
            dataset_name: Name for the dataset
            
        Returns:
            Dict with dataset creation results
        """
        # Create dataset directory
        dataset_path = self.dataset_dir / dataset_name
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (dataset_path / "images").mkdir(exist_ok=True)
        (dataset_path / "faces").mkdir(exist_ok=True)
        (dataset_path / "embeddings").mkdir(exist_ok=True)
        
        dataset_info = {
            'name': dataset_name,
            'created_at': datetime.now().isoformat(),
            'persons': {},
            'total_images': 0,
            'total_faces': 0,
            'total_embeddings': 0
        }
        
        # Process each person
        for person_id in person_ids:
            person_results = self._process_person(person_id, dataset_path)
            if person_results['success']:
                dataset_info['persons'][person_id] = person_results
                dataset_info['total_images'] += person_results['images_count']
                dataset_info['total_faces'] += person_results['faces_count']
                dataset_info['total_embeddings'] += person_results['embeddings_count']
        
        # Save dataset info
        with open(dataset_path / 'dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=2)
            
        return dataset_info
    
    def _process_person(self, person_id: str, dataset_path: Path) -> Dict:
        """Process a single person's images for the dataset"""
        person_dir = self.persons_dir / person_id
        if not person_dir.exists():
            return {'success': False, 'error': f'Person directory not found: {person_id}'}
        
        # Load person metadata
        metadata_path = person_dir / 'metadata.json'
        if not metadata_path.exists():
            return {'success': False, 'error': f'Metadata not found for: {person_id}'}
            
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # Create person subdirectories
        person_images_dir = dataset_path / "images" / person_id
        person_faces_dir = dataset_path / "faces" / person_id
        person_embeddings_dir = dataset_path / "embeddings" / person_id
        
        person_images_dir.mkdir(exist_ok=True)
        person_faces_dir.mkdir(exist_ok=True)
        person_embeddings_dir.mkdir(exist_ok=True)
        
        results = {
            'success': True,
            'person_id': person_id,
            'images_count': 0,
            'faces_count': 0,
            'embeddings_count': 0,
            'face_locations': [],
            'embeddings': []
        }
        
        # Process each image
        for img_data in metadata.get('images', []):
            img_path = person_dir / img_data['filename']
            if not img_path.exists():
                continue
                
            # Copy original image
            dest_img_path = person_images_dir / img_data['filename']
            shutil.copy2(img_path, dest_img_path)
            results['images_count'] += 1
            
            # Extract faces and create embeddings
            face_results = self._extract_faces_and_embeddings(
                str(img_path), 
                person_id,
                img_data['filename'],
                person_faces_dir,
                person_embeddings_dir
            )
            
            if face_results['faces_found'] > 0:
                results['faces_count'] += face_results['faces_found']
                results['embeddings_count'] += len(face_results['embeddings'])
                results['face_locations'].extend(face_results['face_locations'])
                results['embeddings'].extend(face_results['embeddings'])
        
        return results
    
    def _extract_faces_and_embeddings(self, image_path: str, person_id: str, 
                                      filename: str, faces_dir: Path, 
                                      embeddings_dir: Path) -> Dict:
        """Extract faces and create face embeddings from an image"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {'faces_found': 0, 'embeddings': [], 'face_locations': []}
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_image, model='hog')
        
        if not face_locations:
            return {'faces_found': 0, 'embeddings': [], 'face_locations': []}
        
        results = {
            'faces_found': len(face_locations),
            'embeddings': [],
            'face_locations': []
        }
        
        # Process each face
        for idx, (top, right, bottom, left) in enumerate(face_locations):
            # Add padding
            height, width = image.shape[:2]
            face_height = bottom - top
            face_width = right - left
            
            padding_h = int(face_height * self.face_padding)
            padding_w = int(face_width * self.face_padding)
            
            # Apply padding with bounds checking
            top = max(0, top - padding_h)
            bottom = min(height, bottom + padding_h)
            left = max(0, left - padding_w)
            right = min(width, right + padding_w)
            
            # Check minimum size
            if (bottom - top) < self.min_face_size or (right - left) < self.min_face_size:
                continue
            
            # Extract face
            face_img = image[top:bottom, left:right]
            
            # Save face image
            face_filename = f"{person_id}_face_{filename.split('.')[0]}_{idx}.jpg"
            face_path = faces_dir / face_filename
            cv2.imwrite(str(face_path), face_img)
            
            # Generate face embedding
            face_encoding = face_recognition.face_encodings(
                rgb_image, 
                known_face_locations=[(top, right, bottom, left)]
            )
            
            if face_encoding:
                embedding_data = {
                    'person_id': person_id,
                    'source_image': filename,
                    'face_index': idx,
                    'face_location': (top, right, bottom, left),
                    'embedding': face_encoding[0].tolist()
                }
                
                # Save embedding
                embedding_filename = f"{person_id}_embedding_{filename.split('.')[0]}_{idx}.pkl"
                embedding_path = embeddings_dir / embedding_filename
                with open(embedding_path, 'wb') as f:
                    pickle.dump(embedding_data, f)
                
                results['embeddings'].append(embedding_filename)
                results['face_locations'].append({
                    'filename': face_filename,
                    'location': (top, right, bottom, left)
                })
        
        return results
    
    def augment_dataset(self, dataset_name: str, augmentation_factor: int = 3) -> Dict:
        """
        Augment dataset with transformed versions of faces
        
        Args:
            dataset_name: Name of the dataset to augment
            augmentation_factor: How many augmented versions per original
        """
        dataset_path = self.dataset_dir / dataset_name
        if not dataset_path.exists():
            return {'success': False, 'error': 'Dataset not found'}
        
        faces_dir = dataset_path / "faces"
        augmented_dir = dataset_path / "faces_augmented"
        augmented_dir.mkdir(exist_ok=True)
        
        augmented_count = 0
        
        # Process each person's faces
        for person_dir in faces_dir.iterdir():
            if not person_dir.is_dir():
                continue
                
            person_id = person_dir.name
            person_aug_dir = augmented_dir / person_id
            person_aug_dir.mkdir(exist_ok=True)
            
            # Process each face image
            for face_path in person_dir.glob('*.jpg'):
                img = cv2.imread(str(face_path))
                if img is None:
                    continue
                
                # Generate augmented versions
                for i in range(augmentation_factor):
                    aug_img = self._augment_image(img)
                    aug_filename = f"{face_path.stem}_aug_{i}.jpg"
                    cv2.imwrite(str(person_aug_dir / aug_filename), aug_img)
                    augmented_count += 1
        
        return {
            'success': True,
            'augmented_count': augmented_count,
            'augmentation_factor': augmentation_factor
        }
    
    def _augment_image(self, image: np.ndarray) -> np.ndarray:
        """Apply random augmentations to an image"""
        aug_img = image.copy()
        
        # Random brightness adjustment
        if np.random.random() > 0.5:
            value = np.random.randint(-30, 30)
            aug_img = cv2.add(aug_img, value)
        
        # Random contrast adjustment
        if np.random.random() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)
            aug_img = cv2.convertScaleAbs(aug_img, alpha=alpha)
        
        # Random horizontal flip
        if np.random.random() > 0.5:
            aug_img = cv2.flip(aug_img, 1)
        
        # Random rotation (small angles)
        if np.random.random() > 0.5:
            angle = np.random.uniform(-10, 10)
            h, w = aug_img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            aug_img = cv2.warpAffine(aug_img, M, (w, h))
        
        return aug_img
    
    def prepare_training_data(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare training data from dataset
        
        Returns:
            X: Face encodings array
            y: Person ID labels array
            person_ids: List of unique person IDs
        """
        dataset_path = self.dataset_dir / dataset_name
        embeddings_dir = dataset_path / "embeddings"
        
        X = []
        y = []
        person_ids = []
        person_id_map = {}
        
        # Load all embeddings
        for person_dir in embeddings_dir.iterdir():
            if not person_dir.is_dir():
                continue
                
            person_id = person_dir.name
            if person_id not in person_id_map:
                person_id_map[person_id] = len(person_ids)
                person_ids.append(person_id)
            
            person_label = person_id_map[person_id]
            
            # Load person's embeddings
            for embedding_file in person_dir.glob('*.pkl'):
                with open(embedding_file, 'rb') as f:
                    embedding_data = pickle.load(f)
                    X.append(embedding_data['embedding'])
                    y.append(person_label)
        
        return np.array(X), np.array(y), person_ids