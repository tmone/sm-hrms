"""
Person Dataset Creator (Simple Version)
Creates training datasets from reviewed person detections without face_recognition
"""

import os
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import pickle

class PersonDatasetCreatorSimple:
    def __init__(self, persons_dir: str = "processing/outputs/persons", 
                 dataset_dir: str = "datasets/person_recognition"):
        self.persons_dir = Path(persons_dir)
        self.dataset_dir = Path(dataset_dir)
        
    def create_dataset_from_persons(self, person_ids: List[str], dataset_name: str) -> Dict:
        """
        Create a training dataset from selected persons
        Uses existing person images without face extraction
        """
        # Create dataset directory
        dataset_path = self.dataset_dir / dataset_name
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (dataset_path / "images").mkdir(exist_ok=True)
        (dataset_path / "features").mkdir(exist_ok=True)
        
        dataset_info = {
            'name': dataset_name,
            'created_at': datetime.now().isoformat(),
            'persons': {},
            'total_images': 0,
            'total_train_images': 0,
            'total_val_images': 0,
            'total_features': 0,
            'total_train_features': 0,
            'total_val_features': 0,
            'total_faces': 0,  # For compatibility with the template
            'split_ratio': '80-20 (with augmentation)',
            'augmentation_factor': 4
        }
        
        # Process each person
        for person_id in person_ids:
            person_results = self._process_person(person_id, dataset_path)
            if person_results['success']:
                dataset_info['persons'][person_id] = person_results
                dataset_info['total_images'] += person_results['images_count']
                dataset_info['total_train_images'] += person_results['train_images_count']
                dataset_info['total_val_images'] += person_results['val_images_count']
                dataset_info['total_features'] += person_results['features_count']
                dataset_info['total_train_features'] += person_results['train_features_count']
                dataset_info['total_val_features'] += person_results['val_features_count']
                dataset_info['total_faces'] += person_results['features_count']  # Same as features for simple version
        
        # Save dataset info
        with open(dataset_path / 'dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=2)
            
        return dataset_info
    
    def _process_person(self, person_id: str, dataset_path: Path) -> Dict:
        """Process a single person's images for the dataset with train/val split"""
        person_dir = self.persons_dir / person_id
        if not person_dir.exists():
            return {'success': False, 'error': f'Person directory not found: {person_id}'}
        
        # Load person metadata
        metadata_path = person_dir / 'metadata.json'
        if not metadata_path.exists():
            return {'success': False, 'error': f'Metadata not found for: {person_id}'}
            
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # Create person subdirectories for train and validation
        person_train_images_dir = dataset_path / "images_train" / person_id
        person_val_images_dir = dataset_path / "images_val" / person_id
        person_train_features_dir = dataset_path / "features_train" / person_id
        person_val_features_dir = dataset_path / "features_val" / person_id
        
        person_train_images_dir.mkdir(parents=True, exist_ok=True)
        person_val_images_dir.mkdir(parents=True, exist_ok=True)
        person_train_features_dir.mkdir(parents=True, exist_ok=True)
        person_val_features_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'success': True,
            'person_id': person_id,
            'images_count': 0,
            'train_images_count': 0,
            'val_images_count': 0,
            'features_count': 0,
            'train_features_count': 0,
            'val_features_count': 0,
            'faces_count': 0,  # For compatibility
            'embeddings_count': 0,  # For compatibility
            'features': []
        }
        
        # Get all images and shuffle for random split
        all_images = metadata.get('images', [])
        np.random.seed(42)  # For reproducible splits
        np.random.shuffle(all_images)
        
        # Split 50-50 for train and validation
        split_idx = len(all_images) // 2
        train_images = all_images[:split_idx]
        val_images = all_images[split_idx:]
        
        print(f"Processing {person_id}: {len(all_images)} images (train: {len(train_images)}, val: {len(val_images)})")
        
        # Process training images
        for img_data in train_images:
            img_path = person_dir / img_data['filename']
            if not img_path.exists():
                print(f"  ⚠️  Image not found: {img_path}")
                continue
                
            # Copy original image to train directory
            dest_img_path = person_train_images_dir / img_data['filename']
            shutil.copy2(img_path, dest_img_path)
            results['train_images_count'] += 1
            results['images_count'] += 1
            
            # Extract simple features
            features = self._extract_simple_features(str(img_path), person_id, img_data['filename'])
            if features is not None:
                # Save features to train directory
                feature_filename = f"{person_id}_features_{img_data['filename'].split('.')[0]}.pkl"
                feature_path = person_train_features_dir / feature_filename
                
                feature_data = {
                    'person_id': person_id,
                    'source_image': img_data['filename'],
                    'features': features,
                    'bbox': img_data.get('bbox', None),
                    'confidence': img_data.get('confidence', 0.9),
                    'split': 'train'
                }
                
                with open(feature_path, 'wb') as f:
                    pickle.dump(feature_data, f)
                
                results['train_features_count'] += 1
                results['features_count'] += 1
                results['faces_count'] += 1
                results['embeddings_count'] += 1
                results['features'].append(feature_filename)
                
            # Create augmented versions (4 additional copies for 80-20 final ratio)
            for aug_idx in range(4):
                aug_img = cv2.imread(str(img_path))
                if aug_img is None:
                    continue
                    
                aug_img = self._augment_image(aug_img)
                aug_filename = f"{img_data['filename'].split('.')[0]}_aug{aug_idx}.jpg"
                aug_path = person_train_images_dir / aug_filename
                cv2.imwrite(str(aug_path), aug_img)
                
                # Extract features from augmented image
                aug_features = self._extract_simple_features(str(aug_path), person_id, aug_filename)
                if aug_features is not None:
                    aug_feature_filename = f"{person_id}_features_{aug_filename.split('.')[0]}.pkl"
                    aug_feature_path = person_train_features_dir / aug_feature_filename
                    
                    aug_feature_data = {
                        'person_id': person_id,
                        'source_image': aug_filename,
                        'features': aug_features,
                        'bbox': img_data.get('bbox', None),
                        'confidence': img_data.get('confidence', 0.9),
                        'split': 'train',
                        'augmented': True
                    }
                    
                    with open(aug_feature_path, 'wb') as f:
                        pickle.dump(aug_feature_data, f)
                    
                    results['train_features_count'] += 1
                    results['train_images_count'] += 1
        
        # Process validation images (no augmentation)
        for img_data in val_images:
            img_path = person_dir / img_data['filename']
            if not img_path.exists():
                print(f"  ⚠️  Image not found: {img_path}")
                continue
                
            # Copy original image to validation directory
            dest_img_path = person_val_images_dir / img_data['filename']
            shutil.copy2(img_path, dest_img_path)
            results['val_images_count'] += 1
            results['images_count'] += 1
            
            # Extract simple features
            features = self._extract_simple_features(str(img_path), person_id, img_data['filename'])
            if features is not None:
                # Save features to validation directory
                feature_filename = f"{person_id}_features_{img_data['filename'].split('.')[0]}.pkl"
                feature_path = person_val_features_dir / feature_filename
                
                feature_data = {
                    'person_id': person_id,
                    'source_image': img_data['filename'],
                    'features': features,
                    'bbox': img_data.get('bbox', None),
                    'confidence': img_data.get('confidence', 0.9),
                    'split': 'val'
                }
                
                with open(feature_path, 'wb') as f:
                    pickle.dump(feature_data, f)
                
                results['val_features_count'] += 1
                results['features_count'] += 1
                results['faces_count'] += 1
                results['embeddings_count'] += 1
                results['features'].append(feature_filename)
        
        return results
    
    def _extract_simple_features(self, image_path: str, person_id: str, filename: str) -> Optional[np.ndarray]:
        """Extract simple features from an image"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Resize to standard size
            standard_size = (128, 256)  # width, height for person detection
            resized = cv2.resize(image, standard_size)
            
            # Extract color histogram
            hist_features = []
            for i in range(3):  # BGR channels
                hist = cv2.calcHist([resized], [i], None, [32], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                hist_features.extend(hist)
            
            # Extract HOG features (simplified)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            
            # Simple gradient features
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Compute magnitude and angle
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            angle = np.arctan2(sobely, sobelx)
            
            # Create simple HOG-like features
            cell_size = 16
            n_cells_x = standard_size[0] // cell_size
            n_cells_y = standard_size[1] // cell_size
            n_bins = 9
            
            hog_features = []
            for y in range(n_cells_y):
                for x in range(n_cells_x):
                    # Get cell
                    cell_mag = magnitude[y*cell_size:(y+1)*cell_size, x*cell_size:(x+1)*cell_size]
                    cell_ang = angle[y*cell_size:(y+1)*cell_size, x*cell_size:(x+1)*cell_size]
                    
                    # Compute histogram
                    hist, _ = np.histogram(cell_ang.ravel(), bins=n_bins, range=(-np.pi, np.pi), weights=cell_mag.ravel())
                    hog_features.extend(hist)
            
            # Combine all features
            all_features = np.concatenate([hist_features, hog_features])
            
            # Normalize
            all_features = all_features / (np.linalg.norm(all_features) + 1e-6)
            
            return all_features
            
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None
    
    def prepare_training_data(self, dataset_name: str, use_validation: bool = False) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare training data from dataset
        
        Args:
            dataset_name: Name of the dataset
            use_validation: If True, load validation data; if False, load training data
        
        Returns:
            X: Feature vectors array
            y: Person ID labels array
            person_ids: List of unique person IDs
        """
        dataset_path = self.dataset_dir / dataset_name
        
        # Choose appropriate directory based on split
        if use_validation:
            features_dir = dataset_path / "features_val"
        else:
            features_dir = dataset_path / "features_train"
            
        # Also check for old format (backward compatibility)
        if not features_dir.exists():
            features_dir = dataset_path / "features"
        
        X = []
        y = []
        person_ids = []
        person_id_map = {}
        person_sample_count = defaultdict(int)
        
        # First pass: count samples per person
        for person_dir in features_dir.iterdir():
            if not person_dir.is_dir():
                continue
            
            person_id = person_dir.name
            feature_files = list(person_dir.glob('*.pkl'))
            person_sample_count[person_id] = len(feature_files)
        
        # Second pass: only include persons with at least 1 sample
        for person_dir in features_dir.iterdir():
            if not person_dir.is_dir():
                continue
                
            person_id = person_dir.name
            if person_sample_count[person_id] == 0:
                print(f"⚠️  Skipping {person_id} - no valid features")
                continue
                
            if person_id not in person_id_map:
                person_id_map[person_id] = len(person_ids)
                person_ids.append(person_id)
            
            person_label = person_id_map[person_id]
            
            # Load person's features
            for feature_file in person_dir.glob('*.pkl'):
                try:
                    with open(feature_file, 'rb') as f:
                        feature_data = pickle.load(f)
                        if 'features' in feature_data and feature_data['features'] is not None:
                            X.append(feature_data['features'])
                            y.append(person_label)
                except Exception as e:
                    print(f"⚠️  Error loading {feature_file}: {e}")
                    continue
        
        split_type = "validation" if use_validation else "training"
        print(f"📊 Loaded {len(X)} {split_type} samples for {len(person_ids)} persons")
        for i, person_id in enumerate(person_ids):
            count = sum(1 for label in y if label == i)
            print(f"   {person_id}: {count} samples")
        
        # Check which classes are actually present
        unique_labels = np.unique(y) if len(y) > 0 else []
        if len(unique_labels) < len(person_ids):
            missing_classes = set(range(len(person_ids))) - set(unique_labels)
            for missing in missing_classes:
                print(f"⚠️  WARNING: {person_ids[missing]} has no valid features!")
        
        return np.array(X), np.array(y), person_ids
    
    def augment_dataset(self, dataset_name: str, augmentation_factor: int = 3) -> Dict:
        """
        Augment dataset with transformed versions of images
        Since we're not using face detection, we'll augment the full person images
        """
        dataset_path = self.dataset_dir / dataset_name
        if not dataset_path.exists():
            return {'success': False, 'error': 'Dataset not found'}
        
        images_dir = dataset_path / "images"
        augmented_dir = dataset_path / "images_augmented"
        augmented_dir.mkdir(exist_ok=True)
        
        augmented_count = 0
        
        # Process each person's images
        for person_dir in images_dir.iterdir():
            if not person_dir.is_dir():
                continue
                
            person_id = person_dir.name
            person_aug_dir = augmented_dir / person_id
            person_aug_dir.mkdir(exist_ok=True)
            
            # Process each image
            for img_path in person_dir.glob('*.jpg'):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Generate augmented versions
                for i in range(augmentation_factor):
                    aug_img = self._augment_image(img)
                    aug_filename = f"{img_path.stem}_aug_{i}.jpg"
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
            aug_img = np.clip(aug_img.astype(np.int16) + value, 0, 255).astype(np.uint8)
        
        # Random contrast adjustment
        if np.random.random() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)
            aug_img = np.clip(alpha * aug_img, 0, 255).astype(np.uint8)
        
        # Random horizontal flip
        if np.random.random() > 0.5:
            aug_img = cv2.flip(aug_img, 1)
        
        # Random rotation (small angles)
        if np.random.random() > 0.5:
            angle = np.random.uniform(-10, 10)
            h, w = aug_img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            aug_img = cv2.warpAffine(aug_img, M, (w, h))
        
        # Random noise
        if np.random.random() > 0.3:
            noise = np.random.normal(0, 5, aug_img.shape).astype(np.uint8)
            aug_img = cv2.add(aug_img, noise)
        
        return aug_img