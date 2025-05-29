#!/usr/bin/env python3
"""
Enhanced Person Recognition Model Refinement Script

This script provides advanced functionality to refine and retrain person recognition models
with hyperparameter tuning, data quality analysis, and incremental learning support.
"""

import os
import sys
import json
import pickle
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import joblib

from hr_management.processing.person_recognition_trainer import PersonRecognitionTrainer
from hr_management.processing.person_dataset_creator_simple import PersonDatasetCreatorSimple
from hr_management.processing.advanced_feature_extractor import AdvancedFeatureExtractor

# Import Flask app and models
from app import create_app, db
from models.video import PersonDetection, Person


class ModelRefiner:
    """Enhanced model refinement class with advanced training capabilities"""
    
    def __init__(self, existing_model_path: Optional[str] = None):
        self.trainer = PersonRecognitionTrainer()
        self.dataset_creator = PersonDatasetCreatorSimple()
        self.advanced_extractor = AdvancedFeatureExtractor()
        self.existing_model_path = existing_model_path
        self.model_data = None
        
        if existing_model_path and os.path.exists(existing_model_path):
            self.load_existing_model(existing_model_path)
    
    def load_existing_model(self, model_path: str):
        """Load existing model and its metadata"""
        metadata_path = os.path.join(model_path, 'metadata.json')
        model_file = os.path.join(model_path, 'model.pkl')
        scaler_file = os.path.join(model_path, 'scaler.pkl')
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.model_data = json.load(f)
            print(f"Loaded model metadata: {self.model_data['model_type']} with accuracy {self.model_data['test_accuracy']:.2%}")
        
        if os.path.exists(model_file):
            self.existing_model = joblib.load(model_file)
        
        if os.path.exists(scaler_file):
            self.existing_scaler = joblib.load(scaler_file)
    
    def analyze_data_quality(self, X: np.ndarray, y: np.ndarray, person_ids: List[str]) -> Dict[str, Any]:
        """Analyze dataset quality and distribution"""
        unique_classes, class_counts = np.unique(y, return_counts=True)
        
        # Find classes with insufficient samples
        min_samples = 5
        low_sample_classes = []
        for cls, count in zip(unique_classes, class_counts):
            if count < min_samples:
                person_id = person_ids[cls] if cls < len(person_ids) else f"Unknown_{cls}"
                low_sample_classes.append((person_id, count))
        
        # Calculate feature statistics
        feature_means = np.mean(X, axis=0)
        feature_stds = np.std(X, axis=0)
        low_variance_features = np.where(feature_stds < 0.01)[0]
        
        # Check for class imbalance
        imbalance_ratio = max(class_counts) / min(class_counts)
        
        analysis = {
            'total_samples': len(X),
            'num_classes': len(unique_classes),
            'samples_per_class': dict(zip(unique_classes, class_counts)),
            'low_sample_classes': low_sample_classes,
            'imbalance_ratio': imbalance_ratio,
            'low_variance_features': len(low_variance_features),
            'feature_dim': X.shape[1]
        }
        
        return analysis
    
    def augment_low_sample_classes(self, X: np.ndarray, y: np.ndarray, min_samples: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Augment classes with low sample counts"""
        unique_classes, class_counts = np.unique(y, return_counts=True)
        X_augmented = X.copy()
        y_augmented = y.copy()
        
        for cls, count in zip(unique_classes, class_counts):
            if count < min_samples:
                # Get samples for this class
                class_samples = X[y == cls]
                samples_needed = min_samples - count
                
                # Create augmented samples
                for _ in range(samples_needed):
                    # Random sample with noise
                    idx = np.random.randint(0, len(class_samples))
                    sample = class_samples[idx].copy()
                    
                    # Add small random noise
                    noise = np.random.normal(0, 0.01, sample.shape)
                    augmented_sample = sample + noise
                    
                    X_augmented = np.vstack([X_augmented, augmented_sample.reshape(1, -1)])
                    y_augmented = np.append(y_augmented, cls)
        
        return X_augmented, y_augmented
    
    def select_best_features(self, X: np.ndarray, y: np.ndarray, n_features: Optional[int] = None) -> Tuple[np.ndarray, Any]:
        """Feature selection using statistical tests"""
        if n_features is None:
            n_features = min(X.shape[1], 100)  # Default to top 100 features
        
        selector = SelectKBest(f_classif, k=n_features)
        X_selected = selector.fit_transform(X, y)
        
        return X_selected, selector
    
    def apply_pca(self, X: np.ndarray, variance_ratio: float = 0.95) -> Tuple[np.ndarray, Any]:
        """Apply PCA for dimensionality reduction"""
        pca = PCA(n_components=variance_ratio)
        X_pca = pca.fit_transform(X)
        
        print(f"PCA reduced features from {X.shape[1]} to {X_pca.shape[1]} (explaining {variance_ratio:.0%} variance)")
        return X_pca, pca
    
    def get_hyperparameter_grid(self, model_type: str) -> Dict[str, List]:
        """Get hyperparameter grid for different model types"""
        grids = {
            'svm': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'linear', 'poly']
            },
            'svm_linear': {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'mlp': {
                'hidden_layer_sizes': [(100,), (128, 64), (256, 128, 64)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive'],
                'max_iter': [500, 1000, 2000]
            }
        }
        
        return grids.get(model_type, grids['svm'])
    
    def train_with_hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray, 
                                        model_type: str = 'svm', cv_folds: int = 5) -> Tuple[Any, Dict]:
        """Train model with hyperparameter tuning"""
        # Get base model
        if model_type == 'svm':
            base_model = SVC(probability=True, class_weight='balanced')
        elif model_type == 'svm_linear':
            base_model = SVC(kernel='linear', probability=True, class_weight='balanced')
        elif model_type == 'random_forest':
            base_model = RandomForestClassifier(class_weight='balanced', random_state=42)
        elif model_type == 'mlp':
            base_model = MLPClassifier(random_state=42, early_stopping=True)
        else:
            base_model = SVC(probability=True, class_weight='balanced')
        
        # Get hyperparameter grid
        param_grid = self.get_hyperparameter_grid(model_type)
        
        # Adjust cv_folds based on smallest class size
        min_class_size = min(np.bincount(y))
        cv_folds = min(cv_folds, min_class_size)
        
        print(f"Starting hyperparameter tuning for {model_type} with {cv_folds}-fold CV...")
        
        # Grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv_folds,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        print(f"Best parameters: {best_params}")
        print(f"Best CV score: {best_score:.3f}")
        
        return best_model, {
            'best_params': best_params,
            'best_cv_score': best_score,
            'cv_results': grid_search.cv_results_
        }
    
    def extract_advanced_features(self, person_detections: List) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Extract advanced features from person detections"""
        X = []
        y = []
        person_ids = []
        person_id_map = {}
        
        for detection in person_detections:
            if detection.person_id and detection.person_image:
                # Get person index
                if detection.person_id not in person_id_map:
                    person_id_map[detection.person_id] = len(person_id_map)
                    person_ids.append(detection.person_id)
                
                person_idx = person_id_map[detection.person_id]
                
                # Decode image
                img_array = np.frombuffer(detection.person_image, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if img is not None:
                    # Extract advanced features
                    features = self.advanced_extractor.extract_all_features(img)
                    X.append(features)
                    y.append(person_idx)
        
        return np.array(X), np.array(y), person_ids
    
    def refine_model(self, model_type: str = 'svm', 
                    use_feature_selection: bool = True,
                    use_pca: bool = False,
                    augment_data: bool = True,
                    hyperparameter_tuning: bool = True,
                    use_advanced_features: bool = False) -> Dict[str, Any]:
        """Main refinement function"""
        print("Starting model refinement process...")
        
        # Handle data loading based on feature type
        if use_advanced_features:
            print("Extracting advanced features from detections...")
            # Create Flask app context for database access
            app = create_app()
            with app.app_context():
                # Get all person detections
                detections = PersonDetection.query.filter(
                    PersonDetection.person_id.isnot(None),
                    PersonDetection.person_image.isnot(None)
                ).all()
                
                if not detections:
                    raise ValueError("No person detections available for training")
            
            # Extract features outside of app context
            X, y, person_ids = self.extract_advanced_features(detections)
        else:
            # Load dataset
            print("Loading dataset...")
            X, y, person_ids = self.dataset_creator.load_dataset()
            
            if X is None or len(X) == 0:
                print("Creating dataset from detections...")
                X, y, person_ids = self.dataset_creator.create_dataset()
                
                if X is None or len(X) == 0:
                    raise ValueError("No data available for training")
        
        # Analyze data quality
        print("\nAnalyzing data quality...")
        data_analysis = self.analyze_data_quality(X, y, person_ids)
        print(f"Dataset: {data_analysis['total_samples']} samples, {data_analysis['num_classes']} classes")
        print(f"Class imbalance ratio: {data_analysis['imbalance_ratio']:.2f}")
        
        if data_analysis['low_sample_classes']:
            print(f"Warning: {len(data_analysis['low_sample_classes'])} classes have < 5 samples")
        
        # Data augmentation
        if augment_data and data_analysis['low_sample_classes']:
            print("\nAugmenting low-sample classes...")
            X, y = self.augment_low_sample_classes(X, y)
            print(f"Dataset size after augmentation: {len(X)} samples")
        
        # Feature engineering
        transformers = []
        
        if use_feature_selection:
            print("\nApplying feature selection...")
            X, selector = self.select_best_features(X, y)
            transformers.append(('feature_selection', selector))
        
        if use_pca:
            print("\nApplying PCA...")
            X, pca = self.apply_pca(X)
            transformers.append(('pca', pca))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        if hyperparameter_tuning:
            model, tuning_results = self.train_with_hyperparameter_tuning(
                X_train_scaled, y_train, model_type
            )
        else:
            # Use default parameters
            if model_type == 'svm':
                model = SVC(kernel='rbf', probability=True, C=1.0, gamma='scale')
            elif model_type == 'random_forest':
                model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            elif model_type == 'mlp':
                model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42)
            else:
                model = SVC(kernel='rbf', probability=True, C=1.0, gamma='scale')
            
            model.fit(X_train_scaled, y_train)
            tuning_results = None
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        cv_score = cv_scores.mean()
        
        print(f"\nRefined Model Performance:")
        print(f"Test Accuracy: {test_accuracy:.3f}")
        print(f"CV Score: {cv_score:.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Save refined model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"refined_{model_type}_{timestamp}"
        
        results = self.trainer.save_model(
            model=model,
            scaler=scaler,
            person_ids=person_ids,
            model_name=model_name,
            model_type=model_type,
            test_accuracy=test_accuracy,
            cv_score=cv_score,
            training_params={
                'use_feature_selection': use_feature_selection,
                'use_pca': use_pca,
                'augment_data': augment_data,
                'hyperparameter_tuning': hyperparameter_tuning,
                'use_advanced_features': use_advanced_features,
                'transformers': [(name, type(trans).__name__) for name, trans in transformers],
                'best_params': tuning_results['best_params'] if tuning_results else None
            }
        )
        
        # Save transformers
        if transformers:
            transformer_path = os.path.join(results['model_path'], 'transformers.pkl')
            joblib.dump(transformers, transformer_path)
        
        # Generate detailed report
        report = classification_report(y_test, y_pred, target_names=[person_ids[i] for i in np.unique(y_test)])
        report_path = os.path.join(results['model_path'], 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\nModel saved to: {results['model_path']}")
        print("\nClassification Report:")
        print(report)
        
        return {
            'model_path': results['model_path'],
            'test_accuracy': test_accuracy,
            'cv_score': cv_score,
            'data_analysis': data_analysis,
            'tuning_results': tuning_results,
            'improvement': test_accuracy - 0.746 if self.existing_model_path else 0  # Compare to current 74.6%
        }


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Refine person recognition model')
    parser.add_argument('--model-type', type=str, default='svm',
                      choices=['svm', 'svm_linear', 'random_forest', 'mlp'],
                      help='Type of model to train')
    parser.add_argument('--existing-model', type=str,
                      help='Path to existing model to refine')
    parser.add_argument('--no-feature-selection', action='store_true',
                      help='Disable feature selection')
    parser.add_argument('--use-pca', action='store_true',
                      help='Use PCA for dimensionality reduction')
    parser.add_argument('--no-augmentation', action='store_true',
                      help='Disable data augmentation')
    parser.add_argument('--no-tuning', action='store_true',
                      help='Disable hyperparameter tuning')
    parser.add_argument('--advanced-features', action='store_true',
                      help='Use advanced feature extraction methods')
    parser.add_argument('--update-default', action='store_true',
                      help='Update default model configuration')
    
    args = parser.parse_args()
    
    # Initialize refiner
    refiner = ModelRefiner(args.existing_model)
    
    # Run refinement
    results = refiner.refine_model(
        model_type=args.model_type,
        use_feature_selection=not args.no_feature_selection,
        use_pca=args.use_pca,
        augment_data=not args.no_augmentation,
        hyperparameter_tuning=not args.no_tuning,
        use_advanced_features=args.advanced_features
    )
    
    print(f"\nRefinement complete!")
    print(f"Accuracy improvement: {results['improvement']:.1%}")
    
    # Update default model if requested
    if args.update_default and results['test_accuracy'] > 0.746:  # Only update if better
        config_path = 'models/person_recognition/config.json'
        config = {
            'default_model': os.path.basename(results['model_path']),
            'model_type': args.model_type,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\nDefault model updated to: {config['default_model']}")


if __name__ == '__main__':
    main()