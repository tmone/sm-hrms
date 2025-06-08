#!/usr/bin/env python3
"""
Advanced Model Refinement Script - Actually improves model performance
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import joblib

from hr_management.processing.person_recognition_trainer import PersonRecognitionTrainer
from hr_management.processing.person_dataset_creator_simple import PersonDatasetCreatorSimple


class AdvancedModelRefiner:
    """Advanced refinement with multiple improvement strategies"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.load_existing_model()
        
    def load_existing_model(self):
        """Load existing model and metadata"""
        metadata_path = os.path.join(self.model_path, 'metadata.json')
        if not os.path.exists(metadata_path):
            raise ValueError(f"Model metadata not found at {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.current_accuracy = self.metadata.get('test_score', 0)
        print(f"Current model accuracy: {self.current_accuracy:.1%}")
        
    def load_data(self):
        """Load and prepare training data"""
        # Find dataset
        config_path = Path('datasets/person_recognition/config.json')
        dataset_name = None
        
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                    dataset_name = config.get('default_dataset')
            except:
                pass
        
        if not dataset_name:
            datasets_dir = Path('datasets/person_recognition')
            if datasets_dir.exists():
                for d in datasets_dir.iterdir():
                    if d.is_dir() and (d / 'dataset_info.json').exists():
                        dataset_name = d.name
                        break
        
        if not dataset_name:
            raise ValueError("No dataset found")
        
        print(f"Using dataset: {dataset_name}")
        
        # Load data
        creator = PersonDatasetCreatorSimple()
        X, y, person_ids = creator.prepare_training_data(dataset_name)
        
        # Also load validation data
        X_val, y_val, _ = creator.prepare_training_data(dataset_name, use_validation=True)
        
        self.X = np.array(X)
        self.y = np.array(y)
        self.X_val = np.array(X_val) if len(X_val) > 0 else None
        self.y_val = np.array(y_val) if len(y_val) > 0 else None
        self.person_ids = person_ids
        
        print(f"Loaded {len(self.X)} training samples and {len(self.X_val) if self.X_val is not None else 0} validation samples")
        
    def augment_training_data(self):
        """Augment training data for minority classes"""
        from collections import Counter
        from imblearn.over_sampling import SMOTE, RandomOverSampler
        
        # Check class distribution
        class_counts = Counter(self.y)
        print("Class distribution:", dict(class_counts))
        
        # If imbalanced, use SMOTE or RandomOverSampler
        min_samples = min(class_counts.values())
        max_samples = max(class_counts.values())
        
        if max_samples / min_samples > 2:  # Significant imbalance
            print("Addressing class imbalance...")
            
            # Use SMOTE if we have enough samples, otherwise RandomOverSampler
            if min_samples >= 6:
                sampler = SMOTE(random_state=42, k_neighbors=min(5, min_samples-1))
            else:
                sampler = RandomOverSampler(random_state=42)
            
            self.X, self.y = sampler.fit_resample(self.X, self.y)
            print(f"After balancing: {len(self.X)} samples")
    
    def engineer_features(self):
        """Apply advanced feature engineering"""
        print("Engineering features...")
        
        # 1. Add polynomial features for top features
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.feature_selection import SelectKBest, f_classif
        
        # Select top features
        selector = SelectKBest(f_classif, k=min(50, self.X.shape[1]//2))
        top_features = selector.fit_transform(self.X, self.y)
        
        # Add polynomial features
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        poly_features = poly.fit_transform(top_features)
        
        # Combine with original features
        self.X = np.hstack([self.X, poly_features[:, top_features.shape[1]:]])  # Only add new poly features
        
        print(f"Feature dimension after engineering: {self.X.shape[1]}")
    
    def optimize_hyperparameters(self, model_type='svm'):
        """Extensive hyperparameter optimization"""
        print(f"Optimizing hyperparameters for {model_type}...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define parameter grids
        param_grids = {
            'svm': {
                'C': [0.01, 0.1, 1, 10, 100, 1000],
                'gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'poly', 'sigmoid'],
                'class_weight': ['balanced', None]
            },
            'random_forest': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
                'class_weight': ['balanced', 'balanced_subsample', None]
            },
            'mlp': {
                'hidden_layer_sizes': [(100,), (200,), (100, 50), (200, 100), (200, 100, 50)],
                'activation': ['tanh', 'relu'],
                'solver': ['adam', 'lbfgs'],
                'alpha': [0.0001, 0.001, 0.01, 0.1],
                'learning_rate': ['constant', 'adaptive'],
                'max_iter': [1000, 2000]
            },
            'gradient_boost': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'subsample': [0.8, 0.9, 1.0]
            }
        }
        
        # Select model
        if model_type == 'svm':
            base_model = SVC(probability=True)
            param_grid = param_grids['svm']
        elif model_type == 'random_forest':
            base_model = RandomForestClassifier(random_state=42)
            param_grid = param_grids['random_forest']
        elif model_type == 'mlp':
            base_model = MLPClassifier(random_state=42, early_stopping=True)
            param_grid = param_grids['mlp']
        elif model_type == 'gradient_boost':
            base_model = GradientBoostingClassifier(random_state=42)
            param_grid = param_grids['gradient_boost']
        else:
            base_model = SVC(probability=True)
            param_grid = param_grids['svm']
        
        # Use RandomizedSearchCV for faster optimization
        n_iter = 50 if model_type in ['svm', 'mlp'] else 30
        
        search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=n_iter,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='accuracy',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
        
        search.fit(X_train_scaled, y_train)
        
        # Get best model
        best_model = search.best_estimator_
        best_params = search.best_params_
        
        # Evaluate
        y_pred = best_model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Best parameters: {best_params}")
        print(f"Optimized accuracy: {test_accuracy:.3f}")
        
        return best_model, scaler, test_accuracy, X_train_scaled, y_train, X_test_scaled, y_test
    
    def create_ensemble(self):
        """Create an ensemble of different models"""
        print("Creating ensemble model...")
        
        # Train multiple models
        models = []
        
        # SVM with optimized parameters
        svm_model, _, svm_acc, X_train, y_train, X_test, y_test = self.optimize_hyperparameters('svm')
        models.append(('svm', svm_model))
        
        # Random Forest
        rf_model, _, rf_acc, _, _, _, _ = self.optimize_hyperparameters('random_forest')
        models.append(('rf', rf_model))
        
        # Gradient Boosting
        gb_model, _, gb_acc, _, _, _, _ = self.optimize_hyperparameters('gradient_boost')
        models.append(('gb', gb_model))
        
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=models,
            voting='soft',
            n_jobs=-1
        )
        
        # Train ensemble
        ensemble.fit(X_train, y_train)
        
        # Evaluate
        y_pred = ensemble.predict(X_test)
        ensemble_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Individual model accuracies: SVM={svm_acc:.3f}, RF={rf_acc:.3f}, GB={gb_acc:.3f}")
        print(f"Ensemble accuracy: {ensemble_accuracy:.3f}")
        
        return ensemble, ensemble_accuracy
    
    def refine(self, refinement_type='standard'):
        """Main refinement method"""
        print(f"\nStarting {refinement_type} refinement...")
        
        # Load data
        self.load_data()
        
        best_model = None
        best_scaler = None
        best_accuracy = self.current_accuracy
        
        if refinement_type == 'quick':
            # Quick optimization with current model type
            model_type = self.metadata.get('model_type', 'svm')
            model, scaler, accuracy, _, _, _, _ = self.optimize_hyperparameters(model_type)
            if accuracy > best_accuracy:
                best_model = model
                best_scaler = scaler
                best_accuracy = accuracy
                
        elif refinement_type == 'standard':
            # Try data augmentation and feature engineering
            self.augment_training_data()
            self.engineer_features()
            
            # Optimize hyperparameters
            model, scaler, accuracy, _, _, _, _ = self.optimize_hyperparameters('svm')
            if accuracy > best_accuracy:
                best_model = model
                best_scaler = scaler
                best_accuracy = accuracy
                
        elif refinement_type == 'advanced':
            # Full pipeline with ensemble
            self.augment_training_data()
            self.engineer_features()
            
            # Try ensemble
            ensemble, ensemble_acc = self.create_ensemble()
            if ensemble_acc > best_accuracy:
                # For ensemble, we need to recreate scaler
                scaler = StandardScaler()
                scaler.fit(self.X)
                best_model = ensemble
                best_scaler = scaler
                best_accuracy = ensemble_acc
                
        elif refinement_type in ['random_forest', 'mlp', 'gradient_boost']:
            # Specific model type
            self.augment_training_data()
            model, scaler, accuracy, _, _, _, _ = self.optimize_hyperparameters(refinement_type)
            if accuracy > best_accuracy:
                best_model = model
                best_scaler = scaler
                best_accuracy = accuracy
        
        # Check improvement
        improvement = best_accuracy - self.current_accuracy
        print(f"\nResults:")
        print(f"Original accuracy: {self.current_accuracy:.1%}")
        print(f"New accuracy: {best_accuracy:.1%}")
        print(f"Improvement: {improvement:+.1%}")
        
        if improvement > 0 and best_model is not None:
            # Save improved model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"refined_{refinement_type}_{timestamp}"
            
            # Use trainer to save properly
            trainer = PersonRecognitionTrainer()
            trainer.current_dataset_name = 'default'
            
            # We need to retrain with trainer to get proper metadata
            # But use the optimized parameters
            if hasattr(best_model, 'get_params'):
                params = best_model.get_params()
                print(f"Saving model with optimized parameters: {params}")
            
            # Save manually since we have custom model
            model_dir = Path('models/person_recognition') / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            
            joblib.dump(best_model, model_dir / 'model.pkl')
            joblib.dump(best_scaler, model_dir / 'scaler.pkl')
            
            # Evaluate the model properly to get all metrics
            from sklearn.metrics import confusion_matrix, classification_report
            from sklearn.model_selection import cross_val_score
            
            # Get final predictions for confusion matrix
            X_scaled = best_scaler.fit_transform(self.X)
            y_pred = best_model.predict(X_scaled)
            
            # Split for proper evaluation
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
            )
            X_test_scaled = best_scaler.transform(X_test)
            y_test_pred = best_model.predict(X_test_scaled)
            
            # Generate metrics
            cm = confusion_matrix(y_test, y_test_pred)
            cv_scores = cross_val_score(best_model, X_scaled, self.y, cv=5)
            
            # Generate classification report
            report_dict = classification_report(
                y_test, y_test_pred, 
                target_names=[self.person_ids[i] for i in np.unique(y_test)], 
                output_dict=True
            )
            
            # Create complete metadata
            metadata = {
                'model_name': model_name,
                'model_type': refinement_type,
                'created_at': datetime.now().isoformat(),
                'person_ids': self.person_ids,
                'num_persons': len(self.person_ids),
                'num_samples': len(self.X),
                'num_train_samples': len(X_train),
                'num_test_samples': len(X_test),
                'train_score': float(best_model.score(best_scaler.transform(X_train), y_train)),
                'test_score': float(best_accuracy),
                'cv_scores': cv_scores.tolist(),
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std()),
                'confusion_matrix': cm.tolist(),
                'classification_report': report_dict,
                'original_model': os.path.basename(self.model_path),
                'improvement': float(improvement),
                'refinement_type': refinement_type,
                'target_accuracy': 0.9
            }
            
            # Save person ID mapping
            person_id_mapping = {i: person_id for i, person_id in enumerate(self.person_ids)}
            with open(model_dir / 'person_id_mapping.pkl', 'wb') as f:
                import pickle
                pickle.dump(person_id_mapping, f)
            
            with open(model_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"\nModel saved to: {model_dir}")
            return str(model_dir), best_accuracy
        else:
            print("\nNo improvement achieved with current settings.")
            return None, best_accuracy


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced model refinement')
    parser.add_argument('model_path', type=str, help='Path to model directory')
    parser.add_argument('--type', type=str, default='standard',
                      choices=['quick', 'standard', 'advanced', 'random_forest', 'mlp', 'gradient_boost'],
                      help='Refinement type')
    
    args = parser.parse_args()
    
    try:
        refiner = AdvancedModelRefiner(args.model_path)
        new_model_path, accuracy = refiner.refine(args.type)
        
        if new_model_path:
            print(f"\nRefinement successful! New model: {new_model_path}")
        else:
            print(f"\nRefinement completed but no improvement was achieved.")
            print("Consider trying:")
            print("- Different refinement type (advanced, gradient_boost)")
            print("- Creating a larger dataset with more person images")
            print("- Ensuring good quality person detections")
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()