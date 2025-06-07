"""
Person Recognition Trainer
Trains models to recognize specific persons from face embeddings
"""

import os
import json
import pickle
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from typing import Dict, List, Tuple, Optional

# Import compatibility layer for NumPy version issues
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from processing.recognition_compatibility import load_compatible_pickle
except ImportError:
    # Fallback to regular pickle if compatibility layer not available
    def load_compatible_pickle(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

class PersonRecognitionTrainer:
    def __init__(self, models_dir: str = "models/person_recognition"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Available model architectures
        self.model_architectures = {
            'svm': lambda: SVC(kernel='rbf', probability=True, C=1.0, gamma='scale'),
            'svm_linear': lambda: SVC(kernel='linear', probability=True, C=1.0),
            'random_forest': lambda: RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42
            ),
            'mlp': lambda: MLPClassifier(
                hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42
            )
        }
    
    def train_model(self, X: np.ndarray, y: np.ndarray, person_ids: List[str],
                   model_type: str = 'svm', model_name: str = None,
                   target_accuracy: float = 0.9, max_iterations: int = 10,
                   validate_each_person: bool = True) -> Dict:
        """
        Train a person recognition model with continuous training until target accuracy
        
        Args:
            X: Face embeddings array
            y: Person ID labels array
            person_ids: List of unique person IDs
            model_type: Type of model to train
            model_name: Name for the saved model
            target_accuracy: Target accuracy to achieve (default: 0.9)
            max_iterations: Maximum training iterations (default: 10)
            validate_each_person: Whether to validate each person's accuracy (default: True)
            
        Returns:
            Training results dictionary
        """
        if model_type not in self.model_architectures:
            raise ValueError(f"Unknown model type: {model_type}")
        
        if model_name is None:
            model_name = f"person_model_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Check if we have pre-split validation data
        # If X and y are from training set, we need to load validation separately
        X_train, X_test, y_train, y_test = X, None, y, None
        
        # Try to load validation data if available
        try:
            # Import the dataset creator to check for validation data
            from .person_dataset_creator_simple import PersonDatasetCreatorSimple
            creator = PersonDatasetCreatorSimple()
            
            # Get dataset name from model name if possible
            if hasattr(self, 'current_dataset_name'):
                X_val, y_val, _ = creator.prepare_training_data(self.current_dataset_name, use_validation=True)
                if len(X_val) > 0:
                    X_test = X_val
                    y_test = y_val
                    print(f"Using pre-split validation data: {len(X_test)} samples")
        except:
            pass
        
        # If no pre-split validation, use train_test_split
        if X_test is None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Continuous training loop
        best_model = None
        best_test_score = 0
        iteration_results = []
        
        for iteration in range(max_iterations):
            # Create and train model
            model = self.model_architectures[model_type]()
            
            print(f"\nTraining iteration {iteration + 1}/{max_iterations} - {model_type} model with {len(X_train)} samples...")
            
            # For neural networks, increase iterations progressively
            if model_type == 'mlp' and iteration > 0:
                model.max_iter = 1000 + (iteration * 500)
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            # Get predictions for per-person validation
            y_pred_test = model.predict(X_test_scaled)
            
            # Calculate per-person accuracy if requested
            person_accuracies = {}
            if validate_each_person:
                for i, person_id in enumerate(person_ids):
                    # Find indices for this person in test set
                    person_indices = np.where(y_test == i)[0]
                    if len(person_indices) > 0:
                        person_correct = np.sum(y_pred_test[person_indices] == y_test[person_indices])
                        person_accuracy = person_correct / len(person_indices)
                        person_accuracies[person_id] = {
                            'accuracy': float(person_accuracy),
                            'num_samples': len(person_indices),
                            'correct': int(person_correct)
                        }
                        print(f"   {person_id}: {person_accuracy:.3f} ({person_correct}/{len(person_indices)})")
            
            # Store iteration results
            iteration_result = {
                'iteration': iteration + 1,
                'train_score': float(train_score),
                'test_score': float(test_score),
                'person_accuracies': person_accuracies
            }
            iteration_results.append(iteration_result)
            
            print(f"   Overall - Train: {train_score:.3f}, Test: {test_score:.3f}")
            
            # Check if we've reached target accuracy
            if test_score >= target_accuracy:
                print(f"Target accuracy {target_accuracy} reached!")
                best_model = model
                best_test_score = test_score
                break
            
            # Update best model if this is better
            if test_score > best_test_score:
                best_model = model
                best_test_score = test_score
        
        # Use the best model found
        if best_model is None:
            best_model = model  # Use last model if none met criteria
        model = best_model
        test_score = best_test_score
        
        print(f"\nFinal test accuracy: {test_score:.3f}")
        
        # Re-calculate final metrics with best model
        y_pred = best_model.predict(X_test_scaled)
        train_score = best_model.score(X_train_scaled, y_train)
        
        # Cross-validation
        # Adjust cv folds based on minimum class size
        min_class_size = min(np.bincount(y_train))
        cv_folds = min(5, min_class_size) if min_class_size > 1 else 2
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds)
        
        # Get predictions for detailed metrics
        y_pred = model.predict(X_test_scaled)
        
        # Classification report
        # Only use labels that actually appear in the test set
        unique_test_labels = np.unique(y_test)
        unique_pred_labels = np.unique(y_pred)
        all_labels = np.unique(np.concatenate([unique_test_labels, unique_pred_labels]))
        
        # Filter target names to only include classes that exist
        filtered_target_names = [person_ids[i] for i in all_labels if i < len(person_ids)]
        
        report_dict = classification_report(
            y_test, y_pred, 
            labels=all_labels,
            target_names=filtered_target_names,
            output_dict=True,
            zero_division=0
        )
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(i) for i in obj]
            return obj
        
        report = convert_numpy_types(report_dict)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=all_labels)
        
        # Save model and related data
        model_dir = self.models_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        # Save model
        joblib.dump(model, model_dir / 'model.pkl')
        joblib.dump(scaler, model_dir / 'scaler.pkl')
        
        # Save metadata
        # Get actual person_ids that have data
        actual_person_ids = [person_ids[i] for i in sorted(all_labels) if i < len(person_ids)]
        
        # Calculate final per-person accuracies
        final_person_accuracies = {}
        if validate_each_person:
            for i, person_id in enumerate(person_ids):
                person_indices = np.where(y_test == i)[0]
                if len(person_indices) > 0:
                    person_correct = np.sum(y_pred[person_indices] == y_test[person_indices])
                    person_accuracy = person_correct / len(person_indices)
                    final_person_accuracies[person_id] = {
                        'accuracy': float(person_accuracy),
                        'num_samples': len(person_indices),
                        'correct': int(person_correct)
                    }
        
        metadata = {
            'model_name': model_name,
            'model_type': model_type,
            'created_at': datetime.now().isoformat(),
            'person_ids': actual_person_ids,  # Only persons with actual training data
            'all_person_ids': person_ids,  # All persons in the dataset
            'num_persons': len(actual_person_ids),
            'num_persons_total': len(person_ids),
            'num_samples': len(X),
            'num_train_samples': len(X_train),
            'num_test_samples': len(X_test),
            'train_score': float(train_score),
            'test_score': float(test_score),
            'target_accuracy': float(target_accuracy),
            'target_reached': bool(test_score >= target_accuracy),
            'training_iterations': len(iteration_results),
            'iteration_results': convert_numpy_types(iteration_results),
            'final_person_accuracies': convert_numpy_types(final_person_accuracies),
            'cv_scores': cv_scores.tolist(),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
        
        with open(model_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save person ID mapping
        person_id_mapping = {i: person_id for i, person_id in enumerate(person_ids)}
        with open(model_dir / 'person_id_mapping.pkl', 'wb') as f:
            pickle.dump(person_id_mapping, f)
        
        print(f"Model trained successfully!")
        print(f"   Train accuracy: {train_score:.3f}")
        print(f"   Test accuracy: {test_score:.3f}")
        print(f"   CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return metadata
    
    def load_model(self, model_name: str) -> Tuple:
        """Load a trained model and its metadata"""
        model_dir = self.models_dir / model_name
        
        if not model_dir.exists():
            raise ValueError(f"Model not found: {model_name}")
        
        # Load model and scaler
        try:
            model = joblib.load(model_dir / 'model.pkl')
            scaler = joblib.load(model_dir / 'scaler.pkl')
        except Exception as e:
            # Try compatibility loader
            model = load_compatible_pickle(model_dir / 'model.pkl')
            scaler = load_compatible_pickle(model_dir / 'scaler.pkl')
        
        # Load metadata
        with open(model_dir / 'metadata.json') as f:
            metadata = json.load(f)
        
        # Load person ID mapping
        try:
            with open(model_dir / 'person_id_mapping.pkl', 'rb') as f:
                person_id_mapping = pickle.load(f)
        except Exception as e:
            person_id_mapping = load_compatible_pickle(model_dir / 'person_id_mapping.pkl')
        
        return model, scaler, metadata, person_id_mapping
    
    def predict_person(self, face_embedding: np.ndarray, model_name: str,
                      threshold: float = 0.6) -> Dict:
        """
        Predict person from face embedding
        
        Args:
            face_embedding: Face encoding vector
            model_name: Name of the model to use
            threshold: Confidence threshold for prediction
            
        Returns:
            Prediction results
        """
        model, scaler, metadata, person_id_mapping = self.load_model(model_name)
        
        # Prepare embedding
        embedding = np.array(face_embedding).reshape(1, -1)
        embedding_scaled = scaler.transform(embedding)
        
        # Get prediction and probabilities
        prediction = model.predict(embedding_scaled)[0]
        probabilities = model.predict_proba(embedding_scaled)[0]
        
        # Get confidence for predicted class
        confidence = probabilities[prediction]
        
        # Check threshold
        if confidence < threshold:
            return {
                'person_id': 'unknown',
                'confidence': float(confidence),
                'all_probabilities': {}
            }
        
        # Get person ID
        person_id = person_id_mapping[prediction]
        
        # Get all probabilities
        all_probs = {
            person_id_mapping[i]: float(prob) 
            for i, prob in enumerate(probabilities)
        }
        
        return {
            'person_id': person_id,
            'confidence': float(confidence),
            'all_probabilities': all_probs
        }
    
    def evaluate_on_dataset(self, model_name: str, X_test: np.ndarray, 
                          y_test: np.ndarray) -> Dict:
        """Evaluate model on a test dataset"""
        model, scaler, metadata, person_id_mapping = self.load_model(model_name)
        
        # Scale features
        X_test_scaled = scaler.transform(X_test)
        
        # Get predictions
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)
        
        # Calculate metrics
        accuracy = model.score(X_test_scaled, y_test)
        
        # Get confidence scores
        confidences = [y_proba[i, pred] for i, pred in enumerate(y_pred)]
        
        # Classification report
        report = classification_report(
            y_test, y_pred,
            target_names=metadata['person_ids'],
            output_dict=True
        )
        
        return {
            'accuracy': float(accuracy),
            'avg_confidence': float(np.mean(confidences)),
            'min_confidence': float(np.min(confidences)),
            'max_confidence': float(np.max(confidences)),
            'classification_report': report,
            'num_samples': len(X_test)
        }
    
    def get_available_models(self) -> List[Dict]:
        """Get list of available trained models"""
        models = []
        
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir() and (model_dir / 'metadata.json').exists():
                try:
                    with open(model_dir / 'metadata.json') as f:
                        metadata = json.load(f)
                    
                    models.append({
                        'name': model_dir.name,
                        'type': metadata.get('model_type', 'unknown'),
                        'created_at': metadata.get('created_at', 'unknown'),
                        'num_persons': metadata.get('num_persons', len(metadata.get('person_ids', []))),
                        'test_accuracy': metadata.get('test_score', 0),
                        'cv_accuracy': metadata.get('cv_mean', 0),
                        'test_score': metadata.get('test_score', 0)  # For compatibility
                    })
                except (json.JSONDecodeError, Exception) as e:
                    print(f"Warning: Skipping corrupted model {model_dir.name}: {e}")
                    continue
        
        return sorted(models, key=lambda x: x['created_at'], reverse=True)