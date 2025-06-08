#!/usr/bin/env python3
"""
Simple Model Refinement Script - Works reliably without complex dependencies
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
import pickle

from hr_management.processing.person_recognition_trainer import PersonRecognitionTrainer
from hr_management.processing.person_dataset_creator_simple import PersonDatasetCreatorSimple


def refine_model(model_path: str, refinement_type: str = 'quick'):
    """Refine an existing model"""
    
    print(f"Starting {refinement_type} refinement for model: {model_path}")
    
    # Load existing model metadata
    metadata_path = os.path.join(model_path, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise ValueError(f"Model metadata not found at {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    current_accuracy = metadata.get('test_score', 0)
    print(f"Current model accuracy: {current_accuracy:.1%}")
    
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
        # Try to find any dataset
        datasets_dir = Path('datasets/person_recognition')
        if datasets_dir.exists():
            for d in datasets_dir.iterdir():
                if d.is_dir() and (d / 'dataset_info.json').exists():
                    dataset_name = d.name
                    break
    
    if not dataset_name:
        raise ValueError("No dataset found. Please create a dataset first.")
    
    print(f"Using dataset: {dataset_name}")
    
    # Initialize trainer and set dataset name
    trainer = PersonRecognitionTrainer()
    trainer.current_dataset_name = dataset_name
    
    # Load training data
    creator = PersonDatasetCreatorSimple()
    X, y, person_ids = creator.prepare_training_data(dataset_name)
    
    if len(X) == 0:
        raise ValueError("No training data available")
    
    print(f"Loaded {len(X)} samples for {len(person_ids)} persons")
    
    # Determine model type and parameters
    if refinement_type == 'quick':
        model_type = 'svm'
        target_accuracy = current_accuracy + 0.05  # Try to improve by 5%
        max_iterations = 1
    elif refinement_type == 'standard':
        model_type = 'svm'
        target_accuracy = 0.9
        max_iterations = 3
    elif refinement_type == 'random_forest':
        model_type = 'random_forest'
        target_accuracy = 0.9
        max_iterations = 1
    elif refinement_type == 'mlp':
        model_type = 'mlp'
        target_accuracy = 0.9
        max_iterations = 1
    else:
        model_type = 'svm'
        target_accuracy = 0.9
        max_iterations = 1
    
    # Train new model
    print(f"Training {model_type} model...")
    
    # Generate unique model name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"refined_{model_type}_{timestamp}"
    
    # For quick refinement, we need to manually handle the training and save full metadata
    if refinement_type == 'quick':
        # Split data properly
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import confusion_matrix, classification_report
        import joblib
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model with specific parameters for quick improvement
        if model_type == 'svm':
            from sklearn.svm import SVC
            model = SVC(kernel='rbf', probability=True, C=10.0, gamma='scale')
        elif model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
        elif model_type == 'mlp':
            from sklearn.neural_network import MLPClassifier
            model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42)
        else:
            from sklearn.svm import SVC
            model = SVC(kernel='rbf', probability=True, C=10.0, gamma='scale')
        
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_pred, y_test)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        
        # Generate confusion matrix and classification report
        cm = confusion_matrix(y_test, y_pred)
        
        # Create classification report with proper structure
        report_dict = classification_report(y_test, y_pred, target_names=[person_ids[i] for i in np.unique(y_test)], output_dict=True)
        
        # Save model and full metadata
        model_dir = Path('models/person_recognition') / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(model, model_dir / 'model.pkl')
        joblib.dump(scaler, model_dir / 'scaler.pkl')
        
        # Save person ID mapping
        person_id_mapping = {i: person_id for i, person_id in enumerate(person_ids)}
        with open(model_dir / 'person_id_mapping.pkl', 'wb') as f:
            import pickle
            pickle.dump(person_id_mapping, f)
        
        # Create full metadata
        metadata = {
            'model_name': model_name,
            'model_type': model_type,
            'created_at': datetime.now().isoformat(),
            'person_ids': person_ids,
            'num_persons': len(person_ids),
            'num_samples': len(X),
            'num_train_samples': len(X_train),
            'num_test_samples': len(X_test),
            'train_score': float(model.score(X_train_scaled, y_train)),
            'test_score': float(test_accuracy),
            'target_accuracy': target_accuracy,
            'cv_scores': cv_scores.tolist(),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'confusion_matrix': cm.tolist(),
            'classification_report': report_dict,
            'original_model': os.path.basename(model_path),
            'improvement': float(improvement),
            'refinement_type': refinement_type
        }
        
        with open(model_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update results to return metadata
        results = metadata
    else:
        # Use trainer for other types
        results = trainer.train_model(
            X, y, person_ids,
            model_type=model_type,
            model_name=model_name,
            target_accuracy=target_accuracy,
            max_iterations=max_iterations,
            validate_each_person=True
        )
    
    new_accuracy = results.get('test_score', 0)
    improvement = new_accuracy - current_accuracy
    
    print(f"\nRefinement Results:")
    print(f"Original accuracy: {current_accuracy:.1%}")
    print(f"New accuracy: {new_accuracy:.1%}")
    print(f"Improvement: {improvement:+.1%}")
    
    if improvement > 0:
        print(f"\nModel saved to: models/person_recognition/{model_name}")
        return f"models/person_recognition/{model_name}", new_accuracy
    else:
        print(f"\nNo improvement achieved.")
        return None, new_accuracy


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple model refinement')
    parser.add_argument('model_path', type=str, help='Path to model directory')
    parser.add_argument('--type', type=str, default='quick',
                      choices=['quick', 'standard', 'random_forest', 'mlp'],
                      help='Refinement type')
    
    args = parser.parse_args()
    
    try:
        new_model_path, accuracy = refine_model(args.model_path, args.type)
        if new_model_path:
            print(f"\nRefinement successful! New model: {new_model_path}")
        else:
            print(f"\nRefinement completed but no improvement was achieved.")
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()