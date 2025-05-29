#!/usr/bin/env python3
"""
Quick Person Recognition Model Refinement Script

A simplified version that works directly with existing datasets without database dependencies.
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
from sklearn.metrics import accuracy_score, classification_report
import joblib

from hr_management.processing.person_recognition_trainer import PersonRecognitionTrainer
from hr_management.processing.person_dataset_creator_simple import PersonDatasetCreatorSimple


def quick_refine(model_path: str, refinement_type: str = 'quick'):
    """Quick refinement without database dependencies"""
    
    print(f"Starting {refinement_type} refinement for model: {model_path}")
    
    # Load existing model metadata
    metadata_path = os.path.join(model_path, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise ValueError(f"Model metadata not found at {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"Current model accuracy: {metadata.get('test_score', 0):.1%}")
    
    # Initialize dataset creator and trainer
    dataset_creator = PersonDatasetCreatorSimple()
    trainer = PersonRecognitionTrainer()
    
    # Load dataset
    print("Loading dataset...")
    
    # Check if a default dataset exists
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
        # Try to find any existing dataset
        datasets_dir = Path('datasets/person_recognition')
        if datasets_dir.exists():
            for d in datasets_dir.iterdir():
                if d.is_dir() and (d / 'dataset_info.json').exists():
                    dataset_name = d.name
                    break
    
    if not dataset_name:
        raise ValueError("No dataset found. Please create a dataset first from the web interface.")
    
    print(f"Using dataset: {dataset_name}")
    X, y, person_ids = dataset_creator.prepare_training_data(dataset_name)
    
    if len(X) == 0:
        raise ValueError("No data available for training")
    
    print(f"Loaded {len(X)} samples for {len(person_ids)} persons")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train based on refinement type
    if refinement_type == 'quick':
        # Quick training without hyperparameter tuning
        model = SVC(kernel='rbf', probability=True, C=10.0, gamma='scale')
        print("Training SVM with optimized parameters (no grid search)...")
    elif refinement_type == 'standard':
        # Standard training with limited hyperparameter tuning
        print("Performing hyperparameter optimization...")
        param_grid = {
            'C': [1.0, 10.0, 100.0],
            'gamma': ['scale', 0.001, 0.01]
        }
        model = GridSearchCV(
            SVC(kernel='rbf', probability=True),
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1
        )
    elif refinement_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        )
        print("Training Random Forest model...")
    elif refinement_type == 'mlp':
        model = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            max_iter=2000,
            early_stopping=True,
            random_state=42
        )
        print("Training Neural Network model...")
    else:
        raise ValueError(f"Unknown refinement type: {refinement_type}")
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nRefined Model Performance:")
    print(f"Test Accuracy: {test_accuracy:.3f}")
    print(f"Improvement: {(test_accuracy - metadata.get('test_score', 0)):.1%}")
    
    # Save if improved
    if test_accuracy > metadata.get('test_score', 0):
        # Generate classification report
        report = classification_report(y_test, y_pred, target_names=[person_ids[i] for i in np.unique(y_test)])
        print("\nClassification Report:")
        print(report)
        
        # Save refined model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"refined_{refinement_type}_{timestamp}"
        
        # Extract the actual model if it's a GridSearchCV
        if hasattr(model, 'best_estimator_'):
            actual_model = model.best_estimator_
            print(f"Best parameters: {model.best_params_}")
        else:
            actual_model = model
        
        results = trainer.save_model(
            model=actual_model,
            scaler=scaler,
            person_ids=person_ids,
            model_name=model_name,
            model_type=refinement_type,
            test_accuracy=test_accuracy,
            cv_score=test_accuracy,  # Use test accuracy as approximation
            training_params={
                'refinement_type': refinement_type,
                'original_model': os.path.basename(model_path),
                'quick_mode': True
            }
        )
        
        print(f"\nModel saved to: {results['model_path']}")
        return results['model_path'], test_accuracy
    else:
        print(f"\nNo improvement achieved. Original accuracy: {metadata.get('test_score', 0):.1%}")
        return None, test_accuracy


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick model refinement')
    parser.add_argument('model_path', type=str, help='Path to model directory')
    parser.add_argument('--type', type=str, default='quick',
                      choices=['quick', 'standard', 'random_forest', 'mlp'],
                      help='Refinement type')
    
    args = parser.parse_args()
    
    try:
        new_model_path, accuracy = quick_refine(args.model_path, args.type)
        if new_model_path:
            print(f"\nRefinement successful! New model: {new_model_path}")
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()