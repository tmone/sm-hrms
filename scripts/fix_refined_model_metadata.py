#!/usr/bin/env python3
"""
Fix metadata for existing refined models to include all visualization data
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
import joblib

from hr_management.processing.person_dataset_creator_simple import PersonDatasetCreatorSimple


def fix_model_metadata(model_path):
    """Add missing metadata to a refined model"""
    
    print(f"Fixing metadata for: {model_path}")
    
    # Load existing metadata
    metadata_path = os.path.join(model_path, 'metadata.json')
    if not os.path.exists(metadata_path):
        print(f"No metadata found at {metadata_path}")
        return False
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Check if already has full metadata
    if 'confusion_matrix' in metadata and 'classification_report' in metadata:
        print("Model already has complete metadata")
        return True
    
    print("Missing visualization data, regenerating...")
    
    # Load model and scaler
    try:
        model = joblib.load(os.path.join(model_path, 'model.pkl'))
        scaler = joblib.load(os.path.join(model_path, 'scaler.pkl'))
    except:
        print("Could not load model files")
        return False
    
    # Load dataset
    creator = PersonDatasetCreatorSimple()
    
    # Try to find dataset
    config_path = Path('datasets/person_recognition/config.json')
    dataset_name = None
    
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            dataset_name = config.get('default_dataset')
    
    if not dataset_name:
        datasets_dir = Path('datasets/person_recognition')
        if datasets_dir.exists():
            for d in datasets_dir.iterdir():
                if d.is_dir() and (d / 'dataset_info.json').exists():
                    dataset_name = d.name
                    break
    
    if not dataset_name:
        print("No dataset found")
        return False
    
    print(f"Using dataset: {dataset_name}")
    X, y, person_ids = creator.prepare_training_data(dataset_name)
    
    if len(X) == 0:
        print("No data available")
        return False
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale and predict
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Get predictions
    y_pred = model.predict(X_test_scaled)
    
    # Generate missing metrics
    cm = confusion_matrix(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    
    # Generate classification report
    report_dict = classification_report(
        y_test, y_pred,
        target_names=[person_ids[i] for i in np.unique(y_test)],
        output_dict=True
    )
    
    # Update metadata
    metadata.update({
        'num_persons': metadata.get('num_persons', len(person_ids)),
        'num_samples': len(X),
        'num_train_samples': len(X_train),
        'num_test_samples': len(X_test),
        'train_score': float(model.score(X_train_scaled, y_train)),
        'cv_scores': cv_scores.tolist(),
        'cv_mean': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'confusion_matrix': cm.tolist(),
        'classification_report': report_dict
    })
    
    # Save updated metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Metadata updated successfully!")
    return True


def main():
    """Fix all refined models"""
    models_dir = Path('models/person_recognition')
    
    if not models_dir.exists():
        print("No models directory found")
        return
    
    refined_models = []
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir() and 'refined' in model_dir.name:
            refined_models.append(model_dir)
    
    print(f"Found {len(refined_models)} refined models to check")
    
    for model_dir in refined_models:
        fix_model_metadata(str(model_dir))
        print()


if __name__ == '__main__':
    main()