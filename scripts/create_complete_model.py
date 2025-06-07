#!/usr/bin/env python3
"""
Create a complete model with all visualization data
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hr_management.processing.person_recognition_trainer import PersonRecognitionTrainer
from hr_management.processing.person_dataset_creator_simple import PersonDatasetCreatorSimple


def create_complete_model():
    """Create a model with complete metadata for visualization"""
    
    print("Creating a complete model with all visualization data...")
    
    # Initialize trainer
    trainer = PersonRecognitionTrainer()
    dataset_creator = PersonDatasetCreatorSimple()
    
    # Use the fresh dataset
    dataset_name = 'fresh_dataset_20250529'
    trainer.current_dataset_name = dataset_name
    
    # Load data
    print(f"Loading dataset: {dataset_name}")
    X, y, person_ids = dataset_creator.prepare_training_data(dataset_name)
    
    if len(X) == 0:
        print("No training data available")
        return
    
    print(f"Loaded {len(X)} samples for {len(person_ids)} persons")
    
    # Train model with complete metadata
    model_name = f"complete_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    results = trainer.train_model(
        X, y, person_ids,
        model_type='svm',
        model_name=model_name,
        target_accuracy=0.85,
        max_iterations=5,
        validate_each_person=True
    )
    
    print(f"\nModel created successfully!")
    print(f"Model name: {results['model_name']}")
    print(f"Accuracy: {results['test_score']:.1%}")
    print(f"Location: models/person_recognition/{results['model_name']}")
    
    # Check that all visualization data is present
    model_path = Path('models/person_recognition') / results['model_name']
    metadata_path = model_path / 'metadata.json'
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    print("\nMetadata includes:")
    print(f"- Confusion matrix: {'[CHECK]' if 'confusion_matrix' in metadata else '✗'}")
    print(f"- Classification report: {'[CHECK]' if 'classification_report' in metadata else '✗'}")
    print(f"- CV scores: {'[CHECK]' if 'cv_scores' in metadata else '✗'}")
    print(f"- Iteration results: {'[CHECK]' if 'iteration_results' in metadata else '✗'}")
    print(f"- Person accuracies: {'[CHECK]' if 'final_person_accuracies' in metadata else '✗'}")
    
    return results


if __name__ == '__main__':
    create_complete_model()