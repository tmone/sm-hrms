#!/usr/bin/env python3
"""
Test that refined models are actually better
"""

import json
from pathlib import Path
from datetime import datetime


def test_refined_models():
    """Check all models and compare accuracies"""
    
    models_dir = Path('models/person_recognition')
    if not models_dir.exists():
        print("No models directory found")
        return
    
    models = []
    
    # Load all model metadata
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir() and (model_dir / 'metadata.json').exists():
            with open(model_dir / 'metadata.json') as f:
                metadata = json.load(f)
                
            models.append({
                'name': model_dir.name,
                'type': metadata.get('model_type', 'unknown'),
                'accuracy': metadata.get('test_score', 0),
                'created': metadata.get('created_at', 'unknown'),
                'is_refined': 'refined' in model_dir.name,
                'original_model': metadata.get('original_model', None),
                'improvement': metadata.get('improvement', 0)
            })
    
    # Sort by accuracy
    models.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print("=" * 80)
    print("ALL MODELS RANKED BY ACCURACY")
    print("=" * 80)
    print(f"{'Model Name':<40} {'Type':<15} {'Accuracy':<10} {'Status':<20}")
    print("-" * 80)
    
    for model in models:
        status = ""
        if model['is_refined']:
            if model['improvement'] > 0:
                status = f"✓ Improved +{model['improvement']:.1%}"
            else:
                status = "Refined"
        else:
            status = "Original"
        
        print(f"{model['name']:<40} {model['type']:<15} {model['accuracy']:<10.1%} {status:<20}")
    
    # Check current default
    config_path = Path('models/person_recognition/config.json')
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            default_model = config.get('default_model')
            
        print(f"\nCurrent default model: {default_model}")
        
        # Find its accuracy
        for model in models:
            if model['name'] == default_model:
                print(f"Default model accuracy: {model['accuracy']:.1%}")
                break
    
    # Summary
    refined_models = [m for m in models if m['is_refined']]
    if refined_models:
        print(f"\nRefined models: {len(refined_models)}")
        improved = [m for m in refined_models if m.get('improvement', 0) > 0]
        print(f"Models that improved: {len(improved)}")
        
        if improved:
            best_improvement = max(improved, key=lambda x: x['improvement'])
            print(f"Best improvement: {best_improvement['name']} (+{best_improvement['improvement']:.1%})")


if __name__ == '__main__':
    test_refined_models()