#!/usr/bin/env python3
"""
Example usage of the model refinement script

This demonstrates different ways to improve the person recognition model accuracy.
"""

import subprocess
import sys

def run_refinement(args):
    """Run the refinement script with given arguments"""
    cmd = [sys.executable, 'scripts/refine_person_model.py'] + args
    print(f"\nRunning: {' '.join(cmd)}")
    print("-" * 80)
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    return result.returncode == 0

def main():
    print("Person Recognition Model Refinement Examples")
    print("=" * 80)
    
    # Example 1: Basic refinement with hyperparameter tuning (recommended)
    print("\n1. Basic refinement with hyperparameter tuning:")
    print("This will tune hyperparameters to find the best model configuration")
    success = run_refinement([
        '--model-type', 'svm',
        '--existing-model', 'models/person_recognition/person_model_svm_20250529_153351'
    ])
    
    if not success:
        print("\nNote: If the refinement failed, ensure you have:")
        print("- Sufficient person detections in the database")
        print("- At least 2 different person IDs")
        print("- Person images associated with detections")
        return
    
    # Example 2: Quick refinement without hyperparameter tuning
    print("\n2. Quick refinement (no hyperparameter tuning):")
    print("Faster but may not achieve optimal accuracy")
    run_refinement([
        '--model-type', 'svm',
        '--no-tuning',
        '--existing-model', 'models/person_recognition/person_model_svm_20250529_153351'
    ])
    
    # Example 3: Advanced features with PCA
    print("\n3. Advanced feature extraction with PCA:")
    print("Uses more sophisticated features but takes longer")
    run_refinement([
        '--model-type', 'svm',
        '--advanced-features',
        '--use-pca',
        '--existing-model', 'models/person_recognition/person_model_svm_20250529_153351'
    ])
    
    # Example 4: Random Forest model
    print("\n4. Try Random Forest classifier:")
    print("Alternative model type that may work better for some datasets")
    run_refinement([
        '--model-type', 'random_forest',
        '--existing-model', 'models/person_recognition/person_model_svm_20250529_153351'
    ])
    
    # Example 5: MLP (Neural Network) model
    print("\n5. Try MLP (Neural Network) classifier:")
    print("Deep learning approach for complex patterns")
    run_refinement([
        '--model-type', 'mlp',
        '--existing-model', 'models/person_recognition/person_model_svm_20250529_153351'
    ])
    
    print("\n" + "=" * 80)
    print("Refinement examples completed!")
    print("\nTo use a refined model, update the config with:")
    print("python scripts/refine_person_model.py --model-type svm --update-default")
    print("\nFor more options, run:")
    print("python scripts/refine_person_model.py --help")

if __name__ == '__main__':
    main()