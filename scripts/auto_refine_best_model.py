#!/usr/bin/env python3
"""
Automatic Model Refinement - Finds the best model automatically
"""

import os
import sys
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_refinement(model_path, refinement_type):
    """Run a single refinement and return results"""
    print(f"\n{'='*60}")
    print(f"Trying {refinement_type} refinement...")
    print(f"{'='*60}")
    
    # First try advanced script
    script_path = 'scripts/advanced_refine_model.py'
    if not os.path.exists(script_path):
        script_path = 'scripts/simple_refine_model.py'
    
    args = [sys.executable, script_path, model_path, '--type', refinement_type]
    
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=600)  # 10 minute timeout
        
        if result.returncode == 0:
            # Parse output to get accuracy
            output_lines = result.stdout.strip().split('\n')
            new_model_path = None
            test_accuracy = None
            
            for line in output_lines:
                if 'New accuracy:' in line:
                    try:
                        acc_str = line.split(':')[1].strip().replace('%', '')
                        test_accuracy = float(acc_str) / 100 if float(acc_str) > 1 else float(acc_str)
                    except:
                        pass
                elif 'Model saved to:' in line or 'New model:' in line:
                    if 'Model saved to:' in line:
                        new_model_path = line.split('Model saved to:')[1].strip()
                    else:
                        new_model_path = line.split('New model:')[1].strip()
            
            return {
                'success': True,
                'type': refinement_type,
                'accuracy': test_accuracy,
                'model_path': new_model_path,
                'output': result.stdout
            }
        else:
            return {
                'success': False,
                'type': refinement_type,
                'error': result.stderr or result.stdout
            }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'type': refinement_type,
            'error': 'Timeout - refinement took too long'
        }
    except Exception as e:
        return {
            'success': False,
            'type': refinement_type,
            'error': str(e)
        }


def auto_refine_all(model_path):
    """Try all refinement types and find the best model"""
    
    # Load current model metadata
    metadata_path = os.path.join(model_path, 'metadata.json')
    if not os.path.exists(metadata_path):
        print(f"Error: Model metadata not found at {metadata_path}")
        return None
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    current_accuracy = metadata.get('test_score', 0)
    print(f"\nCurrent model accuracy: {current_accuracy:.1%}")
    print(f"Model type: {metadata.get('model_type', 'unknown')}")
    
    # Define refinement strategies to try
    # Order by expected effectiveness and speed
    refinement_types = [
        'gradient_boost',  # Often best results
        'random_forest',   # Good alternative
        'standard',        # Balanced approach
        'mlp',            # Neural network
        'advanced',       # Ensemble (slower)
        'quick'           # Fast baseline
    ]
    
    print(f"\nWill try {len(refinement_types)} refinement strategies...")
    print("This may take 5-15 minutes depending on dataset size.\n")
    
    results = []
    best_result = None
    best_accuracy = current_accuracy
    
    # Try each refinement type
    for i, refinement_type in enumerate(refinement_types):
        print(f"\n[{i+1}/{len(refinement_types)}] Attempting {refinement_type} refinement...")
        start_time = time.time()
        
        result = run_refinement(model_path, refinement_type)
        elapsed_time = time.time() - start_time
        
        if result['success'] and result['accuracy']:
            improvement = result['accuracy'] - current_accuracy
            print(f"Result: Accuracy = {result['accuracy']:.1%} (improvement: {improvement:+.1%})")
            print(f"Time taken: {elapsed_time:.1f} seconds")
            
            results.append({
                'type': refinement_type,
                'accuracy': result['accuracy'],
                'improvement': improvement,
                'model_path': result['model_path'],
                'time': elapsed_time
            })
            
            if result['accuracy'] > best_accuracy:
                best_accuracy = result['accuracy']
                best_result = result
        else:
            print(f"Failed: {result.get('error', 'Unknown error')}")
            results.append({
                'type': refinement_type,
                'success': False,
                'error': result.get('error', 'Unknown error')
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("REFINEMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Original accuracy: {current_accuracy:.1%}")
    
    # Sort results by accuracy
    successful_results = [r for r in results if r.get('accuracy')]
    successful_results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    if successful_results:
        print("\nSuccessful refinements (sorted by accuracy):")
        for r in successful_results:
            print(f"  {r['type']:15} - Accuracy: {r['accuracy']:.1%} (improvement: {r['improvement']:+.1%}) - Time: {r['time']:.1f}s")
    
    failed_results = [r for r in results if not r.get('accuracy')]
    if failed_results:
        print("\nFailed refinements:")
        for r in failed_results:
            print(f"  {r['type']:15} - {r.get('error', 'Unknown error')}")
    
    if best_result and best_result['accuracy'] > current_accuracy:
        improvement = best_result['accuracy'] - current_accuracy
        print(f"\nBEST MODEL FOUND!")
        print(f"Type: {best_result['type']}")
        print(f"Accuracy: {best_result['accuracy']:.1%}")
        print(f"Improvement: {improvement:+.1%}")
        print(f"Model path: {best_result['model_path']}")
        
        # Update default model configuration
        if best_result['model_path']:
            config_path = Path('models/person_recognition/config.json')
            model_name = os.path.basename(best_result['model_path'])
            
            config = {}
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
            
            config['default_model'] = model_name
            config['last_updated'] = datetime.now().isoformat()
            config['auto_refined'] = True
            config['refinement_type'] = best_result['type']
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"\nDefault model updated to: {model_name}")
    else:
        print(f"\nNo improvement found. Current model remains the best.")
        print("Consider:")
        print("- Adding more training data")
        print("- Improving data quality")
        print("- Manual hyperparameter tuning")
    
    return best_result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Automatically find the best refined model')
    parser.add_argument('--model', type=str, help='Model to refine (default: current default model)')
    
    args = parser.parse_args()
    
    # Find model to refine
    if args.model:
        model_path = f'models/person_recognition/{args.model}'
    else:
        # Use default model
        config_path = Path('models/person_recognition/config.json')
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                default_model = config.get('default_model')
                if default_model:
                    model_path = f'models/person_recognition/{default_model}'
                else:
                    print("No default model set. Please specify a model with --model")
                    return
        else:
            print("No model configuration found. Please specify a model with --model")
            return
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    
    print(f"Starting automatic refinement for: {model_path}")
    auto_refine_all(model_path)


if __name__ == '__main__':
    main()