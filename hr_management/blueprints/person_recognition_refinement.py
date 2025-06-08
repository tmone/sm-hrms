"""
Model refinement endpoints for person recognition blueprint
"""

from flask import request, jsonify
from flask_login import login_required
from pathlib import Path
import json
import subprocess
import sys
import uuid
from threading import Thread
from datetime import datetime


def run_refinement_task(task_id, model_name, refinement_type, refinement_tasks):
    """Run model refinement in background thread"""
    try:
        refinement_tasks[task_id]['status'] = 'running'
        refinement_tasks[task_id]['message'] = 'Starting refinement process...'
        
        # Use the advanced refinement script for better results
        model_path = f'models/person_recognition/{model_name}'
        
        # Check if advanced dependencies are available
        try:
            import imblearn
            use_advanced = True
        except ImportError:
            use_advanced = False
            print("Note: Install imbalanced-learn for better refinement: pip install imbalanced-learn")
        
        if use_advanced:
            args = [sys.executable, 'scripts/advanced_refine_model.py', model_path]
        else:
            args = [sys.executable, 'scripts/simple_refine_model.py', model_path]
        
        if refinement_type == 'quick':
            args.extend(['--type', 'quick'])
            refinement_tasks[task_id]['message'] = 'Running quick refinement...'
        elif refinement_type == 'standard':
            args.extend(['--type', 'standard'])
            refinement_tasks[task_id]['message'] = 'Running standard refinement with optimization...'
        elif refinement_type == 'advanced':
            args.extend(['--type', 'advanced' if use_advanced else 'standard'])
            refinement_tasks[task_id]['message'] = 'Running advanced refinement with ensemble models...'
        elif refinement_type == 'random_forest':
            args.extend(['--type', 'random_forest'])
            refinement_tasks[task_id]['message'] = 'Training Random Forest model...'
        elif refinement_type == 'mlp':
            args.extend(['--type', 'mlp'])
            refinement_tasks[task_id]['message'] = 'Training Neural Network model...'
        elif refinement_type == 'gradient_boost':
            args.extend(['--type', 'gradient_boost' if use_advanced else 'random_forest'])
            refinement_tasks[task_id]['message'] = 'Training Gradient Boosting model...'
        elif refinement_type == 'auto':
            # Use the auto refinement script
            args = [sys.executable, 'scripts/auto_refine_best_model.py', '--model', model_name]
            refinement_tasks[task_id]['message'] = 'Automatically finding the best model...'
        else:
            args.extend(['--type', 'standard'])
        
        # Run the refinement script
        result = subprocess.run(args, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Parse output to get new model name
            output_lines = result.stdout.strip().split('\n')
            new_model_name = None
            test_accuracy = None
            
            # For auto refinement, parse the summary
            if 'BEST MODEL FOUND!' in result.stdout:
                # Parse auto refinement output
                for line in output_lines:
                    if 'Model path:' in line:
                        model_path = line.split('Model path:')[1].strip()
                        new_model_name = Path(model_path).name
                    elif 'Accuracy:' in line and 'New accuracy:' not in line and test_accuracy is None:
                        try:
                            acc_str = line.split('Accuracy:')[1].strip().replace('%', '')
                            test_accuracy = float(acc_str) / 100 if float(acc_str) > 1 else float(acc_str)
                        except:
                            pass
                    elif 'Default model updated to:' in line:
                        model_name = line.split('Default model updated to:')[1].strip()
                        if not new_model_name:
                            new_model_name = model_name
            else:
                # Parse regular refinement output
                for line in output_lines:
                    if 'Model saved to:' in line or 'New model:' in line:
                        # Extract model path
                        if 'Model saved to:' in line:
                            model_path = line.split('Model saved to:')[1].strip()
                        else:
                            model_path = line.split('New model:')[1].strip()
                        
                        # Handle both full path and just model name
                        if '/' in model_path or '\\' in model_path:
                            new_model_name = Path(model_path).name
                        else:
                            new_model_name = model_path
                            
                    elif 'Test Accuracy:' in line or 'New accuracy:' in line:
                        try:
                            # Handle both "0.850" and "85.0%" formats
                            acc_str = line.split(':')[1].strip().replace('%', '')
                            test_accuracy = float(acc_str)
                            if test_accuracy > 1:  # Convert percentage to decimal
                                test_accuracy = test_accuracy / 100
                        except:
                            pass
            
            # If we couldn't parse the model name, check the config file
            if not new_model_name and refinement_type == 'auto':
                try:
                    import json
                    config_path = Path('models/person_recognition/config.json')
                    if config_path.exists():
                        with open(config_path) as f:
                            config = json.load(f)
                            new_model_name = config.get('default_model')
                            
                            # Try to get accuracy from the new model
                            if new_model_name and not test_accuracy:
                                model_metadata_path = Path('models/person_recognition') / new_model_name / 'metadata.json'
                                if model_metadata_path.exists():
                                    with open(model_metadata_path) as f:
                                        model_metadata = json.load(f)
                                        test_accuracy = model_metadata.get('test_score', 0)
                except:
                    pass
            
            # Check if this is the best model and if it's actually an improvement
            is_best_model = False
            current_default_model = None
            
            if new_model_name and test_accuracy:
                try:
                    # Check current default model
                    config_path = Path('models/person_recognition/config.json')
                    if config_path.exists():
                        with open(config_path) as f:
                            config = json.load(f)
                            current_default_model = config.get('default_model')
                    
                    # Check if we refined the default model
                    refined_model_name = model_name  # The model we started with
                    is_refining_default = (refined_model_name == current_default_model)
                    
                    from hr_management.processing.person_recognition_trainer import PersonRecognitionTrainer
                    trainer = PersonRecognitionTrainer()
                    all_models = trainer.get_available_models()
                    
                    # Find the highest accuracy among all models
                    best_accuracy = 0
                    current_default_accuracy = 0
                    
                    for model in all_models:
                        model_accuracy = model.get('test_score', 0)
                        if model_accuracy > best_accuracy:
                            best_accuracy = model_accuracy
                        if model['name'] == current_default_model:
                            current_default_accuracy = model_accuracy
                    
                    # Only show modal if:
                    # 1. New model is better than ALL existing models (including the one we refined)
                    # 2. AND it's not just a refinement of the current default that didn't improve
                    if test_accuracy > best_accuracy:
                        # If we refined the default and it improved, auto-set as default
                        if is_refining_default:
                            # Auto-update default without asking
                            config['default_model'] = new_model_name
                            config['last_updated'] = datetime.now().isoformat()
                            config['auto_refined'] = True
                            config['refinement_type'] = refinement_type
                            
                            with open(config_path, 'w') as f:
                                json.dump(config, f, indent=2)
                            
                            is_best_model = False  # Don't show modal
                        else:
                            # Refined a non-default model and it's now the best
                            is_best_model = True
                except:
                    pass
            
            refinement_tasks[task_id]['status'] = 'completed'
            refinement_tasks[task_id]['result'] = {
                'model_name': new_model_name,
                'test_accuracy': test_accuracy,
                'is_best_model': is_best_model,
                'output': result.stdout
            }
            refinement_tasks[task_id]['message'] = 'Refinement completed successfully!'
        else:
            refinement_tasks[task_id]['status'] = 'failed'
            refinement_tasks[task_id]['error'] = result.stderr or result.stdout
            refinement_tasks[task_id]['message'] = 'Refinement failed'
            
    except Exception as e:
        refinement_tasks[task_id]['status'] = 'failed'
        refinement_tasks[task_id]['error'] = str(e)
        refinement_tasks[task_id]['message'] = f'Error: {str(e)}'


def register_refinement_routes(bp, refinement_tasks):
    """Register refinement routes to the blueprint"""
    
    @bp.route('/models/<model_name>/refine', methods=['POST'])
    @login_required
    def refine_model(model_name):
        """Start model refinement process"""
        try:
            data = request.get_json()
            refinement_type = data.get('refinement_type', 'standard')
            
            # Check if model exists
            model_path = Path('models/person_recognition') / model_name
            if not model_path.exists():
                return jsonify({'error': 'Model not found'}), 404
            
            # Create task ID
            task_id = str(uuid.uuid4())
            
            # Initialize task
            refinement_tasks[task_id] = {
                'id': task_id,
                'model_name': model_name,
                'refinement_type': refinement_type,
                'status': 'pending',
                'message': 'Initializing refinement...',
                'created_at': datetime.now().isoformat()
            }
            
            # Start refinement in background thread
            thread = Thread(
                target=run_refinement_task,
                args=(task_id, model_name, refinement_type, refinement_tasks)
            )
            thread.daemon = True
            thread.start()
            
            return jsonify({
                'status': 'started',
                'task_id': task_id,
                'message': 'Refinement process started'
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    
    @bp.route('/tasks/<task_id>/status', methods=['GET'])
    @login_required
    def get_task_status(task_id):
        """Get status of a refinement task"""
        if task_id not in refinement_tasks:
            return jsonify({'error': 'Task not found'}), 404
        
        task = refinement_tasks[task_id]
        response = {
            'task_id': task_id,
            'status': task['status'],
            'message': task.get('message', ''),
            'created_at': task.get('created_at')
        }
        
        if task['status'] == 'completed':
            response['result'] = task.get('result', {})
        elif task['status'] == 'failed':
            response['error'] = task.get('error', 'Unknown error')
        
        # Clean up old completed/failed tasks (older than 1 hour)
        if task['status'] in ['completed', 'failed']:
            created_at = datetime.fromisoformat(task['created_at'])
            if (datetime.now() - created_at).total_seconds() > 3600:
                del refinement_tasks[task_id]
        
        return jsonify(response)