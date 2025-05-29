"""
Person Recognition Blueprint
Manages dataset creation, model training, and inference for person recognition
"""

from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash, send_file
from flask_login import login_required
from pathlib import Path
import json
import os
from datetime import datetime
import tempfile
import shutil
import sys
import numpy as np
import subprocess
import uuid
from threading import Thread

person_recognition_bp = Blueprint('person_recognition', __name__, url_prefix='/person-recognition')

# Store for tracking refinement tasks
refinement_tasks = {}


def get_default_model():
    """Get the default person recognition model"""
    config_path = Path('models/person_recognition/config.json')
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
                default_model = config.get('default_model')
                if default_model:
                    # Check if model still exists
                    model_dir = Path('models/person_recognition') / default_model
                    if model_dir.exists():
                        return default_model
        except Exception as e:
            print(f"Error reading default model config: {e}")
    return None

@person_recognition_bp.route('/')
@login_required
def index():
    """Main person recognition page"""
    try:
        from hr_management.processing.person_recognition_trainer import PersonRecognitionTrainer
        trainer = PersonRecognitionTrainer()
        models = trainer.get_available_models()
    except ImportError as e:
        # If scikit-learn is not installed, show empty models list
        models = []
        flash('Some dependencies are missing. Please install: pip install scikit-learn joblib', 'warning')
    
    # Get available datasets
    datasets_dir = Path('datasets/person_recognition')
    datasets = []
    
    # Get default dataset from config
    default_dataset = None
    config_path = datasets_dir / 'config.json'
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
                default_dataset = config.get('default_dataset')
        except:
            pass
    
    if datasets_dir.exists():
        for dataset_dir in datasets_dir.iterdir():
            if dataset_dir.is_dir() and (dataset_dir / 'dataset_info.json').exists():
                with open(dataset_dir / 'dataset_info.json') as f:
                    info = json.load(f)
                    datasets.append({
                        'name': dataset_dir.name,
                        'created_at': info.get('created_at', 'Unknown'),
                        'num_persons': len(info.get('persons', {})),
                        'total_images': info.get('total_images', 0),
                        'total_faces': info.get('total_faces', info.get('total_features', 0)),  # Fallback to total_features
                        'is_default': dataset_dir.name == default_dataset
                    })
    
    # Get default model
    default_model = get_default_model()
    
    # Mark default model
    for model in models:
        model['is_default'] = model['name'] == default_model
    
    return render_template('person_recognition/index.html', 
                         models=models, datasets=datasets, default_model=default_model)


@person_recognition_bp.route('/datasets/create', methods=['POST'])
@login_required
def create_dataset():
    """Create a new dataset from selected persons"""
    try:
        data = request.get_json()
        person_ids = data.get('person_ids', [])
        dataset_name = data.get('dataset_name', '')
        
        if not person_ids:
            return jsonify({'success': False, 'error': 'No persons selected'})
        
        if not dataset_name:
            dataset_name = f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Validate dataset name
        dataset_name = dataset_name.replace(' ', '_').replace('-', '_')
        
        # Import dataset creator
        # Always use simple version for better compatibility with person detection (not just faces)
        from hr_management.processing.person_dataset_creator_simple import PersonDatasetCreatorSimple
        creator = PersonDatasetCreatorSimple()
        
        # Optional: Try to use face recognition version if explicitly requested
        use_face_recognition = data.get('use_face_recognition', False)
        if use_face_recognition:
            try:
                from hr_management.processing.person_dataset_creator import PersonDatasetCreator
                creator = PersonDatasetCreator()
                print("Using face recognition-based feature extraction")
            except ImportError:
                print("Face recognition not available, using appearance-based features")
                pass
        
        # Create dataset
        print(f"🔄 Creating dataset '{dataset_name}' with {len(person_ids)} persons...")
        dataset_info = creator.create_dataset_from_persons(person_ids, dataset_name)
        
        # Optionally augment dataset
        if data.get('augment', False):
            augmentation_factor = data.get('augmentation_factor', 3)
            aug_results = creator.augment_dataset(dataset_name, augmentation_factor)
            dataset_info['augmentation'] = aug_results
        
        # Set as default dataset
        config_path = Path('datasets/person_recognition/config.json')
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config = {}
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
            except:
                pass
        
        config['default_dataset'] = dataset_name
        config['updated_at'] = datetime.now().isoformat()
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return jsonify({
            'success': True,
            'dataset_name': dataset_name,
            'dataset_info': dataset_info,
            'set_as_default': True
        })
        
    except Exception as e:
        print(f"❌ Dataset creation error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@person_recognition_bp.route('/datasets/<dataset_name>')
@login_required
def dataset_details(dataset_name):
    """View dataset details"""
    dataset_path = Path('datasets/person_recognition') / dataset_name
    
    if not dataset_path.exists():
        flash('Dataset not found', 'error')
        return redirect(url_for('person_recognition.index'))
    
    # Load dataset info
    with open(dataset_path / 'dataset_info.json') as f:
        dataset_info = json.load(f)
    
    # Count images per person
    person_stats = []
    for person_id, person_data in dataset_info['persons'].items():
        if person_data.get('success'):
            stats = {
                'person_id': person_id,
                'images_count': person_data['images_count'],
                'faces_count': person_data.get('faces_count', 0),
                'embeddings_count': person_data.get('embeddings_count', 0),
                'features_count': person_data.get('features_count', 0)
            }
            # Add train/val stats if available
            if 'train_images_count' in person_data:
                stats['train_images_count'] = person_data['train_images_count']
                stats['val_images_count'] = person_data['val_images_count']
                stats['train_features_count'] = person_data.get('train_features_count', 0)
                stats['val_features_count'] = person_data.get('val_features_count', 0)
            person_stats.append(stats)
    
    return render_template('person_recognition/dataset_details.html',
                         dataset_name=dataset_name,
                         dataset_info=dataset_info,
                         person_stats=person_stats)


@person_recognition_bp.route('/train', methods=['POST'])
@login_required
def train_model():
    """Train a new person recognition model"""
    try:
        data = request.get_json()
        dataset_name = data.get('dataset_name')
        model_type = data.get('model_type', 'svm')
        model_name = data.get('model_name', None)
        target_accuracy = data.get('target_accuracy', 0.9)
        max_iterations = data.get('max_iterations', 10)
        validate_each_person = data.get('validate_each_person', True)
        
        if not dataset_name:
            return jsonify({'success': False, 'error': 'No dataset specified'})
        
        # Import necessary modules
        # Always use simple version for better compatibility with person detection (not just faces)
        from hr_management.processing.person_dataset_creator_simple import PersonDatasetCreatorSimple
        creator = PersonDatasetCreatorSimple()
        
        from hr_management.processing.person_recognition_trainer import PersonRecognitionTrainer
        
        # Prepare training data
        print(f"📊 Loading training data from dataset: {dataset_name}")
        X, y, person_ids = creator.prepare_training_data(dataset_name)
        
        print(f"📊 Loaded data: X shape = {np.array(X).shape if len(X) > 0 else 'empty'}, unique persons = {len(set(y)) if len(y) > 0 else 0}")
        
        if len(X) == 0:
            return jsonify({'success': False, 'error': 'No training data found'})
        
        # Check unique classes in the data
        unique_classes = len(np.unique(y))
        if unique_classes < 2:
            return jsonify({'success': False, 'error': f'Need at least 2 persons with valid data for training. Found only {unique_classes} person(s) with features.'})
        
        # Warn if some persons have no data
        if unique_classes < len(person_ids):
            print(f"⚠️  Warning: {len(person_ids) - unique_classes} person(s) have no valid training data")
        
        # Train model with continuous training parameters
        trainer = PersonRecognitionTrainer()
        # Store dataset name for validation data loading
        trainer.current_dataset_name = dataset_name
        results = trainer.train_model(
            X, y, person_ids, 
            model_type=model_type, 
            model_name=model_name,
            target_accuracy=target_accuracy,
            max_iterations=max_iterations,
            validate_each_person=validate_each_person
        )
        
        return jsonify({
            'success': True,
            'model_name': results['model_name'],
            'results': results
        })
        
    except Exception as e:
        print(f"❌ Training error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@person_recognition_bp.route('/models/<model_name>')
@login_required
def model_details(model_name):
    """View model details"""
    model_path = Path('models/person_recognition') / model_name
    
    if not model_path.exists():
        flash('Model not found', 'error')
        return redirect(url_for('person_recognition.index'))
    
    # Load model metadata
    with open(model_path / 'metadata.json') as f:
        metadata = json.load(f)
    
    return render_template('person_recognition/model_details.html',
                         model_name=model_name,
                         metadata=metadata)


@person_recognition_bp.route('/test-video', methods=['POST'])
@login_required
def test_video():
    """Test a model on an uploaded video"""
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No video file provided'})
        
        model_name = request.form.get('model_name')
        if not model_name:
            return jsonify({'success': False, 'error': 'No model specified'})
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'success': False, 'error': 'No video selected'})
        
        # Save uploaded video temporarily
        temp_dir = Path(tempfile.mkdtemp())
        video_path = temp_dir / video_file.filename
        video_file.save(str(video_path))
        
        # Process video
        # Always use simple version for consistency with training
        from hr_management.processing.person_recognition_inference_simple import PersonRecognitionInferenceSimple
        inference_class = PersonRecognitionInferenceSimple
        
        confidence_threshold = float(request.form.get('confidence_threshold', 0.6))
        inference = inference_class(model_name, confidence_threshold)
        
        # Process without output video for speed
        # Use skip_frames=5 for better detection coverage
        results = inference.process_video(str(video_path), skip_frames=5)
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        print(f"❌ Video test error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@person_recognition_bp.route('/test-image', methods=['POST'])
@login_required
def test_image():
    """Test a model on an uploaded image"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'})
        
        model_name = request.form.get('model_name')
        if not model_name:
            return jsonify({'success': False, 'error': 'No model specified'})
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'success': False, 'error': 'No image selected'})
        
        # Save uploaded image temporarily
        temp_dir = Path(tempfile.mkdtemp())
        image_path = temp_dir / image_file.filename
        image_file.save(str(image_path))
        
        # Process image
        # Always use simple version for consistency with training
        from hr_management.processing.person_recognition_inference_simple import PersonRecognitionInferenceSimple
        inference_class = PersonRecognitionInferenceSimple
        
        confidence_threshold = float(request.form.get('confidence_threshold', 0.6))
        is_pre_cropped = request.form.get('is_pre_cropped', 'false').lower() == 'true'
        
        print(f"🔍 Test image request - Model: {model_name}, Threshold: {confidence_threshold}, Pre-cropped: {is_pre_cropped}")
        
        inference = inference_class(model_name, confidence_threshold)
        
        # Use appropriate method based on whether image is pre-cropped
        if is_pre_cropped and hasattr(inference, 'process_cropped_image'):
            print(f"📸 Processing as pre-cropped image: {image_path}")
            results = inference.process_cropped_image(str(image_path))
        else:
            print(f"📸 Processing as full image: {image_path}")
            results = inference.process_image(str(image_path))
        
        print(f"📊 Results: {results}")
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        print(f"❌ Image test error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@person_recognition_bp.route('/datasets/<dataset_name>/delete', methods=['POST'])
@login_required
def delete_dataset(dataset_name):
    """Delete a dataset"""
    try:
        import shutil
        
        # Path to the dataset directory
        dataset_dir = Path('datasets/person_recognition') / dataset_name
        
        if not dataset_dir.exists():
            flash(f'Dataset {dataset_name} not found', 'error')
            return redirect(url_for('person_recognition.index'))
        
        # Check if this is the default dataset
        config_path = Path('datasets/person_recognition/config.json')
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                if config.get('default_dataset') == dataset_name:
                    # Clear the default
                    config['default_dataset'] = None
                    with open(config_path, 'w') as f:
                        json.dump(config, f, indent=2)
        
        # Delete the dataset directory
        shutil.rmtree(dataset_dir)
        
        flash(f'Dataset {dataset_name} has been deleted successfully', 'success')
        return redirect(url_for('person_recognition.index'))
        
    except Exception as e:
        flash(f'Error deleting dataset: {str(e)}', 'error')
        return redirect(url_for('person_recognition.dataset_details', dataset_name=dataset_name))


@person_recognition_bp.route('/datasets/<dataset_name>/set-default', methods=['POST'])
@login_required
def set_default_dataset(dataset_name):
    """Set a dataset as the default for training"""
    try:
        # Path to the dataset directory
        dataset_dir = Path('datasets/person_recognition') / dataset_name
        
        if not dataset_dir.exists():
            return jsonify({'success': False, 'error': f'Dataset {dataset_name} not found'})
        
        # Update config file
        config_path = Path('datasets/person_recognition/config.json')
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        
        config['default_dataset'] = dataset_name
        config['updated_at'] = datetime.now().isoformat()
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return jsonify({'success': True, 'message': f'{dataset_name} set as default dataset'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@person_recognition_bp.route('/api/persons/available')
@login_required
def get_available_persons():
    """Get list of available persons for dataset creation"""
    persons_dir = Path('processing/outputs/persons')
    persons = []
    
    if persons_dir.exists():
        for person_dir in persons_dir.iterdir():
            if person_dir.is_dir() and person_dir.name.startswith('PERSON-'):
                metadata_path = person_dir / 'metadata.json'
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    
                    # Count actual images
                    image_count = len(list(person_dir.glob('*.jpg')) + list(person_dir.glob('*.png')))
                    
                    persons.append({
                        'person_id': person_dir.name,
                        'image_count': image_count,
                        'total_detections': metadata.get('total_detections', 0),
                        'videos': len(metadata.get('videos', []))
                    })
    
    return jsonify({
        'persons': sorted(persons, key=lambda x: x['person_id'])
    })


@person_recognition_bp.route('/models/<model_name>/set-default', methods=['POST'])
@login_required
def set_default_model(model_name):
    """Set a model as the default for person recognition"""
    try:
        # Path to the model directory
        model_dir = Path('models/person_recognition') / model_name
        
        if not model_dir.exists():
            return jsonify({'success': False, 'error': f'Model {model_name} not found'})
        
        # Update config file
        config_path = Path('models/person_recognition/config.json')
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        
        config['default_model'] = model_name
        config['updated_at'] = datetime.now().isoformat()
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return jsonify({'success': True, 'message': f'{model_name} set as default model'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@person_recognition_bp.route('/models/<model_name>/delete', methods=['POST'])
@login_required
def delete_model(model_name):
    """Delete a person recognition model"""
    try:
        import shutil
        
        # Debug logging
        print(f"Attempting to delete model: {model_name}")
        
        # Path to the model directory
        model_dir = Path('models/person_recognition') / model_name
        print(f"Model directory path: {model_dir}")
        print(f"Model directory exists: {model_dir.exists()}")
        
        if not model_dir.exists():
            flash(f'Model {model_name} not found at {model_dir}', 'error')
            return redirect(url_for('person_recognition.index'))
        
        # Check if this is the default model
        config_path = Path('models/person_recognition/config.json')
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                if config.get('default_model') == model_name:
                    flash('Cannot delete the default model. Please set another model as default first.', 'error')
                    return redirect(url_for('person_recognition.model_details', model_name=model_name))
        
        # Delete the model directory
        shutil.rmtree(model_dir)
        
        flash(f'Model {model_name} has been deleted successfully', 'success')
        return redirect(url_for('person_recognition.index'))
        
    except Exception as e:
        flash(f'Error deleting model: {str(e)}', 'error')
        return redirect(url_for('person_recognition.model_details', model_name=model_name))


# Import and register refinement routes
from .person_recognition_refinement import register_refinement_routes
register_refinement_routes(person_recognition_bp, refinement_tasks)