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

person_recognition_bp = Blueprint('person_recognition', __name__, url_prefix='/person-recognition')

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
                        'total_faces': info.get('total_faces', info.get('total_features', 0))  # Fallback to total_features
                    })
    
    return render_template('person_recognition/index.html', 
                         models=models, datasets=datasets)


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
        try:
            from hr_management.processing.person_dataset_creator import PersonDatasetCreator
            creator = PersonDatasetCreator()
        except ImportError:
            # Use simple version if face_recognition is not available
            from hr_management.processing.person_dataset_creator_simple import PersonDatasetCreatorSimple
            creator = PersonDatasetCreatorSimple()
        
        # Create dataset
        print(f"🔄 Creating dataset '{dataset_name}' with {len(person_ids)} persons...")
        dataset_info = creator.create_dataset_from_persons(person_ids, dataset_name)
        
        # Optionally augment dataset
        if data.get('augment', False):
            augmentation_factor = data.get('augmentation_factor', 3)
            aug_results = creator.augment_dataset(dataset_name, augmentation_factor)
            dataset_info['augmentation'] = aug_results
        
        return jsonify({
            'success': True,
            'dataset_name': dataset_name,
            'dataset_info': dataset_info
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
            person_stats.append({
                'person_id': person_id,
                'images_count': person_data['images_count'],
                'faces_count': person_data['faces_count'],
                'embeddings_count': person_data['embeddings_count']
            })
    
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
        
        if not dataset_name:
            return jsonify({'success': False, 'error': 'No dataset specified'})
        
        # Import necessary modules
        try:
            from hr_management.processing.person_dataset_creator import PersonDatasetCreator
            creator = PersonDatasetCreator()
        except ImportError:
            # Use simple version if face_recognition is not available
            from hr_management.processing.person_dataset_creator_simple import PersonDatasetCreatorSimple
            creator = PersonDatasetCreatorSimple()
        
        from hr_management.processing.person_recognition_trainer import PersonRecognitionTrainer
        
        # Prepare training data
        X, y, person_ids = creator.prepare_training_data(dataset_name)
        
        if len(X) == 0:
            return jsonify({'success': False, 'error': 'No training data found'})
        
        # Check unique classes in the data
        unique_classes = len(np.unique(y))
        if unique_classes < 2:
            return jsonify({'success': False, 'error': f'Need at least 2 persons with valid data for training. Found only {unique_classes} person(s) with features.'})
        
        # Warn if some persons have no data
        if unique_classes < len(person_ids):
            print(f"⚠️  Warning: {len(person_ids) - unique_classes} person(s) have no valid training data")
        
        # Train model
        trainer = PersonRecognitionTrainer()
        results = trainer.train_model(X, y, person_ids, model_type, model_name)
        
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
        try:
            from hr_management.processing.person_recognition_inference import PersonRecognitionInference
            inference_class = PersonRecognitionInference
        except ImportError:
            # Use simple version if face_recognition is not available
            from hr_management.processing.person_recognition_inference_simple import PersonRecognitionInferenceSimple
            inference_class = PersonRecognitionInferenceSimple
        
        confidence_threshold = float(request.form.get('confidence_threshold', 0.6))
        inference = inference_class(model_name, confidence_threshold)
        
        # Process without output video for speed
        results = inference.process_video(str(video_path), skip_frames=10)
        
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
        try:
            from hr_management.processing.person_recognition_inference import PersonRecognitionInference
            inference_class = PersonRecognitionInference
        except ImportError:
            # Use simple version if face_recognition is not available
            from hr_management.processing.person_recognition_inference_simple import PersonRecognitionInferenceSimple
            inference_class = PersonRecognitionInferenceSimple
        
        confidence_threshold = float(request.form.get('confidence_threshold', 0.6))
        inference = inference_class(model_name, confidence_threshold)
        
        results = inference.process_image(str(image_path))
        
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