from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_required
from models.face_recognition import FaceDataset, TrainedModel, RecognitionResult
from models.video import Video, DetectedPerson
from models.employee import Employee
from models.base import db
import os
from datetime import datetime

face_recognition_bp = Blueprint('face_recognition', __name__)

@face_recognition_bp.route('/datasets')
@login_required
def datasets():
    """Display face datasets"""
    datasets = FaceDataset.query.order_by(FaceDataset.created_at.desc()).all()
    return render_template('face_recognition/datasets.html', datasets=datasets)

@face_recognition_bp.route('/datasets/create', methods=['GET', 'POST'])
@login_required
def create_dataset():
    """Create a new face dataset"""
    if request.method == 'POST':
        dataset = FaceDataset(
            name=request.form['name'],
            description=request.form.get('description', ''),
            dataset_path=f"datasets/faces/{request.form['name'].lower().replace(' ', '_')}",
            format_type=request.form.get('format_type', 'yolo')
        )
        
        try:
            # Create directory structure
            os.makedirs(dataset.dataset_path, exist_ok=True)
            
            db.session.add(dataset)
            db.session.commit()
            
            flash(f'Dataset "{dataset.name}" created successfully!', 'success')
            return redirect(url_for('face_recognition.dataset_detail', id=dataset.id))
        except Exception as e:
            db.session.rollback()
            flash(f'Error creating dataset: {str(e)}', 'error')
    
    return render_template('face_recognition/create_dataset.html')

@face_recognition_bp.route('/datasets/<int:id>')
@login_required
def dataset_detail(id):
    """Display dataset details"""
    dataset = FaceDataset.query.get_or_404(id)
    
    # Get statistics
    person_dirs = []
    total_images = 0
    
    if os.path.exists(dataset.dataset_path):
        for person_dir in os.listdir(dataset.dataset_path):
            person_path = os.path.join(dataset.dataset_path, person_dir)
            if os.path.isdir(person_path):
                image_count = len([f for f in os.listdir(person_path) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                person_dirs.append({
                    'name': person_dir,
                    'image_count': image_count
                })
                total_images += image_count
    
    # Update dataset statistics
    dataset.person_count = len(person_dirs)
    dataset.image_count = total_images
    db.session.commit()
    
    return render_template('face_recognition/dataset_detail.html',
                         dataset=dataset,
                         person_dirs=person_dirs)

@face_recognition_bp.route('/datasets/<int:id>/extract-faces', methods=['POST'])
@login_required
def extract_faces(id):
    """Extract faces from detected persons to build dataset"""
    dataset = FaceDataset.query.get_or_404(id)
    
    # Get selected videos or use all completed videos
    video_ids = request.form.getlist('video_ids')
    if not video_ids:
        videos = Video.query.filter_by(status='completed').all()
        video_ids = [v.id for v in videos]
    
    try:
        # Queue face extraction task (would integrate with Celery here)
        # extract_faces_task.delay(dataset.id, video_ids)
        
        dataset.status = 'processing'
        db.session.commit()
        
        flash('Face extraction started! This may take a while.', 'info')
    except Exception as e:
        flash(f'Error starting face extraction: {str(e)}', 'error')
    
    return redirect(url_for('face_recognition.dataset_detail', id=id))

@face_recognition_bp.route('/models')
@login_required
def models():
    """Display trained models"""
    models = TrainedModel.query.order_by(TrainedModel.created_at.desc()).all()
    return render_template('face_recognition/models.html', models=models)

@face_recognition_bp.route('/models/create', methods=['GET', 'POST'])
@login_required
def create_model():
    """Create and train a new model"""
    if request.method == 'POST':
        dataset = FaceDataset.query.get(request.form['dataset_id'])
        if not dataset:
            flash('Selected dataset not found', 'error')
            return redirect(request.url)
        
        model = TrainedModel(
            name=request.form['name'],
            description=request.form.get('description', ''),
            model_type=request.form.get('model_type', 'face_recognition'),
            dataset_id=dataset.id,
            epochs=int(request.form.get('epochs', 10)),
            batch_size=int(request.form.get('batch_size', 32)),
            learning_rate=float(request.form.get('learning_rate', 0.001)),
            model_path=f"models/{request.form['name'].lower().replace(' ', '_')}",
            version=request.form.get('version', '1.0.0')
        )
        
        try:
            # Create model directory
            os.makedirs(model.model_path, exist_ok=True)
            
            db.session.add(model)
            db.session.commit()
            
            # Queue training task (would integrate with Celery here)
            # train_model_task.delay(model.id)
            
            model.status = 'training'
            db.session.commit()
            
            flash(f'Model "{model.name}" created and training started!', 'success')
            return redirect(url_for('face_recognition.model_detail', id=model.id))
        except Exception as e:
            db.session.rollback()
            flash(f'Error creating model: {str(e)}', 'error')
    
    # Get available datasets
    datasets = FaceDataset.query.filter_by(status='ready').all()
    return render_template('face_recognition/create_model.html', datasets=datasets)

@face_recognition_bp.route('/models/<int:id>')
@login_required
def model_detail(id):
    """Display model details"""
    model = TrainedModel.query.get_or_404(id)
    
    # Get recent recognition results
    recent_results = RecognitionResult.query.filter_by(model_id=id)\
                                           .order_by(RecognitionResult.created_at.desc())\
                                           .limit(10).all()
    
    return render_template('face_recognition/model_detail.html',
                         model=model,
                         recent_results=recent_results)

@face_recognition_bp.route('/models/<int:id>/deploy', methods=['POST'])
@login_required
def deploy_model(id):
    """Deploy model for production use"""
    model = TrainedModel.query.get_or_404(id)
    
    if model.status != 'completed':
        flash('Model must be completed before deployment', 'error')
        return redirect(url_for('face_recognition.model_detail', id=id))
    
    try:
        # Deactivate other models of the same type
        TrainedModel.query.filter_by(model_type=model.model_type, is_active=True)\
                         .update({'is_active': False})
        
        # Activate this model
        model.is_active = True
        model.deployed_at = datetime.utcnow()
        
        db.session.commit()
        
        flash(f'Model "{model.name}" deployed successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deploying model: {str(e)}', 'error')
    
    return redirect(url_for('face_recognition.model_detail', id=id))

@face_recognition_bp.route('/recognition')
@login_required
def recognition():
    """Real-time recognition interface"""
    # Get active models
    active_models = TrainedModel.query.filter_by(is_active=True).all()
    
    # Get recent recognition results
    recent_results = RecognitionResult.query.order_by(RecognitionResult.created_at.desc())\
                                           .limit(50).all()
    
    return render_template('face_recognition/recognition.html',
                         active_models=active_models,
                         recent_results=recent_results)

@face_recognition_bp.route('/recognition/video/<int:id>', methods=['POST'])
@login_required
def recognize_video(id):
    """Run recognition on a specific video"""
    video = Video.query.get_or_404(id)
    
    if video.status != 'completed':
        flash('Video must be processed before running recognition', 'error')
        return redirect(url_for('videos.detail', id=id))
    
    # Get active face recognition model
    active_model = TrainedModel.query.filter_by(
        model_type='face_recognition',
        is_active=True
    ).first()
    
    if not active_model:
        flash('No active face recognition model found', 'error')
        return redirect(url_for('videos.detail', id=id))
    
    try:
        # Queue recognition task (would integrate with Celery here)
        # recognize_video_task.delay(video.id, active_model.id)
        
        flash('Face recognition started for this video!', 'info')
    except Exception as e:
        flash(f'Error starting recognition: {str(e)}', 'error')
    
    return redirect(url_for('videos.detail', id=id))

@face_recognition_bp.route('/api/models')
@login_required
def api_models():
    """API endpoint to get models data"""
    models = TrainedModel.query.all()
    return jsonify([model.to_dict() for model in models])

@face_recognition_bp.route('/api/datasets')
@login_required
def api_datasets():
    """API endpoint to get datasets data"""
    datasets = FaceDataset.query.all()
    return jsonify([dataset.to_dict() for dataset in datasets])

@face_recognition_bp.route('/api/recognition-results')
@login_required
def api_recognition_results():
    """API endpoint to get recognition results"""
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    results = RecognitionResult.query.order_by(RecognitionResult.created_at.desc())\
                                    .paginate(page=page, per_page=per_page, error_out=False)
    
    return jsonify({
        'results': [result.to_dict() for result in results.items],
        'total': results.total,
        'pages': results.pages,
        'current_page': page
    })