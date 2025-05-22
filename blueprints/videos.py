from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app
from flask_login import login_required
from werkzeug.utils import secure_filename
from models.video import Video, DetectedPerson
from models.base import db
import os
import uuid
from datetime import datetime

videos_bp = Blueprint('videos', __name__)

@videos_bp.route('/')
@login_required
def index():
    # Get filter parameters
    status = request.args.get('status', '')
    search = request.args.get('search', '')
    page = request.args.get('page', 1, type=int)
    per_page = 12
    
    # Build query
    query = Video.query
    
    if status:
        query = query.filter_by(status=status)
    
    if search:
        query = query.filter(Video.filename.contains(search))
    
    # Order by creation date (newest first)
    query = query.order_by(Video.created_at.desc())
    
    # Paginate results
    videos = query.paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    # Get status counts for filter buttons
    status_counts = {
        'all': Video.query.count(),
        'uploaded': Video.query.filter_by(status='uploaded').count(),
        'processing': Video.query.filter_by(status='processing').count(),
        'completed': Video.query.filter_by(status='completed').count(),
        'failed': Video.query.filter_by(status='failed').count()
    }
    
    return render_template('videos/index.html',
                         videos=videos,
                         status_counts=status_counts,
                         current_status=status,
                         search=search)

@videos_bp.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Generate unique filename
            filename = secure_filename(file.filename)
            name, ext = os.path.splitext(filename)
            unique_filename = f"{name}_{uuid.uuid4().hex[:8]}{ext}"
            
            # Save file
            upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], 'videos')
            os.makedirs(upload_path, exist_ok=True)
            filepath = os.path.join(upload_path, unique_filename)
            file.save(filepath)
            
            # Create database record
            video = Video(
                filename=unique_filename,
                original_filename=filename,
                filepath=filepath,
                file_size=os.path.getsize(filepath),
                status='uploaded'
            )
            
            try:
                db.session.add(video)
                db.session.commit()
                
                # Queue for processing (would integrate with Celery here)
                # process_video.delay(video.id)
                
                flash(f'Video "{filename}" uploaded successfully!', 'success')
                return redirect(url_for('videos.detail', id=video.id))
            except Exception as e:
                db.session.rollback()
                # Clean up file if database insert fails
                if os.path.exists(filepath):
                    os.remove(filepath)
                flash(f'Error saving video: {str(e)}', 'error')
        else:
            flash('Invalid file type. Please upload MP4, AVI, MOV, MKV, or WEBM files.', 'error')
    
    return render_template('videos/upload.html')

@videos_bp.route('/<int:id>')
@login_required
def detail(id):
    video = Video.query.get_or_404(id)
    
    # Get detected persons for this video
    detected_persons = DetectedPerson.query.filter_by(video_id=id)\
                                          .order_by(DetectedPerson.start_time)\
                                          .all()
    
    return render_template('videos/detail.html',
                         video=video,
                         detected_persons=detected_persons)

@videos_bp.route('/<int:id>/delete', methods=['POST'])
@login_required
def delete(id):
    video = Video.query.get_or_404(id)
    
    try:
        # Delete file from disk
        if os.path.exists(video.filepath):
            os.remove(video.filepath)
        
        # Delete from database (cascade will handle related records)
        db.session.delete(video)
        db.session.commit()
        
        flash(f'Video "{video.original_filename}" deleted successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting video: {str(e)}', 'error')
    
    return redirect(url_for('videos.index'))

@videos_bp.route('/<int:id>/reprocess', methods=['POST'])
@login_required
def reprocess(id):
    video = Video.query.get_or_404(id)
    
    if video.status in ['processing']:
        flash('Video is already being processed', 'warning')
        return redirect(url_for('videos.detail', id=id))
    
    # Reset status and clear previous results
    video.status = 'uploaded'
    video.processing_progress = 0
    video.error_message = None
    
    # Clear previous detections
    DetectedPerson.query.filter_by(video_id=id).delete()
    
    try:
        db.session.commit()
        
        # Queue for processing (would integrate with Celery here)
        # process_video.delay(video.id)
        
        flash('Video queued for reprocessing', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error reprocessing video: {str(e)}', 'error')
    
    return redirect(url_for('videos.detail', id=id))

@videos_bp.route('/api/upload', methods=['POST'])
@login_required
def api_upload():
    """API endpoint for AJAX file uploads"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file type'}), 400
    
    try:
        # Generate unique filename
        filename = secure_filename(file.filename)
        name, ext = os.path.splitext(filename)
        unique_filename = f"{name}_{uuid.uuid4().hex[:8]}{ext}"
        
        # Save file
        upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], 'videos')
        os.makedirs(upload_path, exist_ok=True)
        filepath = os.path.join(upload_path, unique_filename)
        file.save(filepath)
        
        # Create database record
        video = Video(
            filename=unique_filename,
            original_filename=filename,
            filepath=filepath,
            file_size=os.path.getsize(filepath),
            status='uploaded'
        )
        
        db.session.add(video)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'video_id': video.id,
            'filename': video.original_filename,
            'redirect_url': url_for('videos.detail', id=video.id)
        })
        
    except Exception as e:
        db.session.rollback()
        # Clean up file if database insert fails
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'success': False, 'error': str(e)}), 500

@videos_bp.route('/api/<int:id>/persons')
@login_required
def api_persons(id):
    """API endpoint to get detected persons for a video"""
    video = Video.query.get_or_404(id)
    detected_persons = DetectedPerson.query.filter_by(video_id=id).all()
    
    return jsonify([person.to_dict() for person in detected_persons])

@videos_bp.route('/api/processing-status')
@login_required
def api_processing_status():
    """API endpoint to get current processing status"""
    processing_videos = Video.query.filter_by(status='processing').all()
    
    return jsonify([{
        'id': video.id,
        'filename': video.original_filename,
        'progress': video.processing_progress,
        'status': video.status
    } for video in processing_videos])

def allowed_file(filename):
    """Check if file extension is allowed"""
    if '.' not in filename:
        return False
    
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in current_app.config['ALLOWED_VIDEO_EXTENSIONS']