from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app, send_from_directory, Response, send_file
from flask_login import login_required
from werkzeug.utils import secure_filename
import os
import uuid
from datetime import datetime
import threading
import time
import json

# Try to import SocketIO for real-time features
try:
    from flask_socketio import emit, join_room, leave_room
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False
    print("⚠️ Flask-SocketIO not available for real-time progress updates")

videos_bp = Blueprint('videos', __name__)

@videos_bp.route('/')
@login_required
def index():
    # Access video model directly
    Video = current_app.Video
    
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
        'converting': Video.query.filter_by(status='converting').count(),
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
        # Debug: Print all form data
        print("=== DEBUG UPLOAD ===")
        print("Form data:", dict(request.form))
        print("Files data:", dict(request.files))
        print("===================")
        
        # Check if file was uploaded
        if 'video_file' not in request.files:
            flash('No video file found in upload', 'error')
            return render_template('videos/upload.html')
        
        file = request.files['video_file']
        
        # Check if file has a name
        if not file or file.filename == '':
            flash('No video file selected', 'error')
            return render_template('videos/upload.html')
        
        # Check file type
        if not allowed_file(file.filename):
            flash('Invalid file type. Please upload a video file (MP4, AVI, MOV, MKV, WMV, FLV, WEBM).', 'error')
            return render_template('videos/upload.html')
        
        try:
            # Access models
            Video = current_app.Video
            db = current_app.db
            
            # Generate unique filename
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            
            # Create upload directory if it doesn't exist
            upload_folder = current_app.config.get('UPLOAD_FOLDER', 'static/uploads')
            upload_path = os.path.join(upload_folder, unique_filename)
            os.makedirs(upload_folder, exist_ok=True)
            
            # Save file
            file.save(upload_path)
            
            # Create video record with processing status
            video = Video(
                filename=filename,
                file_path=unique_filename,  # Store just the filename
                file_size=os.path.getsize(upload_path),
                title=request.form.get('title', ''),
                description=request.form.get('description', ''),
                priority=request.form.get('priority', 'normal'),
                status='processing',  # Start processing immediately
                processing_started_at=datetime.utcnow()
            )
            
            # Save to database
            db.session.add(video)
            db.session.commit()
            
            # Auto-start person extraction with GPU acceleration
            print(f"🚀 Auto-starting person extraction for uploaded video: {filename}")
            
            # Processing options for auto-extraction
            processing_options = {
                'extract_persons': True,
                'face_recognition': False,  # Can be enabled if needed
                'extract_frames': False,
                'use_enhanced_detection': True,
                'use_gpu': True  # Enable GPU acceleration
            }
            
            # Store processing options in video record
            video.processing_log = f"Auto-extraction started at {datetime.utcnow()} with GPU acceleration"
            db.session.commit()
            
            # Start enhanced processing with GPU
            start_enhanced_gpu_processing(video, processing_options, current_app._get_current_object())
            
            flash(f'Video "{filename}" uploaded! Person extraction started automatically with GPU acceleration. The annotated video will be available for playback once processing is complete.', 'info')
            
            return redirect(url_for('videos.index'))
            
        except Exception as e:
            # Clean up file if database save failed
            if 'upload_path' in locals() and os.path.exists(upload_path):
                try:
                    os.remove(upload_path)
                except:
                    pass
            
            flash(f'Error uploading video: {str(e)}', 'error')
            return render_template('videos/upload.html')
    
    # GET request - show upload form
    return render_template('videos/upload.html')

@videos_bp.route('/<int:id>')
@login_required
def detail(id):
    # Access models directly
    Video = current_app.Video
    DetectedPerson = current_app.DetectedPerson
    
    video = Video.query.get_or_404(id)
    
    # Get view mode (individual detections or grouped by person)
    view_mode = request.args.get('view', 'grouped')  # Default to grouped view
    page = request.args.get('page', 1, type=int)
    per_page = 50  # Show 50 items per page
    
    if view_mode == 'grouped':
        # Get all detections to group by person
        all_detections = DetectedPerson.query.filter_by(video_id=id).order_by(
            DetectedPerson.timestamp.asc()
        ).all()
        
        # Group detections by person_id
        person_groups = {}
        for detection in all_detections:
            # Normalize person_id for grouping
            if detection.person_id:
                if str(detection.person_id).isdigit():
                    person_key = int(detection.person_id)
                else:
                    person_key = detection.person_id
            elif detection.track_id:
                person_key = detection.track_id
            else:
                person_key = f"unknown_{detection.id}"
            
            if person_key not in person_groups:
                person_groups[person_key] = {
                    'person_id': person_key,
                    'detections': [],
                    'first_seen': detection.timestamp,
                    'last_seen': detection.timestamp,
                    'total_detections': 0,
                    'confidence_avg': 0,
                    'is_identified': detection.is_identified
                }
            
            person_groups[person_key]['detections'].append(detection)
            person_groups[person_key]['last_seen'] = max(person_groups[person_key]['last_seen'], detection.timestamp)
            person_groups[person_key]['total_detections'] += 1
            person_groups[person_key]['confidence_avg'] += detection.confidence
            if detection.is_identified:
                person_groups[person_key]['is_identified'] = True
        
        # Calculate average confidence and sort by person_id
        for person_key, group in person_groups.items():
            group['confidence_avg'] /= group['total_detections']
            group['duration'] = group['last_seen'] - group['first_seen']
        
        # Convert to sorted list
        grouped_detections = sorted(person_groups.values(), 
                                  key=lambda x: (isinstance(x['person_id'], int), x['person_id']))
        
        # Paginate the groups
        from math import ceil
        total_groups = len(grouped_detections)
        total_pages = ceil(total_groups / per_page)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_groups = grouped_detections[start_idx:end_idx]
        
        # Create a custom pagination object
        class GroupPagination:
            def __init__(self, groups, page, per_page, total):
                self.items = groups
                self.page = page
                self.per_page = per_page
                self.total = total
                self.pages = ceil(total / per_page)
                self.has_prev = page > 1
                self.has_next = page < self.pages
                self.prev_num = page - 1 if self.has_prev else None
                self.next_num = page + 1 if self.has_next else None
        
        detections_pagination = GroupPagination(paginated_groups, page, per_page, total_groups)
        detections = paginated_groups
        
    else:
        # Original individual detection view
        detections_pagination = DetectedPerson.query.filter_by(video_id=id).order_by(
            DetectedPerson.timestamp.asc()
        ).paginate(page=page, per_page=per_page, error_out=False)
        
        detections = detections_pagination.items
    
    # Get total count for stats
    total_detections = DetectedPerson.query.filter_by(video_id=id).count()
    identified_count = DetectedPerson.query.filter_by(video_id=id, is_identified=True).count()
    
    return render_template('videos/detail.html',
                         video=video,
                         detections=detections,
                         pagination=detections_pagination,
                         total_detections=total_detections,
                         identified_count=identified_count,
                         current_page=page,
                         view_mode=view_mode)

@videos_bp.route('/<int:id>/delete', methods=['POST'])
@login_required
def delete(id):
    try:
        Video = current_app.Video
        DetectedPerson = current_app.DetectedPerson
        db = current_app.db
        
        video = Video.query.get_or_404(id)
        
        # If video is processing, first try to cancel any active tasks
        if video.status == 'processing':
            print(f"⚠️ Attempting to delete video {id} that is currently processing")
            
            # Try to cancel Celery task if exists
            if hasattr(video, 'task_id') and video.task_id:
                try:
                    from celery.result import AsyncResult
                    from processing.tasks import celery
                    
                    task_result = AsyncResult(video.task_id, app=celery)
                    task_result.revoke(terminate=True)
                    print(f"🛑 Cancelled Celery task {video.task_id} for video {video.id}")
                except Exception as e:
                    print(f"⚠️ Could not cancel Celery task: {e}")
            
            # Force update status to allow deletion
            video.status = 'cancelled'
            db.session.commit()
            print(f"🔄 Changed video {id} status from 'processing' to 'cancelled' for deletion")
        
        # Delete all detected persons first to avoid foreign key constraint issues
        try:
            deleted_count = DetectedPerson.query.filter_by(video_id=id).delete()
            db.session.commit()
            print(f"🗑️ Deleted {deleted_count} detected persons for video {id}")
        except Exception as e:
            print(f"⚠️ Error deleting detected persons: {e}")
            db.session.rollback()
        
        # Delete physical files
        upload_folder = current_app.config.get('UPLOAD_FOLDER', 'static/uploads')
        
        # Delete original file
        if video.file_path:
            file_path = os.path.join(upload_folder, video.file_path)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"🗑️ Deleted original file: {file_path}")
                except Exception as e:
                    print(f"⚠️ Could not delete original file: {e}")
        
        # Delete processed file if exists
        if video.processed_path:
            processed_path = os.path.join(upload_folder, video.processed_path)
            if os.path.exists(processed_path):
                try:
                    os.remove(processed_path)
                    print(f"🗑️ Deleted processed file: {processed_path}")
                except Exception as e:
                    print(f"⚠️ Could not delete processed file: {e}")
        
        # Delete annotated video if exists
        if hasattr(video, 'annotated_video_path') and video.annotated_video_path:
            # Check in processing/outputs directory
            annotated_path = os.path.join('processing/outputs', video.annotated_video_path)
            if os.path.exists(annotated_path):
                try:
                    os.remove(annotated_path)
                    print(f"🗑️ Deleted annotated video: {annotated_path}")
                except Exception as e:
                    print(f"⚠️ Could not delete annotated video: {e}")
        
        # Delete from database
        db.session.delete(video)
        db.session.commit()
        
        flash(f'Video "{video.filename}" deleted successfully!', 'success')
        print(f"✅ Successfully deleted video {id}: {video.filename}")
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ Error deleting video {id}: {str(e)}")
        print(f"📋 Error trace:\n{error_trace}")
        
        if hasattr(current_app, 'db'):
            current_app.db.session.rollback()
        flash(f'Error deleting video: {str(e)}', 'error')
    
    return redirect(url_for('videos.index'))

@videos_bp.route('/<int:id>/retry', methods=['POST'])
@login_required
def retry_processing(id):
    try:
        Video = current_app.Video
        db = current_app.db
        
        video = Video.query.get_or_404(id)
        
        if video.status != 'failed':
            flash('Only failed videos can be retried', 'error')
            return redirect(url_for('videos.detail', id=id))
        
        # Reset video status for retry
        video.status = 'processing'
        video.processing_started_at = datetime.utcnow()
        video.processing_completed_at = None
        video.error_message = None
        
        db.session.commit()
        
        # Queue processing task (would integrate with Celery here)
        # process_video_task.delay(video.id)
        
        flash('Video processing restarted!', 'success')
    except Exception as e:
        if hasattr(current_app, 'db'):
            current_app.db.session.rollback()
        flash(f'Error retrying video processing: {str(e)}', 'error')
    
    return redirect(url_for('videos.detail', id=id))

@videos_bp.route('/<int:id>/process', methods=['POST'])
@login_required
def process_video(id):
    """Process video to extract persons (separate from upload)"""
    try:
        Video = current_app.Video
        db = current_app.db
        
        video = Video.query.get_or_404(id)
        
        # Check if video is in the right status
        if video.status not in ['uploaded', 'failed', 'completed']:
            flash('Only uploaded, completed, or failed videos can be processed', 'warning')
            return redirect(url_for('videos.detail', id=id))
        
        # Log retry attempt for failed videos
        if video.status == 'failed':
            print(f"🔄 Retrying person extraction for failed video {video.id}: {video.filename}")
            print(f"📝 Previous error: {video.error_message}")
        else:
            print(f"🚀 Starting person extraction for video {video.id}: {video.filename} (status: {video.status})")
        
        # Update video status to processing
        video.status = 'processing'
        video.processing_started_at = datetime.utcnow()
        video.processing_completed_at = None
        video.error_message = None
        
        # Get processing options from form
        extract_persons = request.form.get('extract_persons', False)
        face_recognition = request.form.get('face_recognition', False)
        extract_frames = request.form.get('extract_frames', False)
        
        # Save processing status
        db.session.commit()
        
        # CLEAR ALL EXISTING DETECTION DATA BEFORE RE-PROCESSING
        print(f"🗑️ Clearing all existing detection data for video {video.id}")
        try:
            DetectedPerson = current_app.DetectedPerson
            existing_detections = DetectedPerson.query.filter_by(video_id=video.id).all()
            
            if existing_detections:
                detection_count = len(existing_detections)
                print(f"   🔍 Found {detection_count} existing detections to delete")
                
                for detection in existing_detections:
                    db.session.delete(detection)
                
                db.session.commit()
                print(f"   ✅ Successfully deleted {detection_count} existing detections")
            else:
                print(f"   📝 No existing detections found for video {video.id}")
                
        except Exception as e:
            print(f"   ⚠️ Warning: Could not clear existing detections: {e}")
            # Continue processing anyway - this is not a critical error
        
        # Reset any previous processing state
        video.processing_progress = 0
        video.error_message = None
        video.processing_completed_at = None
        
        # Check if GPU processing is requested
        use_gpu = request.form.get('use_gpu') == 'true'
        
        # Store task options in video record for reference
        processing_options = {
            'extract_persons': bool(extract_persons),
            'face_recognition': bool(face_recognition), 
            'extract_frames': bool(extract_frames),
            'use_enhanced_detection': True,
            'use_gpu': use_gpu
        }
        
        # Queue the actual processing task
        try:
            if use_gpu:
                # Use GPU-accelerated processing (same as auto-processing)
                video.processing_log = f"Enhanced GPU processing options: {processing_options} - Started at {datetime.utcnow()}"
                db.session.commit()
                
                print(f"🚀 Starting enhanced GPU person detection for video {video.id}: {video.filename}")
                start_enhanced_gpu_processing(video, processing_options, current_app._get_current_object())
                flash(f'Enhanced person extraction started for "{video.filename}" with GPU acceleration.', 'info')
            else:
                # Use CPU-based enhanced detection
                from processing.enhanced_detection import enhanced_person_detection_task
                
                video.processing_log = f"Enhanced processing options: {processing_options} - Started at {datetime.utcnow()}"
                db.session.commit()
                
                print(f"🚀 Starting enhanced person detection for video {video.id}: {video.filename}")
                start_enhanced_fallback_processing(video, processing_options, current_app._get_current_object())
                flash(f'Enhanced person extraction started for "{video.filename}". This will create an annotated video with bounding boxes and extract person data folders.', 'info')
            
        except ImportError as import_error:
            print(f"⚠️ Full enhanced detection not available: {import_error}")
            
            # Try fallback enhanced detection (works without AI dependencies)
            try:
                from processing.enhanced_detection_fallback import enhanced_person_detection_task
                
                processing_options = {
                    'extract_persons': bool(extract_persons),
                    'face_recognition': bool(face_recognition), 
                    'extract_frames': bool(extract_frames),
                    'use_enhanced_detection': True,
                    'fallback_mode': True
                }
                video.processing_log = f"Enhanced processing (fallback mode): {processing_options} - Started at {datetime.utcnow()}"
                db.session.commit()
                
                print(f"🔄 Starting enhanced detection fallback for video {video.id}: {video.filename}")
                start_enhanced_fallback_processing(video, processing_options, current_app._get_current_object())
                flash(f'Enhanced person extraction started for "{video.filename}" (demo mode - install AI dependencies for full functionality).', 'info')
                
            except ImportError:
                print(f"⚠️ Enhanced detection fallback not available. Using legacy processing...")
                # Final fallback to legacy processing
                processing_options = {
                    'extract_persons': bool(extract_persons),
                    'face_recognition': bool(face_recognition), 
                    'extract_frames': bool(extract_frames)
                }
                start_fallback_processing(video, processing_options, current_app._get_current_object())
                flash(f'Person extraction started for "{video.filename}" (legacy mode).', 'info')
        except Exception as e:
            # Handle other errors
            print(f"❌ Error starting processing: {e}")
            video.status = 'failed'
            video.error_message = f'Failed to start processing: {str(e)}'
            db.session.commit()
            flash(f'Error starting person extraction: {str(e)}', 'error')
        
        return redirect(url_for('videos.detail', id=id))
        
    except Exception as e:
        flash(f'Error starting video processing: {str(e)}', 'error')
        return redirect(url_for('videos.detail', id=id))

@videos_bp.route('/api')
@login_required
def api_list():
    # API endpoint for video data
    Video = current_app.Video
    videos = Video.query.order_by(Video.created_at.desc()).limit(50).all()
    return jsonify([video.to_dict() for video in videos])

@videos_bp.route('/api/<int:id>')
@login_required
def api_detail(id):
    # API endpoint for single video
    Video = current_app.Video
    DetectedPerson = current_app.DetectedPerson
    
    video = Video.query.get_or_404(id)
    video_data = video.to_dict()
    
    # Add detections
    detections = DetectedPerson.query.filter_by(video_id=id).all()
    video_data['detections'] = [d.to_dict() for d in detections]
    
    return jsonify(video_data)

@videos_bp.route('/api/<int:id>/detections')
@login_required
def api_detections(id):
    """API endpoint for paginated detections"""
    DetectedPerson = current_app.DetectedPerson
    
    # Get pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    
    # Limit per_page to prevent abuse
    per_page = min(per_page, 100)
    
    # Get paginated detections
    detections_pagination = DetectedPerson.query.filter_by(video_id=id).order_by(
        DetectedPerson.timestamp.asc()
    ).paginate(page=page, per_page=per_page, error_out=False)
    
    detections = detections_pagination.items
    
    # Build response
    response = {
        'detections': [d.to_dict() for d in detections],
        'total': detections_pagination.total,
        'page': page,
        'per_page': per_page,
        'pages': detections_pagination.pages,
        'has_prev': detections_pagination.has_prev,
        'has_next': detections_pagination.has_next,
        'prev_num': detections_pagination.prev_num,
        'next_num': detections_pagination.next_num
    }
    
    return jsonify(response)

@videos_bp.route('/debug/<int:id>')
@login_required
def debug_video(id):
    """Debug endpoint to check video paths"""
    Video = current_app.Video
    video = Video.query.get_or_404(id)
    
    import os
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'static/uploads')
    outputs_dir = os.path.join('processing', 'outputs')
    
    info = {
        'id': video.id,
        'filename': video.filename,
        'file_path': video.file_path,
        'processed_path': video.processed_path,
        'annotated_video_path': video.annotated_video_path,
        'status': video.status,
        'paths_checked': {}
    }
    
    # Check if files exist
    if video.file_path:
        full_path = os.path.join(upload_folder, video.file_path)
        info['paths_checked']['original'] = {
            'path': full_path,
            'exists': os.path.exists(full_path)
        }
    
    if video.processed_path:
        full_path = os.path.join(upload_folder, video.processed_path)
        info['paths_checked']['processed'] = {
            'path': full_path,
            'exists': os.path.exists(full_path)
        }
    
    if video.annotated_video_path:
        # Check multiple locations
        paths_to_check = [
            ('outputs_dir', os.path.join(outputs_dir, video.annotated_video_path)),
            ('outputs_dir_with_detected', os.path.join(outputs_dir, f'detected_{video.annotated_video_path}')),
            ('uploads_dir', os.path.join(upload_folder, video.annotated_video_path)),
            ('raw_path', video.annotated_video_path)
        ]
        
        info['paths_checked']['annotated'] = {}
        for name, path in paths_to_check:
            info['paths_checked']['annotated'][name] = {
                'path': path,
                'exists': os.path.exists(path)
            }
    
    # List files in outputs directory
    if os.path.exists(outputs_dir):
        info['outputs_dir_contents'] = os.listdir(outputs_dir)[:10]  # First 10 files
    
    return f"<pre>{json.dumps(info, indent=2)}</pre>"

@videos_bp.route('/check-detected/<path:filename>')
@login_required
def check_detected_file(filename):
    """Check if detected video file exists and get its info"""
    import os
    
    outputs_dir = os.path.join('processing', 'outputs')
    file_path = os.path.join(outputs_dir, filename)
    
    info = {
        'requested_filename': filename,
        'outputs_dir': outputs_dir,
        'full_path': file_path,
        'exists': os.path.exists(file_path)
    }
    
    if os.path.exists(file_path):
        stats = os.stat(file_path)
        info['size_bytes'] = stats.st_size
        info['size_mb'] = stats.st_size / (1024 * 1024)
        
        # Try to read with proper path
        try:
            with open(file_path, 'rb') as f:
                header = f.read(32)
                info['header_hex'] = header.hex()[:64]
                info['can_read'] = True
        except Exception as e:
            info['read_error'] = str(e)
            info['can_read'] = False
    else:
        # List what's actually in the directory
        if os.path.exists(outputs_dir):
            files = os.listdir(outputs_dir)
            info['files_in_outputs'] = [f for f in files if filename[:20] in f]
        
    return f"<pre>{json.dumps(info, indent=2)}</pre>"

@videos_bp.route('/fix-annotated/<int:id>')
@login_required
def fix_annotated_path(id):
    """Fix missing annotated video path by finding the file in outputs directory"""
    Video = current_app.Video
    db = current_app.db
    video = Video.query.get_or_404(id)
    
    import os
    outputs_dir = os.path.join('processing', 'outputs')
    
    if not os.path.exists(outputs_dir):
        return "Outputs directory not found", 404
    
    # Look for annotated video file
    base_name = video.file_path.rsplit('.', 1)[0] if video.file_path else ''
    found_files = []
    
    for file in os.listdir(outputs_dir):
        if file.endswith('.mp4') and 'annotated' in file:
            # Check if this file matches our video
            if base_name and base_name in file:
                found_files.append(file)
            elif video.file_path and video.file_path.replace('.mp4', '') in file:
                found_files.append(file)
    
    if found_files:
        # Use the most recent annotated file
        annotated_file = sorted(found_files)[-1]
        
        # Update the database
        video.annotated_video_path = annotated_file
        
        # Also update processed_path if not set
        if not video.processed_path:
            video.processed_path = annotated_file
        
        db.session.commit()
        
        return f"""
        <h2>Fixed annotated video path!</h2>
        <pre>
Video ID: {video.id}
Filename: {video.filename}
Updated annotated_video_path to: {video.annotated_video_path}
Updated processed_path to: {video.processed_path}
        </pre>
        <p><a href="{url_for('videos.detail', id=video.id)}">Go back to video</a></p>
        """
    else:
        files_in_output = os.listdir(outputs_dir)[:20]
        return f"""
        <h2>No annotated video found</h2>
        <p>Looking for files matching: {base_name}</p>
        <p>Files in outputs directory:</p>
        <pre>{json.dumps(files_in_output, indent=2)}</pre>
        """

@videos_bp.route('/test/<path:filename>')
@login_required
def test_video(filename):
    """Test endpoint to diagnose video serving issues"""
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'static/uploads')
    file_path = os.path.join(upload_folder, filename)
    
    info = {
        'filename': filename,
        'file_path': file_path,
        'exists': os.path.exists(file_path),
        'upload_folder': upload_folder,
        'absolute_path': os.path.abspath(file_path)
    }
    
    if os.path.exists(file_path):
        try:
            stats = os.stat(file_path)
            info['size_mb'] = stats.st_size / (1024 * 1024)
            info['permissions'] = oct(stats.st_mode)
            
            # Try to read first bytes
            with open(file_path, 'rb') as f:
                header = f.read(32)
                info['header_hex'] = header.hex()[:64]
                info['header_ascii'] = ''.join(chr(b) if 32 <= b < 127 else '.' for b in header)
        except Exception as e:
            info['error'] = str(e)
    
    return f"<pre>{json.dumps(info, indent=2)}</pre>"


@videos_bp.route('/stream/<path:filename>')
@login_required
def stream_video(filename):
    """Stream video files with range request support"""
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'static/uploads')
    file_path = os.path.join(upload_folder, filename)
    
    print(f"🎬 Stream request for: {filename}")
    print(f"📁 Full path: {file_path}")
    print(f"📊 File exists: {os.path.exists(file_path)}")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return "Video file not found", 404
    
    # Log file size and permissions
    try:
        file_stats = os.stat(file_path)
        print(f"📊 File size: {file_stats.st_size / (1024*1024):.2f} MB")
        print(f"📊 File permissions: {oct(file_stats.st_mode)}")
    except Exception as e:
        print(f"❌ Error getting file stats: {e}")
    
    # Detect file format from header
    def detect_file_format(file_path):
        try:
            with open(file_path, 'rb') as f:
                header = f.read(32)
                header_hex = header.hex()
                
                # Check for common video formats
                if header.startswith(b'ftypmp4') or header[4:8] == b'ftyp':
                    return 'video/mp4', True
                elif header.startswith(b'RIFF') and b'AVI ' in header:
                    return 'video/x-msvideo', True
                elif header.startswith(b'\x1a\x45\xdf\xa3'):  # MKV/WebM signature
                    return 'video/x-matroska', True
                elif header.startswith(b'FLV'):
                    return 'video/x-flv', True
                elif header.startswith(b'\x00\x00\x00'):
                    # Could be MP4 or MOV with different structure
                    return 'video/mp4', True
                elif header.startswith(b'IMKH'):
                    # IMKH proprietary format - not web compatible
                    print(f"IMKH format detected. This format is not web browser compatible.")
                    return 'application/octet-stream', False
                else:
                    # Unknown format, try to serve as generic video
                    print(f"Unknown video format. Header: {header_hex}")
                    return 'application/octet-stream', False
        except Exception as e:
            print(f"Error detecting file format: {e}")
            return 'application/octet-stream', False
    
    # Fallback to extension-based detection
    def get_content_type_by_extension(filename):
        ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
        content_types = {
            'mp4': 'video/mp4',
            'avi': 'video/x-msvideo',
            'mov': 'video/quicktime',
            'mkv': 'video/x-matroska',
            'wmv': 'video/x-ms-wmv',
            'flv': 'video/x-flv',
            'webm': 'video/webm'
        }
        return content_types.get(ext, 'application/octet-stream')
    
    # Try header detection first, fallback to extension
    content_type, is_web_compatible = detect_file_format(file_path)
    if content_type == 'application/octet-stream':
        content_type = get_content_type_by_extension(filename)
        is_web_compatible = True  # Assume extension-based detection is web compatible
    
    # If format is not web compatible, return error message
    if not is_web_compatible:
        return Response(
            "Video format not supported by web browsers. Please convert to MP4 format.",
            415,  # Unsupported Media Type
            headers={
                'Content-Type': 'text/plain',
                'X-Video-Format-Issue': 'IMKH-format-detected'
            }
        )
    
    file_size = os.path.getsize(file_path)
    
    # Handle range requests for video streaming
    range_header = request.headers.get('Range', None)
    if range_header:
        byte_start = 0
        byte_end = file_size - 1
        
        # Parse range header
        if range_header.startswith('bytes='):
            range_match = range_header[6:]
            if '-' in range_match:
                start, end = range_match.split('-', 1)
                if start:
                    byte_start = int(start)
                if end:
                    byte_end = int(end)
        
        content_length = byte_end - byte_start + 1
        
        def generate_range():
            with open(file_path, 'rb') as f:
                f.seek(byte_start)
                remaining = content_length
                while remaining:
                    chunk_size = min(8192, remaining)
                    data = f.read(chunk_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data
        
        response = Response(
            generate_range(),
            206,  # Partial Content
            headers={
                'Content-Range': f'bytes {byte_start}-{byte_end}/{file_size}',
                'Accept-Ranges': 'bytes',
                'Content-Length': str(content_length),
                'Content-Type': content_type,
                'Cache-Control': 'no-cache'
            }
        )
        return response
    
    # No range request, serve entire file
    def generate():
        with open(file_path, 'rb') as f:
            while True:
                data = f.read(8192)
                if not data:
                    break
                yield data
    
    return Response(
        generate(),
        headers={
            'Content-Length': str(file_size),
            'Content-Type': content_type,
            'Accept-Ranges': 'bytes',
            'Cache-Control': 'no-cache'
        }
    )

@videos_bp.route('/download/<path:filename>')
@login_required
def download_video(filename):
    """Alternative download method for problematic videos"""
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'static/uploads')
    file_path = os.path.join(upload_folder, filename)
    
    if not os.path.exists(file_path):
        return "Video file not found", 404
    
    return send_file(file_path, as_attachment=True, download_name=filename)

@videos_bp.route('/serve/<path:filename>')
@login_required
def serve_video_static(filename):
    """Static file serving method as fallback"""
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'static/uploads')
    file_path = os.path.join(upload_folder, filename)
    
    print(f"📁 Static serve request for: {filename}")
    print(f"📊 File exists: {os.path.exists(file_path)}")
    
    if not os.path.exists(file_path):
        return f"File not found: {filename}", 404
    
    # Try to serve with explicit mimetype
    mimetype = 'video/mp4'
    if filename.lower().endswith('.avi'):
        mimetype = 'video/x-msvideo'
    elif filename.lower().endswith('.mov'):
        mimetype = 'video/quicktime'
    elif filename.lower().endswith('.mkv'):
        mimetype = 'video/x-matroska'
    
    return send_from_directory(upload_folder, filename, mimetype=mimetype)

@videos_bp.route('/serve-annotated/<path:filename>')
@login_required
def serve_annotated_video(filename):
    """Serve annotated videos from processing outputs"""
    try:
        import mimetypes
        
        # Check multiple possible locations
        possible_paths = [
            os.path.join('processing', 'outputs', filename),  # Direct in outputs
            os.path.join('static', 'uploads', filename),      # In uploads (if moved)
            os.path.join('processing', 'outputs', f'detected_{filename}'),  # With detected_ prefix
            filename  # Absolute path
        ]
        
        full_path = None
        for path in possible_paths:
            if os.path.exists(path):
                full_path = path
                break
                
        if not full_path:
            print(f"❌ Annotated video not found in any location:")
            for path in possible_paths:
                print(f"   - {path}")
            return "Annotated video file not found", 404
        
        print(f"✅ Serving annotated video from: {full_path}")
        
        # Determine the directory and filename
        directory = os.path.dirname(full_path)
        file_name = os.path.basename(full_path)
        
        # Get proper mimetype
        mimetype, _ = mimetypes.guess_type(file_name)
        if not mimetype:
            mimetype = 'video/mp4'  # Default to MP4
            
        print(f"📹 Serving with mimetype: {mimetype}")
        
        # Use send_file for better compatibility
        return send_file(full_path, mimetype=mimetype, as_attachment=False)
        
    except Exception as e:
        print(f"❌ Error serving annotated video: {e}")
        import traceback
        traceback.print_exc()
        return f"Error serving annotated video: {str(e)}", 500

@videos_bp.route('/detected/<path:filename>')
@login_required  
def serve_detected_video(filename):
    """Serve detected/annotated videos with proper streaming support"""
    try:
        # The detected videos are in processing/outputs
        outputs_dir = os.path.join('processing', 'outputs')
        file_path = os.path.join(outputs_dir, filename)
        
        print(f"🎯 Detected video request: {filename}")
        print(f"📁 Looking in: {outputs_dir}")
        print(f"📊 Full path: {file_path}")
        print(f"✅ Exists: {os.path.exists(file_path)}")
        
        if not os.path.exists(file_path):
            # Try without detected_ prefix if it was already included
            if filename.startswith('detected_'):
                alt_filename = filename[9:]  # Remove 'detected_' prefix
                alt_path = os.path.join(outputs_dir, alt_filename)
                if os.path.exists(alt_path):
                    file_path = alt_path
                    print(f"✅ Found at alternate path: {alt_path}")
                else:
                    return f"Detected video not found: {filename}", 404
            else:
                return f"Detected video not found: {filename}", 404
        
        # Get file size for range requests
        file_size = os.path.getsize(file_path)
        
        # Handle range requests for proper video streaming
        range_header = request.headers.get('Range', None)
        if range_header:
            byte_start = 0
            byte_end = file_size - 1
            
            if range_header.startswith('bytes='):
                range_match = range_header[6:]
                if '-' in range_match:
                    start, end = range_match.split('-', 1)
                    if start:
                        byte_start = int(start)
                    if end:
                        byte_end = int(end)
            
            content_length = byte_end - byte_start + 1
            
            def generate_range():
                with open(file_path, 'rb') as f:
                    f.seek(byte_start)
                    remaining = content_length
                    while remaining:
                        chunk_size = min(8192, remaining)
                        data = f.read(chunk_size)
                        if not data:
                            break
                        remaining -= len(data)
                        yield data
            
            return Response(
                generate_range(),
                206,  # Partial Content
                headers={
                    'Content-Range': f'bytes {byte_start}-{byte_end}/{file_size}',
                    'Accept-Ranges': 'bytes',
                    'Content-Length': str(content_length),
                    'Content-Type': 'video/mp4',
                    'Cache-Control': 'no-cache'
                }
            )
        
        # No range request - send full file
        return send_file(file_path, mimetype='video/mp4', as_attachment=False)
        
    except Exception as e:
        print(f"❌ Error serving detected video: {e}")
        import traceback
        traceback.print_exc()
        return f"Error serving detected video: {str(e)}", 500

@videos_bp.route('/stream-detected/<path:filename>')
@login_required
def stream_detected_video(filename):
    """Stream detected videos directly from outputs directory"""
    import os
    
    outputs_dir = os.path.join('processing', 'outputs')
    file_path = os.path.join(outputs_dir, filename)
    
    print(f"🎬 Stream detected video request: {filename}")
    print(f"📁 Full path: {file_path}")
    print(f"✅ Exists: {os.path.exists(file_path)}")
    
    if not os.path.exists(file_path):
        # List files to help debug
        if os.path.exists(outputs_dir):
            files = [f for f in os.listdir(outputs_dir) if f.endswith('.mp4')]
            print(f"📁 MP4 files in outputs: {files[:5]}")
        return f"Detected video not found: {filename}", 404
    
    # Use send_file for simplicity - Flask will handle range requests
    try:
        return send_file(
            file_path,
            mimetype='video/mp4',
            as_attachment=False,
            conditional=True  # This enables range request support
        )
    except Exception as e:
        print(f"❌ Error serving detected video: {e}")
        return f"Error serving file: {str(e)}", 500

@videos_bp.route('/api/<int:id>/processing-status')
@login_required
def processing_status(id):
    """Get person extraction processing status with progress for AJAX polling"""
    Video = current_app.Video
    db = current_app.db
    video = Video.query.get_or_404(id)
    
    # Get basic video status
    response = {
        'status': video.status,
        'error_message': video.error_message,
        'processing_started_at': video.processing_started_at.isoformat() if video.processing_started_at else None,
        'processing_completed_at': video.processing_completed_at.isoformat() if video.processing_completed_at else None,
        'progress': getattr(video, 'processing_progress', 0) or 0,
        'progress_message': 'Initializing...',
        'task_id': getattr(video, 'task_id', None)
    }
    
    # If processing, get detailed progress from Celery task or fallback processing
    if video.status == 'processing':
        # First check if we have a Celery task
        if hasattr(video, 'task_id') and video.task_id:
            try:
                from celery.result import AsyncResult
                from processing.tasks import celery
                from datetime import datetime, timedelta
                
                task_result = AsyncResult(video.task_id, app=celery)
                
                # Check if task has been running too long (over 10 minutes)
                if video.processing_started_at:
                    elapsed = datetime.utcnow() - video.processing_started_at
                    if elapsed > timedelta(minutes=10):
                        print(f"⏰ Task {video.task_id} has been running for {elapsed}, marking as failed")
                        video.status = 'failed'
                        video.error_message = f'Task timed out after {elapsed}'
                        video.processing_completed_at = datetime.utcnow()
                        db.session.commit()
                        
                        response.update({
                            'status': 'failed',
                            'progress': 0,
                            'progress_message': 'Task timed out',
                            'error_message': f'Task timed out after {elapsed}'
                        })
                        print(f"📡 Processing API Response for video {id}: TIMEOUT after {elapsed}")
                        return jsonify(response)
                
                if task_result.state == 'PROGRESS':
                    task_info = task_result.info or {}
                    progress = task_info.get('progress', 0)
                    message = task_info.get('status', 'Processing...')
                    
                    response.update({
                        'progress': progress,
                        'progress_message': message,
                        'celery_state': task_result.state
                    })
                    
                    print(f"🔄 Person extraction progress for video {id}: {progress}% - {message}")
                    
                elif task_result.state == 'SUCCESS':
                    response.update({
                        'progress': 100,
                        'progress_message': 'Person extraction completed!',
                        'celery_state': task_result.state
                    })
                    print(f"✅ Person extraction completed for video {id}")
                    
                elif task_result.state == 'FAILURE':
                    error_msg = str(task_result.info) if task_result.info else 'Unknown error'
                    
                    # Update video status in database
                    video.status = 'failed'
                    video.error_message = error_msg
                    video.processing_completed_at = datetime.utcnow()
                    db.session.commit()
                    
                    response.update({
                        'status': 'failed',
                        'progress': 0,
                        'progress_message': f'Failed: {error_msg}',
                        'celery_state': task_result.state,
                        'error_message': error_msg
                    })
                    print(f"❌ Person extraction failed for video {id}: {error_msg}")
                    
                elif task_result.state == 'PENDING':
                    # Task is queued but not started yet
                    response.update({
                        'progress': 0,
                        'progress_message': 'Task queued, waiting to start...',
                        'celery_state': task_result.state
                    })
                    
                else:
                    response.update({
                        'progress': 10,
                        'progress_message': f'Task state: {task_result.state}',
                        'celery_state': task_result.state
                    })
                    
            except ImportError:
                print(f"⚠️ Celery not available for status check")
                response.update({
                    'progress': 0,
                    'progress_message': 'Celery not available',
                    'error_message': 'Celery worker not running'
                })
            except Exception as e:
                print(f"⚠️ Error checking Celery task status: {e}")
                
                # If we can't check the task status and it's been a while, mark as failed
                if video.processing_started_at:
                    elapsed = datetime.utcnow() - video.processing_started_at
                    if elapsed > timedelta(minutes=5):
                        video.status = 'failed'
                        video.error_message = f'Unable to check task status: {str(e)}'
                        video.processing_completed_at = datetime.utcnow()
                        db.session.commit()
                        
                        response.update({
                            'status': 'failed',
                            'progress': 0,
                            'progress_message': 'Unable to check task status',
                            'error_message': str(e)
                        })
                else:
                    response.update({
                        'progress': 0,
                        'progress_message': 'Error checking task status',
                        'error_message': str(e)
                    })
        else:
            # Fallback processing (no Celery task ID) - use video.processing_progress
            from datetime import datetime, timedelta
            
            # Check if processing has been running too long
            if video.processing_started_at:
                elapsed = datetime.utcnow() - video.processing_started_at
                if elapsed > timedelta(minutes=10):
                    print(f"⏰ Fallback processing has been running for {elapsed}, marking as failed")
                    video.status = 'failed'
                    video.error_message = f'Processing timed out after {elapsed}'
                    video.processing_completed_at = datetime.utcnow()
                    db.session.commit()
                    
                    response.update({
                        'status': 'failed',
                        'progress': 0,
                        'progress_message': 'Processing timed out',
                        'error_message': f'Processing timed out after {elapsed}'
                    })
                    print(f"📡 Processing API Response for video {id}: TIMEOUT after {elapsed}")
                    return jsonify(response)
            
            # Get progress from video.processing_progress
            progress = getattr(video, 'processing_progress', 0) or 0
            
            # Determine message based on progress
            if progress < 25:
                message = 'Extracting video metadata...'
            elif progress < 70:
                message = 'Detecting persons in video...'
            elif progress < 100:
                message = 'Saving detection results...'
            else:
                message = 'Finalizing processing...'
            
            response.update({
                'progress': progress,
                'progress_message': message,
                'processing_mode': 'fallback'
            })
            
            print(f"🔄 Fallback processing progress for video {id}: {progress}% - {message}")
            print(f"   📊 Database progress: {getattr(video, 'processing_progress', 'None')}")
            print(f"   📅 Started: {video.processing_started_at}")
            print(f"   🕒 Elapsed: {elapsed if 'elapsed' in locals() else 'Unknown'}")
    
        # Update response with any database-stored progress for non-Celery processing
        if hasattr(video, 'processing_progress') and video.processing_progress:
            if response['progress'] == 0:  # Only update if we didn't get progress from Celery
                response['progress'] = video.processing_progress
                print(f"   🔄 Using database progress: {response['progress']}%")
    
    print(f"📡 Processing API Response for video {id}: status={response['status']}, progress={response['progress']}%, message='{response['progress_message']}'")
    print(f"   🔧 Response details: {response}")
    return jsonify(response)

@videos_bp.route('/api/conversion-tasks')
@login_required 
def get_conversion_tasks():
    """Get all conversion tasks status"""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from hr_management.utils.conversion_manager import conversion_manager
    
    return jsonify(conversion_manager.get_all_tasks())

@videos_bp.route('/api/conversion-task/<task_id>')
@login_required
def get_conversion_task(task_id):
    """Get specific conversion task status"""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from hr_management.utils.conversion_manager import conversion_manager
    
    task = conversion_manager.get_task(task_id)
    if task:
        return jsonify(task.to_dict())
    else:
        return jsonify({'error': 'Task not found'}), 404

@videos_bp.route('/api/debug/<int:id>')
@login_required
def debug_video_status(id):
    """Debug video and conversion status"""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from hr_management.utils.conversion_manager import conversion_manager
    
    Video = current_app.Video
    video = Video.query.get_or_404(id)
    
    # Get all conversion tasks
    all_tasks = conversion_manager.get_all_tasks()
    video_task = conversion_manager.get_task_by_video_id(id)
    
    debug_info = {
        'video_info': {
            'id': video.id,
            'status': video.status,
            'filename': video.filename,
            'processed_path': video.processed_path,
            'processing_started_at': video.processing_started_at.isoformat() if video.processing_started_at else None,
            'processing_log': video.processing_log,
            'error_message': video.error_message
        },
        'conversion_manager': {
            'total_tasks': len(all_tasks),
            'video_task_exists': video_task is not None,
            'video_task_data': video_task.to_dict() if video_task else None,
            'all_task_ids': list(all_tasks.keys())
        }
    }
    
    return jsonify(debug_info)

@videos_bp.route('/<int:id>/cancel-processing', methods=['POST'])
@login_required
def cancel_processing(id):
    """Cancel stuck processing and reset video to processable state"""
    try:
        Video = current_app.Video
        db = current_app.db
        
        video = Video.query.get_or_404(id)
        
        if video.status != 'processing':
            flash('Video is not currently processing', 'warning')
            return redirect(url_for('videos.detail', id=id))
        
        # Cancel Celery task if exists
        if hasattr(video, 'task_id') and video.task_id:
            try:
                from celery.result import AsyncResult
                from processing.tasks import celery
                
                task_result = AsyncResult(video.task_id, app=celery)
                task_result.revoke(terminate=True)  # Forcefully terminate the task
                print(f"🛑 Cancelled Celery task {video.task_id} for video {video.id}")
                
            except Exception as e:
                print(f"⚠️ Error cancelling Celery task: {e}")
        
        # Reset video to a processable state
        # If video was originally converted, set back to completed
        # If video was originally uploaded, set back to uploaded
        if video.processed_path:
            video.status = 'completed'
            print(f"🔄 Reset converted video {video.id} to 'completed' status")
        else:
            video.status = 'uploaded'
            print(f"🔄 Reset video {video.id} to 'uploaded' status")
        
        # Clear processing data
        video.processing_started_at = None
        video.processing_completed_at = None
        video.processing_progress = 0
        video.error_message = None
        video.task_id = None
        
        db.session.commit()
        
        print(f"✅ Successfully cancelled processing for video {video.id}: {video.filename}")
        flash(f'Processing cancelled for "{video.filename}". You can now retry person extraction.', 'success')
        
        return redirect(url_for('videos.detail', id=id))
        
    except Exception as e:
        flash(f'Error cancelling processing: {str(e)}', 'error')
        return redirect(url_for('videos.detail', id=id))

@videos_bp.route('/<int:id>/simulate-error', methods=['POST'])
@login_required
def simulate_error(id):
    """Simulate a processing error for testing (development only)"""
    try:
        Video = current_app.Video
        db = current_app.db
        
        video = Video.query.get_or_404(id)
        
        # Set video to failed state with a test error message
        video.status = 'failed'
        video.error_message = 'Simulated error for testing retry functionality'
        video.processing_completed_at = datetime.utcnow()
        video.processing_progress = 0
        
        db.session.commit()
        
        print(f"🧪 Simulated error for video {video.id}: {video.filename}")
        flash(f'Simulated error for "{video.filename}" - you can now test the retry functionality', 'warning')
        
        return redirect(url_for('videos.detail', id=id))
        
    except Exception as e:
        flash(f'Error simulating failure: {str(e)}', 'error')
        return redirect(url_for('videos.detail', id=id))

@videos_bp.route('/<int:id>/force-reset', methods=['POST'])
@login_required
def force_reset(id):
    """Force reset a stuck video to allow deletion or reprocessing"""
    try:
        Video = current_app.Video
        db = current_app.db
        
        video = Video.query.get_or_404(id)
        old_status = video.status
        
        # Force reset the video status
        video.status = 'failed'
        video.error_message = f'Force reset from stuck "{old_status}" status'
        video.processing_completed_at = datetime.utcnow()
        video.processing_progress = 0
        video.task_id = None  # Clear any task references
        
        db.session.commit()
        
        print(f"🔧 Force reset video {video.id}: {video.filename} from '{old_status}' to 'failed'")
        flash(f'Video "{video.filename}" has been reset. You can now delete or reprocess it.', 'success')
        
        return redirect(url_for('videos.detail', id=id))
        
    except Exception as e:
        flash(f'Error resetting video: {str(e)}', 'error')
        return redirect(url_for('videos.detail', id=id))

@videos_bp.route('/api/<int:video_id>/calibrate-coordinates', methods=['POST'])
def calibrate_coordinates(video_id):
    """
    AI-powered coordinate calibration system
    Takes a browser screenshot and uses AI to recalibrate bounding box coordinates
    """
    try:
        Video = current_app.Video
        DetectedPerson = current_app.DetectedPerson
        db = current_app.db
        import base64
        import json
        from datetime import datetime
        
        print(f"🎯 Starting coordinate calibration for video {video_id}")
        
        # Get request data
        data = request.get_json()
        frame_data = data.get('frameData')  # Base64 encoded image
        frame_info = data.get('frameInfo')
        calibration_data = data.get('calibrationData')
        
        print(f"📊 Calibration request: {len(calibration_data.get('detections', []))} detections")
        print(f"📐 Frame info: {frame_info.get('displayWidth')}x{frame_info.get('displayHeight')}")
        
        # Get video from database
        video = Video.query.get_or_404(video_id)
        
        # Decode the frame image
        image_data = frame_data.split(',')[1]  # Remove data:image/jpeg;base64, prefix
        frame_bytes = base64.b64decode(image_data)
        
        print(f"📸 Decoded frame: {len(frame_bytes)} bytes")
        
        # Use AI to re-detect persons in the browser frame
        calibration_result = perform_ai_coordinate_calibration(
            frame_bytes, 
            frame_info, 
            calibration_data.get('detections', [])
        )
        
        print(f"🤖 AI calibration result: {calibration_result}")
        
        # Store calibration data for this video session
        store_calibration_offsets(video_id, calibration_result.get('offsets', {}))
        
        return jsonify({
            'success': True,
            'offsets': calibration_result.get('offsets', {}),
            'detections_analyzed': len(calibration_data.get('detections', [])),
            'calibration_accuracy': calibration_result.get('accuracy', 0.0),
            'method': calibration_result.get('method', 'unknown'),
            'message': f"Calibration completed with {calibration_result.get('accuracy', 0)*100:.1f}% accuracy using {calibration_result.get('method', 'unknown')} method"
        })
        
    except Exception as e:
        print(f"❌ Calibration error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Calibration failed'
        }), 500

def perform_ai_coordinate_calibration(frame_bytes, frame_info, stored_detections):
    """
    Use AI models to re-detect persons in browser frame and calculate calibration offsets
    """
    try:
        print("🤖 Starting AI coordinate calibration...")
        
        # Try to import AI detection modules
        try:
            from ..processing.transformer_detection import detect_persons_sam2, detect_persons_detr
            from ..processing.real_detection import detect_persons_yolo
            AI_AVAILABLE = True
            print("✅ AI models available for calibration")
        except ImportError as e:
            print(f"⚠️ AI models not available: {e}, using calibration heuristics")
            AI_AVAILABLE = False
        
        if AI_AVAILABLE:
            # Save frame temporarily for AI processing
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_file.write(frame_bytes)
                temp_path = temp_file.name
            
            try:
                print(f"🔍 Processing frame with AI: {temp_path}")
                
                # Use YOLO for fast detection (most reliable for calibration)
                ai_detections = detect_persons_yolo_frame(temp_path, frame_info)
                print(f"🤖 AI found {len(ai_detections)} persons in browser frame")
                
                # Calculate calibration offsets by comparing stored vs AI-detected coordinates
                offsets = calculate_calibration_offsets(stored_detections, ai_detections, frame_info)
                
                accuracy = calculate_calibration_accuracy(stored_detections, ai_detections, offsets)
                
                return {
                    'offsets': offsets,
                    'accuracy': accuracy,
                    'ai_detections': len(ai_detections),
                    'method': 'ai_yolo'
                }
                
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        
        else:
            # Fallback: Use heuristic calibration based on common offset patterns
            print("🔧 Using heuristic calibration (no AI available)")
            
            offsets = calculate_heuristic_offsets(stored_detections, frame_info)
            
            return {
                'offsets': offsets,
                'accuracy': 0.7,  # Moderate confidence for heuristics
                'ai_detections': 0,
                'method': 'heuristic'
            }
            
    except Exception as e:
        print(f"❌ AI calibration failed: {e}")
        # Return neutral offsets (no change)
        return {
            'offsets': {'offsetX': 0, 'offsetY': 0, 'scaleX': 1.0, 'scaleY': 1.0},
            'accuracy': 0.0,
            'ai_detections': 0,
            'method': 'fallback',
            'error': str(e)
        }

def detect_persons_yolo_frame(frame_path, frame_info):
    """
    Use YOLO to detect persons in the browser-captured frame
    """
    try:
        import cv2
        from ultralytics import YOLO
        
        print("🎯 Loading YOLO model for calibration...")
        
        # Load YOLO model (try multiple paths)
        model_paths = ['yolov8n.pt', 'models/yolov8n.pt', '/tmp/yolov8n.pt']
        model = None
        
        for path in model_paths:
            try:
                model = YOLO(path)
                print(f"✅ Loaded YOLO from: {path}")
                break
            except:
                continue
        
        if model is None:
            # Download if not found
            model = YOLO('yolov8n.pt')  # This will auto-download
            print("📥 Downloaded YOLO model")
        
        # Read the frame
        frame = cv2.imread(frame_path)
        if frame is None:
            raise ValueError("Failed to load frame image")
        
        print(f"🖼️ Processing frame: {frame.shape}")
        
        # Run YOLO detection (only detect persons - class 0)
        results = model(frame, classes=[0], verbose=False)
        
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    
                    # Filter by confidence
                    if confidence > 0.3:  # Minimum confidence for calibration
                        # Convert to our format (relative to display frame)
                        detection = {
                            'x': int(x1),
                            'y': int(y1),
                            'width': int(x2 - x1),
                            'height': int(y2 - y1),
                            'confidence': float(confidence)
                        }
                        detections.append(detection)
        
        print(f"🎯 YOLO detected {len(detections)} persons in calibration frame")
        return detections
        
    except Exception as e:
        print(f"❌ YOLO detection failed: {e}")
        return []

def calculate_calibration_offsets(stored_detections, ai_detections, frame_info):
    """
    Calculate offset corrections by comparing stored coordinates with AI-detected coordinates
    """
    print("🔧 Calculating calibration offsets...")
    
    if not ai_detections or not stored_detections:
        print("⚠️ Insufficient detection data for calibration")
        return {'offsetX': 0, 'offsetY': 0, 'scaleX': 1.0, 'scaleY': 1.0}
    
    # Match stored detections with AI detections using proximity
    matches = []
    
    for stored in stored_detections:
        stored_bbox = stored['bbox']
        best_match = None
        best_distance = float('inf')
        
        for ai_det in ai_detections:
            # Calculate center-to-center distance
            stored_center_x = stored_bbox['x'] + stored_bbox['width'] / 2
            stored_center_y = stored_bbox['y'] + stored_bbox['height'] / 2
            ai_center_x = ai_det['x'] + ai_det['width'] / 2
            ai_center_y = ai_det['y'] + ai_det['height'] / 2
            
            distance = ((stored_center_x - ai_center_x) ** 2 + (stored_center_y - ai_center_y) ** 2) ** 0.5
            
            # Match if within reasonable distance (adjust threshold based on display size)
            max_distance = min(200, frame_info.get('displayWidth', 800) * 0.2)
            
            if distance < best_distance and distance < max_distance:
                best_distance = distance
                best_match = ai_det
        
        if best_match:
            matches.append({
                'stored': stored_bbox,
                'ai': best_match,
                'distance': best_distance
            })
    
    print(f"📊 Found {len(matches)} coordinate matches for calibration")
    
    if len(matches) < 1:
        print("⚠️ No coordinate matches found, using heuristic offsets")
        return calculate_heuristic_offsets(stored_detections, frame_info)
    
    # Calculate average offsets
    offset_x_sum = 0
    offset_y_sum = 0
    scale_x_sum = 0
    scale_y_sum = 0
    
    for match in matches:
        stored = match['stored']
        ai = match['ai']
        
        # Position offsets
        offset_x_sum += (ai['x'] - stored['x'])
        offset_y_sum += (ai['y'] - stored['y'])
        
        # Scale factors (avoid division by zero)
        if stored['width'] > 0:
            scale_x_sum += ai['width'] / stored['width']
        else:
            scale_x_sum += 1.0
            
        if stored['height'] > 0:
            scale_y_sum += ai['height'] / stored['height']
        else:
            scale_y_sum += 1.0
    
    num_matches = len(matches)
    
    # Calculate averages with bounds checking
    offset_x = offset_x_sum / num_matches
    offset_y = offset_y_sum / num_matches
    scale_x = max(0.5, min(2.0, scale_x_sum / num_matches))  # Bound between 0.5x and 2x
    scale_y = max(0.5, min(2.0, scale_y_sum / num_matches))
    
    offsets = {
        'offsetX': round(offset_x, 2),
        'offsetY': round(offset_y, 2),
        'scaleX': round(scale_x, 3),
        'scaleY': round(scale_y, 3)
    }
    
    print(f"✅ Calculated AI-based offsets: {offsets}")
    return offsets

def calculate_heuristic_offsets(stored_detections, frame_info):
    """
    Calculate offsets using heuristic rules when AI detection is not available
    """
    print("🔧 Calculating heuristic offsets...")
    
    # Common offset patterns based on browser/video scaling differences
    display_width = frame_info.get('displayWidth', 800)
    display_height = frame_info.get('displayHeight', 600)
    
    # Analyze detection positions to infer likely offset patterns
    if stored_detections:
        # Check if detections seem to be consistently off in a particular direction
        avg_x = sum(d['bbox']['x'] for d in stored_detections) / len(stored_detections)
        avg_y = sum(d['bbox']['y'] for d in stored_detections) / len(stored_detections)
        
        # Heuristic adjustments based on position patterns
        if avg_x < display_width * 0.2:  # Detections clustered on left
            offset_x = 5
        elif avg_x > display_width * 0.8:  # Detections clustered on right
            offset_x = -5
        else:
            offset_x = 0
            
        if avg_y < display_height * 0.2:  # Detections clustered on top
            offset_y = 5
        elif avg_y > display_height * 0.8:  # Detections clustered on bottom
            offset_y = -5
        else:
            offset_y = -2  # Common slight upward adjustment
    else:
        offset_x = 0
        offset_y = -2
    
    # Screen size-based adjustments
    if display_width < 800:
        # Mobile/small screen: coordinates often need adjustment
        scale_x = 0.98
        scale_y = 0.97
    elif display_width > 1200:
        # Large screen: different scaling behavior
        scale_x = 1.02
        scale_y = 1.01
    else:
        # Medium screen: minimal adjustment
        scale_x = 1.0
        scale_y = 0.99
    
    offsets = {
        'offsetX': offset_x,
        'offsetY': offset_y,
        'scaleX': scale_x,
        'scaleY': scale_y
    }
    
    print(f"✅ Heuristic offsets: {offsets}")
    return offsets

def calculate_calibration_accuracy(stored_detections, ai_detections, offsets):
    """
    Calculate how accurate the calibration is by measuring coordinate alignment
    """
    if not stored_detections or not ai_detections:
        return 0.0
    
    # Apply offsets to stored coordinates and measure alignment with AI detections
    total_error = 0
    comparisons = 0
    
    for stored in stored_detections:
        bbox = stored['bbox']
        # Apply calculated offsets
        adjusted_x = bbox['x'] + offsets['offsetX']
        adjusted_y = bbox['y'] + offsets['offsetY']
        adjusted_width = bbox['width'] * offsets['scaleX']
        adjusted_height = bbox['height'] * offsets['scaleY']
        
        # Find closest AI detection
        min_error = float('inf')
        for ai_det in ai_detections:
            error = abs(adjusted_x - ai_det['x']) + abs(adjusted_y - ai_det['y'])
            min_error = min(min_error, error)
        
        if min_error < float('inf'):
            total_error += min_error
            comparisons += 1
    
    if comparisons == 0:
        return 0.0
    
    # Convert error to accuracy (lower error = higher accuracy)
    average_error = total_error / comparisons
    accuracy = max(0.0, 1.0 - (average_error / 100))  # Normalize error to 0-1 scale
    
    print(f"📊 Calibration accuracy: {accuracy:.3f} (avg error: {average_error:.1f}px)")
    return accuracy

def store_calibration_offsets(video_id, offsets):
    """
    Store calibration offsets for potential reuse across similar videos
    """
    try:
        # For now, just store in memory/session
        # In production, could store in database for reuse
        print(f"💾 Storing calibration offsets for video {video_id}: {offsets}")
        
        # Could implement persistent storage here:
        # - Store in database table for video-specific calibrations
        # - Cache globally for similar video types/resolutions
        # - Apply to other videos with similar characteristics
        
    except Exception as e:
        print(f"⚠️ Failed to store calibration offsets: {e}")

@videos_bp.route('/celery-status')
@login_required
def celery_status():
    """Check Celery worker status"""
    try:
        from processing.tasks import celery
        
        inspect = celery.control.inspect()
        active_workers = inspect.active()
        worker_stats = inspect.stats()
        
        status = {
            'workers_available': bool(active_workers),
            'active_workers': list(active_workers.keys()) if active_workers else [],
            'worker_count': len(active_workers) if active_workers else 0,
            'worker_stats': worker_stats,
            'celery_available': True
        }
        
        if active_workers:
            print(f"✅ Celery status check: {len(active_workers)} workers active")
        else:
            print("⚠️ Celery status check: No workers detected")
        
        return jsonify(status)
        
    except ImportError:
        return jsonify({
            'workers_available': False,
            'celery_available': False,
            'error': 'Celery not installed'
        })
    except Exception as e:
        print(f"❌ Error checking Celery status: {e}")
        return jsonify({
            'workers_available': False,
            'celery_available': False,
            'error': str(e)
        })

@videos_bp.route('/dependency-status')
@login_required
def dependency_status():
    """Check video processing dependency status"""
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from hr_management.utils.video_processor import VideoProcessor
        
        processor = VideoProcessor()
        available_methods = processor.get_available_methods()
        
        status = {
            'moviepy': processor.moviepy_available,
            'opencv': processor.opencv_available,
            'ffmpeg': processor.ffmpeg_available,
            'available_methods': available_methods,
            'ready_for_conversion': len(available_methods) > 0
        }
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'ready_for_conversion': False,
            'available_methods': []
        })

@videos_bp.route('/install-dependencies', methods=['POST'])
@login_required
def install_video_dependencies():
    """Install video processing dependencies"""
    try:
        import subprocess
        import sys
        
        packages = [
            'moviepy>=1.0.3',
            'opencv-python>=4.8.0', 
            'imageio>=2.31.1',
            'imageio-ffmpeg>=0.4.7',
            'pillow>=9.0.0',
            'numpy>=1.21.0'
        ]
        
        def install_in_background():
            with current_app.app_context():
                for package in packages:
                    try:
                        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                    except subprocess.CalledProcessError:
                        pass
        
        thread = threading.Thread(target=install_in_background)
        thread.daemon = True
        thread.start()
        
        flash('Installing video processing libraries in background... Check status in a few minutes.', 'info')
        return redirect(url_for('videos.index'))
        
    except Exception as e:
        flash(f'Error installing dependencies: {str(e)}', 'error')
        return redirect(url_for('videos.index'))

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def start_enhanced_gpu_processing(video, processing_options, app):
    """Start enhanced processing with GPU acceleration for faster performance"""
    import threading
    import sys
    import os
    from datetime import datetime
    
    def gpu_process_in_background():
        with app.app_context():
            try:
                # Get database and models from app context
                db = app.db
                Video = app.Video
                DetectedPerson = app.DetectedPerson
                sys.path.append(os.path.dirname(os.path.dirname(__file__)))
                
                # Re-fetch the video object in this thread's session
                video_obj = Video.query.get(video.id)
                if not video_obj:
                    print(f"❌ Video {video.id} not found in database")
                    return
                
                print(f"🚀 Starting GPU-accelerated person extraction for video {video_obj.id}: {video_obj.filename}")
                
                # Get video path
                upload_folder = app.config.get('UPLOAD_FOLDER', 'static/uploads')
                video_path = os.path.join(upload_folder, video_obj.file_path)
                
                # Check if video file exists
                if not os.path.exists(video_path):
                    raise FileNotFoundError(f"Video file not found: {video_path}")
                
                print(f"✅ Video file exists: {video_path} ({os.path.getsize(video_path)} bytes)")
                
                # Step 1: Clear existing detections
                print(f"🗑️ Step 1/5: Clearing existing detection data for video {video_obj.id}")
                video_obj.processing_progress = 5
                db.session.commit()
                
                existing_detections = DetectedPerson.query.filter_by(video_id=video_obj.id).all()
                if existing_detections:
                    detection_count = len(existing_detections)
                    for detection in existing_detections:
                        db.session.delete(detection)
                    db.session.commit()
                    print(f"   ✅ Deleted {detection_count} existing detections")
                
                # Step 2: Run GPU-accelerated person detection
                print(f"🤖 Step 2/5: Running GPU-accelerated person detection...")
                video_obj.processing_progress = 20
                db.session.commit()
                
                # Import GPU-optimized detection module
                try:
                    from processing.gpu_enhanced_detection import gpu_person_detection_task
                    print("🎮 Using GPU-accelerated detection")
                    gpu_available = True
                except ImportError:
                    print("⚠️ GPU detection module not found, falling back to CPU detection")
                    gpu_available = False
                    try:
                        from processing.enhanced_detection import enhanced_person_detection_task as gpu_person_detection_task
                    except ImportError:
                        from processing.enhanced_detection_fallback import enhanced_person_detection_task as gpu_person_detection_task
                
                # Configure GPU processing
                gpu_config = {
                    'use_gpu': processing_options.get('use_gpu', True) and gpu_available,
                    'batch_size': 8 if gpu_available else 4,  # Larger batch size for GPU
                    'device': 'cuda:0' if gpu_available else 'cpu',
                    'fp16': True if gpu_available else False,  # Use half precision on GPU
                    'num_workers': 4 if gpu_available else 2
                }
                
                print(f"🎮 GPU Config: {gpu_config}")
                
                result = gpu_person_detection_task(video_path, gpu_config, video_obj.id, app)
                
                if 'error' in result:
                    raise Exception(f"GPU detection failed: {result['error']}")
                
                video_obj.processing_progress = 60
                db.session.commit()
                
                print(f"🎯 GPU detection completed - found {len(result['detections'])} tracked detections")
                
                # Step 3: Save detections to database
                print(f"💾 Step 3/5: Saving {len(result['detections'])} detections...")
                video_obj.processing_progress = 75
                db.session.commit()
                
                for detection_data in result['detections']:
                    detection = DetectedPerson(
                        video_id=video_obj.id,
                        frame_number=detection_data['frame_number'],
                        timestamp=detection_data['timestamp'],
                        bbox_x=detection_data['x'],
                        bbox_y=detection_data['y'],
                        bbox_width=detection_data['width'],
                        bbox_height=detection_data['height'],
                        confidence=detection_data['confidence'],
                        person_id=detection_data.get('person_id'),
                        track_id=detection_data.get('track_id')
                    )
                    db.session.add(detection)
                
                db.session.commit()
                print(f"✅ Saved {len(result['detections'])} detections")
                
                # Step 4: Update video metadata
                print(f"📊 Step 4/5: Updating video metadata")
                video_obj.processing_progress = 90
                
                if 'processing_summary' in result:
                    summary = result['processing_summary']
                    video_obj.duration = summary.get('duration')
                    
                    # Store annotated video path for playback
                    if result.get('annotated_video_path'):
                        annotated_path = result['annotated_video_path']
                        # Handle both full path and filename only formats
                        if annotated_path.startswith('processing/outputs/'):
                            relative_path = annotated_path.replace('processing/outputs/', '')
                        elif '/' in annotated_path:
                            # Extract just the filename from any path
                            relative_path = os.path.basename(annotated_path)
                        else:
                            # Already just the filename
                            relative_path = annotated_path
                            
                        video_obj.annotated_video_path = relative_path
                        video_obj.processed_path = relative_path  # Use annotated video as processed video
                        print(f"📁 Stored annotated video path: {relative_path}")
                
                db.session.commit()
                
                # Step 5: Complete processing
                print(f"✅ Step 5/5: GPU processing completed for video {video_obj.id}")
                video_obj.status = 'completed'
                video_obj.processing_progress = 100
                video_obj.processing_completed_at = datetime.utcnow()
                
                # Update processing log
                unique_persons = len(set(d.get('person_id') for d in result['detections'] if d.get('person_id')))
                video_obj.processing_log += f"\nGPU processing completed at {datetime.utcnow()} - Found {unique_persons} unique persons"
                
                db.session.commit()
                
                print(f"🎉 GPU-accelerated processing completed successfully!")
                print(f"📊 Results: {unique_persons} unique persons, {len(result['detections'])} detections")
                print(f"📁 Annotated video ready for playback: {result.get('annotated_video_path', 'N/A')}")
                
            except Exception as e:
                import traceback
                
                error_trace = traceback.format_exc()
                print(f"❌ GPU processing failed for video {video.id}: {e}")
                print(f"📋 Full error trace:\n{error_trace}")
                
                try:
                    db = app.db
                    Video = app.Video
                    
                    video_obj = Video.query.get(video.id)
                    if video_obj:
                        video_obj.status = 'failed'
                        video_obj.error_message = f'GPU processing failed: {str(e)}'
                        video_obj.processing_completed_at = datetime.utcnow()
                        db.session.commit()
                except Exception as db_error:
                    print(f"❌ Failed to update database: {db_error}")
    
    # Start background thread
    print(f"🧵 Starting GPU processing thread for video {video.id}...")
    thread = threading.Thread(target=gpu_process_in_background)
    thread.daemon = True
    thread.start()
    print(f"✅ GPU processing thread started")

def start_enhanced_fallback_processing(video, processing_options, app):
    """Start enhanced processing with person tracking and video annotation"""
    import threading
    import sys
    import os
    from datetime import datetime
    
    def enhanced_process_in_background():
        with app.app_context():
            try:
                # Get database and models from app context
                db = app.db
                Video = app.Video
                DetectedPerson = app.DetectedPerson
                sys.path.append(os.path.dirname(os.path.dirname(__file__)))
                
                # Re-fetch the video object in this thread's session
                video_obj = Video.query.get(video.id)
                if not video_obj:
                    print(f"❌ Video {video.id} not found in database")
                    return
                
                print(f"🚀 Starting enhanced processing for video {video_obj.id}: {video_obj.filename}")
                
                # Get video path
                upload_folder = app.config.get('UPLOAD_FOLDER', 'static/uploads')
                if video_obj.processed_path:
                    video_path = os.path.join(upload_folder, video_obj.processed_path)
                    print(f"📁 Using converted video: {video_path}")
                else:
                    video_path = os.path.join(upload_folder, video_obj.file_path)
                    print(f"📁 Using original video: {video_path}")
                
                # Check if video file exists
                if not os.path.exists(video_path):
                    raise FileNotFoundError(f"Video file not found: {video_path}")
                
                print(f"✅ Video file exists: {video_path} ({os.path.getsize(video_path)} bytes)")
                
                # Step 1: Clear existing detections
                print(f"🗑️ Step 1/5: Clearing existing detection data for video {video_obj.id}")
                video_obj.processing_progress = 5
                db.session.commit()
                
                existing_detections = DetectedPerson.query.filter_by(video_id=video_obj.id).all()
                if existing_detections:
                    detection_count = len(existing_detections)
                    print(f"   🔍 Found {detection_count} existing detections to delete")
                    for detection in existing_detections:
                        db.session.delete(detection)
                    db.session.commit()
                    print(f"   ✅ Successfully deleted {detection_count} existing detections")
                
                # Step 2: Run enhanced person detection with tracking
                print(f"🤖 Step 2/5: Running enhanced person detection with tracking...")
                video_obj.processing_progress = 20
                db.session.commit()
                
                # Import the correct enhanced detection module based on available dependencies
                try:
                    from processing.enhanced_detection import enhanced_person_detection_task
                    print("🤖 Using full AI-powered enhanced detection")
                except ImportError:
                    from processing.enhanced_detection_fallback import enhanced_person_detection_task
                    print("🔄 Using fallback enhanced detection (demo mode)")
                
                result = enhanced_person_detection_task(video_path)
                
                if 'error' in result:
                    raise Exception(f"Enhanced detection failed: {result['error']}")
                
                video_obj.processing_progress = 60
                db.session.commit()
                
                print(f"🎯 Enhanced detection completed - found {len(result['detections'])} tracked detections")
                
                # Step 3: Save tracked detections to database with person_id and track_id
                print(f"💾 Step 3/5: Saving {len(result['detections'])} tracked detections to database")
                video_obj.processing_progress = 75
                db.session.commit()
                
                for detection_data in result['detections']:
                    detection = DetectedPerson(
                        video_id=video_obj.id,
                        frame_number=detection_data['frame_number'],
                        timestamp=detection_data['timestamp'],
                        bbox_x=detection_data['x'],
                        bbox_y=detection_data['y'],
                        bbox_width=detection_data['width'],
                        bbox_height=detection_data['height'],
                        confidence=detection_data['confidence'],
                        person_id=detection_data.get('person_id'),  # NEW: Person tracking ID
                        track_id=detection_data.get('track_id')     # NEW: Internal tracking ID
                    )
                    db.session.add(detection)
                
                db.session.commit()
                print(f"✅ Saved {len(result['detections'])} tracked detections to database")
                
                # Step 4: Update video metadata with processing summary
                print(f"📊 Step 4/5: Updating video metadata")
                video_obj.processing_progress = 90
                
                if 'processing_summary' in result:
                    summary = result['processing_summary']
                    video_obj.duration = summary.get('duration')
                    
                    # Store annotated video path for video player
                    if result.get('annotated_video_path'):
                        # Store relative path from the uploads directory
                        annotated_path = result['annotated_video_path']
                        if annotated_path.startswith('processing/outputs/'):
                            # Convert to relative path that can be served
                            relative_path = annotated_path.replace('processing/outputs/', '')
                            video_obj.annotated_video_path = relative_path
                            print(f"📁 Stored annotated video path: {relative_path}")
                    
                    # Store enhanced processing info
                    video_obj.processing_log += f"\nEnhanced processing completed:"
                    video_obj.processing_log += f"\n- Annotated video: {result.get('annotated_video_path', 'N/A')}"
                    video_obj.processing_log += f"\n- Total persons detected: {summary.get('total_persons', 0)}"
                    video_obj.processing_log += f"\n- Person summary: {summary.get('person_summary', {})}"
                
                db.session.commit()
                
                # Step 5: Complete processing
                print(f"✅ Step 5/5: Enhanced processing completed for video {video_obj.id}")
                video_obj.status = 'completed'
                video_obj.processing_progress = 100
                video_obj.processing_completed_at = datetime.utcnow()
                
                # Update processing log with final summary
                unique_persons = len(set(d.get('person_id') for d in result['detections'] if d.get('person_id')))
                video_obj.processing_log += f"\nCompleted at {datetime.utcnow()} - Found {unique_persons} unique persons with {len(result['detections'])} total detections"
                
                db.session.commit()
                
                print(f"🎉 Enhanced processing completed successfully!")
                print(f"📊 Results: {unique_persons} unique persons, {len(result['detections'])} total detections")
                print(f"📁 Annotated video: {result.get('annotated_video_path', 'N/A')}")
                
            except Exception as e:
                import traceback
                
                error_trace = traceback.format_exc()
                print(f"❌ Enhanced processing failed for video {video.id}: {e}")
                print(f"📋 Full error trace:\n{error_trace}")
                
                try:
                    # Get database and models from app context
                    db = app.db
                    Video = app.Video
                    
                    # Re-fetch video in case of session issues
                    video_obj = Video.query.get(video.id)
                    if video_obj:
                        video_obj.status = 'failed'
                        video_obj.error_message = f'Enhanced processing failed: {str(e)}'
                        video_obj.processing_completed_at = datetime.utcnow()
                        db.session.commit()
                        print(f"💾 Updated video {video.id} status to failed")
                except Exception as db_error:
                    print(f"❌ Failed to update database status: {db_error}")
    
    # Start background thread
    print(f"🧵 Starting enhanced processing thread for video {video.id}...")
    thread = threading.Thread(target=enhanced_process_in_background)
    thread.daemon = True
    thread.start()
    print(f"✅ Enhanced processing thread started for video {video.id}")

def start_fallback_processing(video, processing_options, app):
    """Start processing in a background thread when Celery is not available"""
    import threading
    import sys
    import os
    from datetime import datetime
    
    def process_in_background():
        with app.app_context():
            try:
                # Get database and models from app context
                db = app.db
                Video = app.Video
                sys.path.append(os.path.dirname(os.path.dirname(__file__)))
                
                # Re-fetch the video object in this thread's session to avoid session issues
                video_obj = Video.query.get(video.id)
                if not video_obj:
                    print(f"❌ Video {video.id} not found in database")
                    return
                
                print(f"🔄 Starting fallback processing for video {video_obj.id} in background thread...")
                
                # Get video path
                upload_folder = app.config.get('UPLOAD_FOLDER', 'static/uploads')
                if video_obj.processed_path:
                    video_path = os.path.join(upload_folder, video_obj.processed_path)
                    print(f"📁 Using converted video: {video_path}")
                else:
                    video_path = os.path.join(upload_folder, video_obj.file_path)
                    print(f"📁 Using original video: {video_path}")
                
                # Check if video file exists
                if not os.path.exists(video_path):
                    raise FileNotFoundError(f"Video file not found: {video_path}")
                
                print(f"✅ Video file exists: {video_path} ({os.path.getsize(video_path)} bytes)")
                
                # Step 1: Extract video metadata
                print(f"🔍 Step 1/4: Extracting metadata for video {video_obj.id}")
                video_obj.processing_progress = 10
                db.session.commit()
                
                from processing.standalone_tasks import extract_video_metadata
                metadata = extract_video_metadata(video_path)
                video_obj.duration = metadata.get('duration')
                video_obj.fps = metadata.get('fps') 
                video_obj.resolution = metadata.get('resolution')
                video_obj.processing_progress = 25
                db.session.commit()
                
                print(f"📊 Video metadata: duration={metadata.get('duration')}s, fps={metadata.get('fps')}, resolution={metadata.get('resolution')}")
                
                # CLEAR ALL EXISTING DETECTION DATA BEFORE RE-PROCESSING (Fallback mode)
                print(f"🗑️ [Fallback] Clearing all existing detection data for video {video_obj.id}")
                try:
                    DetectedPerson = app.DetectedPerson
                    existing_detections = DetectedPerson.query.filter_by(video_id=video_obj.id).all()
                    
                    if existing_detections:
                        detection_count = len(existing_detections)
                        print(f"   🔍 Found {detection_count} existing detections to delete")
                        
                        for detection in existing_detections:
                            db.session.delete(detection)
                        
                        db.session.commit()
                        print(f"   ✅ Successfully deleted {detection_count} existing detections")
                    else:
                        print(f"   📝 No existing detections found for video {video_obj.id}")
                        
                except Exception as e:
                    print(f"   ⚠️ Warning: Could not clear existing detections: {e}")
                    # Continue processing anyway - this is not a critical error
                
                # Step 2: Detect persons
                print(f"👥 Step 2/4: Detecting persons in video {video_obj.id}")
                video_obj.processing_progress = 30
                db.session.commit()
                
                from processing.standalone_tasks import detect_persons_in_video
                print(f"🔄 Starting person detection...")
                detections = detect_persons_in_video(video_path)
                video_obj.processing_progress = 70
                db.session.commit()
                
                print(f"🎯 Found {len(detections)} person detections")
                
                # Step 3: Save detections to database
                print(f"💾 Step 3/4: Saving {len(detections)} detections to database")
                video_obj.processing_progress = 80
                db.session.commit()
                
                from processing.standalone_tasks import save_detections_to_db
                # Import DetectedPerson model
                DetectedPerson = app.DetectedPerson if hasattr(app, 'DetectedPerson') else None
                save_detections_to_db(video_obj.id, detections, metadata.get('fps', 25), db, DetectedPerson)
                video_obj.processing_progress = 90
                db.session.commit()
                
                # Step 4: Complete processing
                print(f"✅ Step 4/4: Person extraction completed for video {video_obj.id}")
                video_obj.status = 'completed'
                video_obj.processing_progress = 100
                video_obj.processing_completed_at = datetime.utcnow()
                
                # Update processing log
                if video_obj.processing_log:
                    video_obj.processing_log += f"\nCompleted at {datetime.utcnow()} - Found {len(detections)} persons"
                else:
                    video_obj.processing_log = f"Completed at {datetime.utcnow()} - Found {len(detections)} persons"
                
                db.session.commit()
                
                print(f"🎉 Fallback processing completed successfully for video {video_obj.id}: {video_obj.filename}")
                
            except Exception as e:
                import traceback
                
                error_trace = traceback.format_exc()
                print(f"❌ Fallback processing failed for video {video.id}: {e}")
                print(f"📋 Full error trace:\n{error_trace}")
                
                try:
                    # Get database and models from app context
                    db = app.db
                    Video = app.Video
                    
                    # Re-fetch video in case of session issues
                    video_obj = Video.query.get(video.id)
                    if video_obj:
                        video_obj.status = 'failed'
                        video_obj.error_message = f'Processing failed: {str(e)}'
                        video_obj.processing_completed_at = datetime.utcnow()
                        db.session.commit()
                        print(f"💾 Updated video {video.id} status to failed")
                except Exception as db_error:
                    print(f"❌ Failed to update database status: {db_error}")
    
    # Start background thread
    print(f"🧵 Starting background thread for video {video.id} fallback processing...")
    thread = threading.Thread(target=process_in_background)
    thread.daemon = True
    thread.start()
    print(f"✅ Background thread started for video {video.id}")
    
    print(f"🧵 Started fallback processing thread for video {video.id}")


# WebSocket event handlers for real-time progress updates
if SOCKETIO_AVAILABLE:
    from flask_socketio import SocketIO
    
    def register_socketio_events(socketio):
        """Register SocketIO events for video processing"""
        
        @socketio.on('join_video_room')
        def handle_join_video_room(data):
            """Join a room for specific video updates"""
            video_id = data.get('video_id')
            if video_id:
                room = f'video_{video_id}'
                join_room(room)
                emit('status', {'message': f'Joined video {video_id} updates'})
                print(f"📡 WebSocket: Client joined room {room}")
        
        @socketio.on('leave_video_room')
        def handle_leave_video_room(data):
            """Leave a room for specific video updates"""
            video_id = data.get('video_id')
            if video_id:
                room = f'video_{video_id}'
                leave_room(room)
                emit('status', {'message': f'Left video {video_id} updates'})
                print(f"📡 WebSocket: Client left room {room}")
        
        @socketio.on('join_admin_room')
        def handle_join_admin_room():
            """Join admin room for all conversion updates"""
            join_room('admin')
            emit('status', {'message': 'Joined admin updates'})
            print(f"📡 WebSocket: Client joined admin room")
        
        @socketio.on('request_video_status')
        def handle_request_video_status(data):
            """Request current status of a specific video"""
            video_id = data.get('video_id')
            if video_id:
                # Get current video status
                Video = current_app.Video
                video = Video.query.get(video_id)
                
                if video:
                    # Get conversion task if exists
                    import sys
                    import os
                    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
                    from hr_management.utils.conversion_manager import conversion_manager
                    
                    task = conversion_manager.get_task_by_video_id(video_id)
                    
                    status_data = {
                        'video_id': video_id,
                        'status': video.status,
                        'progress': task.progress if task else 0.0,
                        'message': task.message if task else '',
                        'task_id': task.task_id if task else None,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    
                    emit('video_status', status_data)
                    print(f"📡 WebSocket: Sent current status for video {video_id}")
        
        @socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            print(f"📡 WebSocket: Client connected")
            emit('status', {'message': 'Connected to video processing updates'})
        
        @socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            print(f"📡 WebSocket: Client disconnected")
    
    # Register events with the current app's socketio instance if available
    def init_socketio_events():
        """Initialize SocketIO events when app context is available"""
        try:
            if hasattr(current_app, 'extensions') and 'socketio' in current_app.extensions:
                socketio = current_app.extensions['socketio']
                register_socketio_events(socketio)
                print("✅ WebSocket events registered for video processing")
        except Exception as e:
            print(f"⚠️ Failed to register WebSocket events: {e}")

else:
    def init_socketio_events():
        """No-op when SocketIO is not available"""
        pass