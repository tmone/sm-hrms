from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app, send_from_directory, Response, send_file
from flask_login import login_required
from werkzeug.utils import secure_filename
import os
import uuid
from datetime import datetime
import threading
import time

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
            
            # Check if video format is web-compatible
            is_web_compatible, format_detected = detect_video_format(upload_path)
            
            # Determine initial status based on format compatibility
            if is_web_compatible:
                initial_status = 'uploaded'
                status_message = f'Video "{filename}" uploaded successfully! Format: {format_detected}'
            else:
                initial_status = 'converting'
                status_message = f'Video "{filename}" uploaded! Detected format: {format_detected}. Auto-converting to web-compatible MP4...'
            
            # Create video record
            video = Video(
                filename=filename,
                file_path=unique_filename,  # Store just the filename
                file_size=os.path.getsize(upload_path),
                title=request.form.get('title', ''),
                description=request.form.get('description', ''),
                priority=request.form.get('priority', 'normal'),
                status=initial_status,
                processing_started_at=datetime.utcnow() if not is_web_compatible else None
            )
            
            # Save to database
            db.session.add(video)
            db.session.commit()
            
            # Auto-convert if not web-compatible
            if not is_web_compatible:
                start_auto_conversion(video, upload_path, upload_folder, current_app._get_current_object())
                flash(status_message + ' Conversion will complete in the background.', 'info')
            else:
                flash(status_message + ' You can now process it to extract persons.', 'success')
            
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
    
    # Get detections
    detections = DetectedPerson.query.filter_by(video_id=id).all()
    
    return render_template('videos/detail.html',
                         video=video,
                         detections=detections)

@videos_bp.route('/<int:id>/delete', methods=['POST'])
@login_required
def delete(id):
    try:
        Video = current_app.Video
        db = current_app.db
        
        video = Video.query.get_or_404(id)
        
        # Delete physical file
        if os.path.exists(video.file_path):
            os.remove(video.file_path)
        
        # Delete processed file if exists
        if video.processed_path and os.path.exists(video.processed_path):
            os.remove(video.processed_path)
        
        # Delete from database
        db.session.delete(video)
        db.session.commit()
        
        flash(f'Video "{video.filename}" deleted successfully!', 'success')
    except Exception as e:
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
        if video.status not in ['uploaded', 'failed']:
            flash('Only uploaded or failed videos can be processed', 'warning')
            return redirect(url_for('videos.detail', id=id))
        
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
        
        # Here you would queue the actual processing task
        # For now, we'll simulate by marking as completed
        # In a real implementation, you'd use Celery or similar:
        # process_video_task.delay(video.id, extract_persons, face_recognition, extract_frames)
        
        flash(f'Processing started for "{video.filename}". This may take a while depending on video length.', 'info')
        
        # For demo purposes, you could add some mock detected persons here
        # This would normally be done by the background processing task
        
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

@videos_bp.route('/<int:id>/delete', methods=['POST'])
@login_required
def delete_video(id):
    """Delete a video and its file"""
    try:
        Video = current_app.Video
        db = current_app.db
        
        video = Video.query.get_or_404(id)
        filename = video.filename
        
        # Delete the physical file
        upload_folder = current_app.config.get('UPLOAD_FOLDER', 'static/uploads')
        file_path = os.path.join(upload_folder, video.file_path)
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Delete processed file if exists
        if video.processed_path:
            processed_path = os.path.join('static/processed', video.processed_path)
            if os.path.exists(processed_path):
                os.remove(processed_path)
        
        # Delete from database
        db.session.delete(video)
        db.session.commit()
        
        flash(f'Video "{filename}" has been deleted successfully.', 'success')
    except Exception as e:
        if hasattr(current_app, 'db'):
            current_app.db.session.rollback()
        flash(f'Error deleting video: {str(e)}', 'error')
    
    return redirect(url_for('videos.index'))

@videos_bp.route('/stream/<path:filename>')
@login_required
def stream_video(filename):
    """Stream video files with range request support"""
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'static/uploads')
    file_path = os.path.join(upload_folder, filename)
    
    if not os.path.exists(file_path):
        return "Video file not found", 404
    
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
    return send_from_directory(upload_folder, filename)

@videos_bp.route('/<int:id>/convert', methods=['POST'])
@login_required
def convert_video(id):
    """Convert video to web-compatible format"""
    try:
        Video = current_app.Video
        db = current_app.db
        
        video = Video.query.get_or_404(id)
        
        # Check if video needs conversion
        upload_folder = current_app.config.get('UPLOAD_FOLDER', 'static/uploads')
        input_path = os.path.join(upload_folder, video.file_path)
        
        if not os.path.exists(input_path):
            flash('Video file not found', 'error')
            return redirect(url_for('videos.detail', id=id))
        
        # Check if already converting
        if video.status == 'converting':
            flash('Video is already being converted', 'warning')
            return redirect(url_for('videos.detail', id=id))
        
        # Get app instance for background thread
        app = current_app._get_current_object()
        
        # Start conversion in background
        def convert_in_background():
            # Create application context for background thread
            with app.app_context():
                try:
                    # Import video processor
                    import sys
                    import os
                    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
                    from utils.video_processor import VideoProcessor
                    
                    processor = VideoProcessor()
                    available_methods = processor.get_available_methods()
                    
                    if not available_methods:
                        video.status = 'failed'
                        video.error_message = 'No video conversion libraries available. Please install: pip install moviepy opencv-python'
                        app.db.session.commit()
                        return
                    
                    # Update status to converting
                    video.status = 'converting'
                    video.processing_started_at = datetime.utcnow()
                    video.error_message = None
                    app.db.session.commit()
                    
                    # Generate output path
                    output_filename = f"{uuid.uuid4()}_converted_{video.filename}"
                    output_path = os.path.join(upload_folder, output_filename)
                    
                    # Convert video (prefer moviepy/opencv over ffmpeg for easier installation)
                    print(f"🔄 Starting manual conversion: {input_path} -> {output_path}")
                    print(f"📊 Available methods: {available_methods}")
                    
                    success, output_file, message = processor.convert_video(
                        input_path,
                        output_path,
                        method='auto',  # Will try moviepy first, then opencv
                        quality='medium'
                    )
                    
                    print(f"🎯 Manual conversion result: success={success}, message={message}")
                    
                    if success:
                        # Update video record
                        video.status = 'completed'
                        video.processed_path = output_filename
                        video.processing_completed_at = datetime.utcnow()
                        video.error_message = None
                        
                        # Get converted video info
                        try:
                            info = processor.get_video_info(output_path)
                            if info and info.get('format') == 'readable':
                                video.duration = info.get('duration', 0)
                                video.resolution = f"{info.get('width', 0)}x{info.get('height', 0)}"
                                video.fps = info.get('fps', 0)
                                video.codec = 'h264'
                        except Exception as info_error:
                            # Don't fail conversion if we can't get info
                            print(f"Warning: Could not get video info: {info_error}")
                        
                        app.db.session.commit()
                    else:
                        video.status = 'failed'
                        video.error_message = f'Conversion failed: {message}'
                        video.processing_completed_at = datetime.utcnow()
                        app.db.session.commit()
                        
                except Exception as e:
                    try:
                        video.status = 'failed'
                        video.error_message = f'Conversion error: {str(e)}'
                        video.processing_completed_at = datetime.utcnow()
                        app.db.session.commit()
                    except Exception as db_error:
                        print(f"Database error in conversion thread: {db_error}")
                        print(f"Original conversion error: {e}")
        
        # Start background thread
        thread = threading.Thread(target=convert_in_background)
        thread.daemon = True
        thread.start()
        
        flash('Video conversion started! This may take several minutes.', 'info')
        return redirect(url_for('videos.detail', id=id))
        
    except Exception as e:
        flash(f'Error starting conversion: {str(e)}', 'error')
        return redirect(url_for('videos.detail', id=id))

@videos_bp.route('/api/<int:id>/conversion-status')
@login_required
def conversion_status(id):
    """Get conversion status with progress for AJAX polling"""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from utils.conversion_manager import conversion_manager
    
    Video = current_app.Video
    video = Video.query.get_or_404(id)
    
    # Get basic video status
    response = {
        'status': video.status,
        'error_message': video.error_message,
        'processing_started_at': video.processing_started_at.isoformat() if video.processing_started_at else None,
        'processing_completed_at': video.processing_completed_at.isoformat() if video.processing_completed_at else None,
        'processed_path': video.processed_path,
        'progress': 0.0,
        'progress_message': ''
    }
    
    # If converting, get detailed progress from conversion manager
    if video.status == 'converting':
        task = conversion_manager.get_task_by_video_id(id)
        if task:
            response.update({
                'progress': task.progress,
                'progress_message': task.message,
                'task_id': task.task_id
            })
            print(f"🔄 API: Sending progress for video {id}: {task.progress}% - {task.message}")
        else:
            print(f"⚠️ API: No conversion task found for video {id}")
            response.update({
                'progress': 0.0,
                'progress_message': 'No active conversion task found'
            })
    
    print(f"📡 API Response for video {id}: status={response['status']}, progress={response['progress']}%")
    return jsonify(response)

@videos_bp.route('/api/conversion-tasks')
@login_required 
def get_conversion_tasks():
    """Get all conversion tasks status"""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from utils.conversion_manager import conversion_manager
    
    return jsonify(conversion_manager.get_all_tasks())

@videos_bp.route('/api/conversion-task/<task_id>')
@login_required
def get_conversion_task(task_id):
    """Get specific conversion task status"""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from utils.conversion_manager import conversion_manager
    
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
    from utils.conversion_manager import conversion_manager
    
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

@videos_bp.route('/dependency-status')
@login_required
def dependency_status():
    """Check video processing dependency status"""
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from utils.video_processor import VideoProcessor
        
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

def detect_video_format(file_path):
    """Detect if video format is web-compatible"""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(32)
            
            # Check for web-compatible formats
            if header.startswith(b'ftypmp4') or header[4:8] == b'ftyp':
                return True, 'MP4'
            elif header.startswith(b'\x1a\x45\xdf\xa3'):
                # Check if it's WebM (web-compatible) or MKV (needs conversion)
                # WebM usually has 'webm' in the header further down
                try:
                    with open(file_path, 'rb') as f2:
                        first_1kb = f2.read(1024)
                        if b'webm' in first_1kb.lower():
                            return True, 'WebM'
                        else:
                            return False, 'MKV'
                except:
                    return False, 'MKV'
            elif header.startswith(b'RIFF') and b'AVI ' in header:
                return False, 'AVI'
            elif header.startswith(b'FLV'):
                return False, 'FLV'
            elif header.startswith(b'IMKH'):
                return False, 'IMKH'
            elif header.startswith(b'\x00\x00\x00'):
                # Could be QuickTime/MOV - usually needs conversion for web
                return False, 'MOV'
            else:
                # Unknown format - assume needs conversion for safety
                return False, 'Unknown'
    except:
        return False, 'Unknown'

def start_auto_conversion(video, input_path, upload_folder, app):
    """Start automatic conversion using conversion manager"""
    import sys
    import os
    from datetime import datetime
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from utils.conversion_manager import conversion_manager, create_conversion_processor
    from utils.video_processor import VideoProcessor
    
    # Generate output path
    output_filename = f"{uuid.uuid4()}_converted_{video.filename}"
    output_path = os.path.join(upload_folder, output_filename)
    
    # Create conversion task
    task_id = conversion_manager.create_task(video.id, input_path, output_path)
    
    # Update video status to converting BEFORE starting the task
    video.status = 'converting'
    video.processing_started_at = datetime.utcnow()
    video.processing_log = f"Conversion Task ID: {task_id}"
    video.error_message = None
    app.db.session.commit()
    
    print(f"🔄 Video {video.id} status updated to 'converting' with task {task_id[:8]}")
    
    # Create processor
    processor = VideoProcessor()
    conversion_processor = create_conversion_processor(processor)
    
    # Start conversion
    success = conversion_manager.start_conversion(task_id, app, conversion_processor)
    
    if success:
        print(f"✅ Auto-conversion task {task_id[:8]} started for video: {video.filename}")
    else:
        print(f"❌ Failed to start auto-conversion task for video: {video.filename}")
        video.status = 'failed'
        video.error_message = 'Failed to start conversion task'
        app.db.session.commit()

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
                    from utils.conversion_manager import conversion_manager
                    
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