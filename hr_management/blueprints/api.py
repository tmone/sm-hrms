from flask import Blueprint, jsonify, request, current_app
from flask_login import login_required
from datetime import datetime, date
from sqlalchemy import func

api_bp = Blueprint('api', __name__)

@api_bp.route('/stats')
@login_required
def stats():
    """Get dashboard statistics"""
    return jsonify(get_dashboard_stats())

@api_bp.route('/test-endpoint')
@login_required  
def test_endpoint():
    """Test endpoint to verify API is working"""
    return jsonify({
        'message': 'Enhanced API endpoint is working!',
        'timestamp': datetime.utcnow().isoformat(),
        'version': 'enhanced_v2'
    })

@api_bp.route('/processing-queue')
@login_required
def processing_queue():
    """Get detailed processing queue status"""
    import sys
    import os
    
    print("[SEARCH] API: Enhanced processing-queue endpoint called")
    
    try:
        Video = getattr(current_app, 'Video', None)
        if not Video:
            return jsonify({
                'error': 'Video processing not available',
                'queue': {'pending': 0, 'converting': 0, 'processing': 0, 'completed': 0, 'failed': 0},
                'active_tasks': [],
                'total_tasks': 0
            })
        
        # Get video queue status from database
        queue_status = {
            'pending': Video.query.filter_by(status='uploaded').count(),
            'converting': Video.query.filter_by(status='converting').count(),
            'processing': Video.query.filter_by(status='processing').count(),
            'completed': Video.query.filter_by(status='completed').count(),
            'failed': Video.query.filter_by(status='failed').count()
        }
        
        # Get conversion manager status
        active_tasks = []
        total_tasks = 0
        conversion_manager_status = "unavailable"
        
        try:
            sys.path.append(os.path.dirname(os.path.dirname(__file__)))
            from hr_management.utils.conversion_manager import conversion_manager
            
            all_tasks = conversion_manager.get_all_tasks()
            total_tasks = len(all_tasks)
            conversion_manager_status = "connected"
            
            print(f"[SEARCH] API: Found {total_tasks} total tasks in conversion manager")
            
            # Get active tasks
            for task_id, task_data in all_tasks.items():
                print(f"[SEARCH] API: Task {task_id[:8]} - Status: {task_data['status']}, Progress: {task_data['progress']}%")
                
                if task_data['status'] in ['queued', 'running']:
                    # Get video info
                    video = Video.query.get(task_data['video_id'])
                    
                    active_tasks.append({
                        'task_id': task_id,
                        'video_id': task_data['video_id'],
                        'video_filename': video.filename if video else 'Unknown',
                        'status': task_data['status'],
                        'progress': task_data['progress'],
                        'message': task_data['message'],
                        'started_at': task_data['started_at'],
                        'elapsed_time': get_elapsed_time(task_data['started_at'])
                    })
            
            print(f"[SEARCH] API: Found {len(active_tasks)} active tasks")
            
        except Exception as e:
            print(f"[ERROR] API: Error getting conversion manager status: {e}")
            conversion_manager_status = f"error: {str(e)}"
        
        # Get recent activity (last 10 videos)
        recent_videos = Video.query.order_by(Video.updated_at.desc()).limit(10).all()
        recent_activity = []
        
        for video in recent_videos:
            recent_activity.append({
                'id': video.id,
                'filename': video.filename,
                'status': video.status,
                'created_at': video.created_at.isoformat(),
                'updated_at': video.updated_at.isoformat(),
                'file_size_mb': round(video.file_size / 1024 / 1024, 1) if video.file_size else 0,
                'processing_started_at': video.processing_started_at.isoformat() if video.processing_started_at else None,
                'processing_completed_at': video.processing_completed_at.isoformat() if video.processing_completed_at else None,
                'error_message': video.error_message
            })
        
        response_data = {
            'queue_status': queue_status,
            'active_tasks': active_tasks,
            'total_tasks': total_tasks,
            'recent_activity': recent_activity,
            'conversion_manager_status': conversion_manager_status,
            'summary': {
                'total_videos': sum(queue_status.values()),
                'active_processing': queue_status['converting'] + queue_status['processing'],
                'completion_rate': round(queue_status['completed'] / sum(queue_status.values()) * 100, 1) if sum(queue_status.values()) > 0 else 0
            },
            'timestamp': datetime.utcnow().isoformat(),
            'endpoint_version': 'enhanced_v2'  # To confirm we're hitting the right endpoint
        }
        
        print(f"[SEARCH] API: Returning response with {len(active_tasks)} active tasks, queue status: {queue_status}")
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            'error': f'Error fetching queue status: {str(e)}',
            'queue_status': {'pending': 0, 'converting': 0, 'processing': 0, 'completed': 0, 'failed': 0},
            'active_tasks': [],
            'total_tasks': 0
        }), 500

def get_elapsed_time(started_at_str):
    """Calculate elapsed time from start timestamp"""
    if not started_at_str:
        return None
    
    try:
        from datetime import datetime
        started_at = datetime.fromisoformat(started_at_str.replace('Z', '+00:00'))
        elapsed = datetime.utcnow() - started_at.replace(tzinfo=None)
        
        total_seconds = int(elapsed.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    except Exception:
        return None

@api_bp.route('/employees')
@login_required
def employees():
    """Get employees list"""
    Employee = current_app.Employee
    
    search = request.args.get('search', '')
    department = request.args.get('department', '')
    status = request.args.get('status', 'active')
    
    query = Employee.query
    
    if search:
        query = query.filter(Employee.name.contains(search) | 
                           Employee.email.contains(search))
    
    if department:
        query = query.filter_by(department=department)
    
    if status:
        query = query.filter_by(status=status)
    
    employees = query.all()
    return jsonify([emp.to_dict() for emp in employees])

@api_bp.route('/videos')
@login_required
def videos():
    """Get videos list"""
    status = request.args.get('status', '')
    
    # Video management is optional feature
    try:
        Video = getattr(current_app, 'Video', None)
        if Video:
            query = Video.query
            if status:
                query = query.filter_by(status=status)
            videos = query.order_by(Video.created_at.desc()).all()
            return jsonify([video.to_dict() for video in videos])
        else:
            return jsonify([])
    except Exception:
        return jsonify([])

@api_bp.route('/videos/<int:id>/detections')
@login_required
def video_detections(id):
    """Get detections for a specific video"""
    # Face detection is optional feature
    try:
        DetectedPerson = getattr(current_app, 'DetectedPerson', None)
        if DetectedPerson:
            detections = DetectedPerson.query.filter_by(video_id=id).all()
            return jsonify([detection.to_dict() for detection in detections])
        else:
            return jsonify([])
    except Exception:
        return jsonify([])

@api_bp.route('/attendance/today')
@login_required
def attendance_today():
    """Get today's attendance"""
    Employee = current_app.Employee
    AttendanceRecord = current_app.AttendanceRecord
    db = current_app.db
    
    today = date.today()
    
    attendance = db.session.query(
        AttendanceRecord.employee_id,
        Employee.name,
        AttendanceRecord.check_in_time,
        AttendanceRecord.check_out_time,
        AttendanceRecord.status
    ).join(Employee).filter(AttendanceRecord.date == today).all()
    
    return jsonify([{
        'employee_id': a.employee_id,
        'name': a.name,
        'check_in_time': a.check_in_time.isoformat() if a.check_in_time else None,
        'check_out_time': a.check_out_time.isoformat() if a.check_out_time else None,
        'status': a.status
    } for a in attendance])

@api_bp.route('/attendance/mark', methods=['POST'])
@login_required
def mark_attendance():
    """Mark attendance for an employee"""
    Employee = current_app.Employee
    AttendanceRecord = current_app.AttendanceRecord
    db = current_app.db
    
    data = request.get_json()
    
    employee_id = data.get('employee_id')
    action = data.get('action')  # 'check_in' or 'check_out'
    
    if not employee_id or not action:
        return jsonify({'error': 'Missing employee_id or action'}), 400
    
    employee = Employee.query.get(employee_id)
    if not employee:
        return jsonify({'error': 'Employee not found'}), 404
    
    today = date.today()
    
    # Get or create attendance record for today
    attendance = AttendanceRecord.query.filter_by(
        employee_id=employee_id,
        date=today
    ).first()
    
    if not attendance:
        attendance = AttendanceRecord(
            employee_id=employee_id,
            date=today,
            status='present'
        )
        db.session.add(attendance)
    
    try:
        if action == 'check_in':
            if attendance.check_in_time:
                return jsonify({'error': 'Already checked in today'}), 400
            attendance.check_in_time = datetime.now()
        elif action == 'check_out':
            if not attendance.check_in_time:
                return jsonify({'error': 'Must check in first'}), 400
            if attendance.check_out_time:
                return jsonify({'error': 'Already checked out today'}), 400
            attendance.check_out_time = datetime.now()
        else:
            return jsonify({'error': 'Invalid action'}), 400
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Successfully {action.replace("_", " ")} for {employee.name}',
            'attendance': attendance.to_dict()
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@api_bp.route('/recognition/identify', methods=['POST'])
@login_required
def identify_person():
    """Manually identify a detected person"""
    # Face recognition is optional feature
    try:
        DetectedPerson = getattr(current_app, 'DetectedPerson', None)
        RecognitionResult = getattr(current_app, 'RecognitionResult', None)
        Employee = current_app.Employee
        db = current_app.db
        
        if not DetectedPerson or not RecognitionResult:
            return jsonify({'error': 'Face recognition not available'}), 404
            
        data = request.get_json()
        
        detection_id = data.get('detection_id')
        employee_id = data.get('employee_id')
        
        if not detection_id:
            return jsonify({'error': 'Missing detection_id'}), 400
        
        detection = DetectedPerson.query.get(detection_id)
        if not detection:
            return jsonify({'error': 'Detection not found'}), 404
        
        if employee_id:
            employee = Employee.query.get(employee_id)
            if not employee:
                return jsonify({'error': 'Employee not found'}), 404
            
            detection.employee_id = employee_id
            detection.is_identified = True
            
            # Create recognition result record
            result = RecognitionResult(
                video_id=detection.video_id,
                model_id=None,  # Manual identification
                detected_person_id=detection.id,
                employee_id=employee_id,
                confidence=1.0,  # Manual identification is 100% confident
                is_verified=True
            )
            db.session.add(result)
            
            message = f'Person identified as {employee.name}'
        else:
            # Remove identification
            detection.employee_id = None
            detection.is_identified = False
            message = 'Person identification removed'
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': message,
            'detection': detection.to_dict()
        })
        
    except Exception as e:
        if hasattr(current_app, 'db'):
            current_app.db.session.rollback()
        return jsonify({'error': str(e)}), 500
    except Exception:
        return jsonify({'error': 'Face recognition not available'}), 404

@api_bp.route('/models/performance')
@login_required
def models_performance():
    """Get model performance metrics"""
    # Model performance is optional feature
    try:
        TrainedModel = getattr(current_app, 'TrainedModel', None)
        RecognitionResult = getattr(current_app, 'RecognitionResult', None)
        
        if not TrainedModel or not RecognitionResult:
            return jsonify([])
            
        models = TrainedModel.query.filter_by(status='completed').all()
        
        performance_data = []
        for model in models:
            # Get recognition results for this model
            results = RecognitionResult.query.filter_by(model_id=model.id).all()
            
            total_predictions = len(results)
            verified_correct = len([r for r in results if r.is_verified and r.employee_id])
            
            accuracy = (verified_correct / total_predictions * 100) if total_predictions > 0 else 0
            
            performance_data.append({
                'model_id': model.id,
                'model_name': model.name,
                'model_type': model.model_type,
                'total_predictions': total_predictions,
                'verified_correct': verified_correct,
                'accuracy': accuracy,
                'training_accuracy': model.accuracy,
                'validation_accuracy': model.validation_accuracy,
                'is_active': model.is_active
            })
        
        return jsonify(performance_data)
    except Exception:
        return jsonify([])

@api_bp.route('/system/health')
@login_required
def system_health():
    """Get system health status"""
    try:
        # Test database connection
        db = current_app.db
        db.session.execute('SELECT 1')
        db_status = 'healthy'
    except Exception:
        db_status = 'error'
    
    # Get processing queue status (optional feature)
    try:
        Video = getattr(current_app, 'Video', None)
        if Video:
            processing_count = Video.query.filter_by(status='processing').count()
            pending_count = Video.query.filter_by(status='uploaded').count()
        else:
            processing_count = pending_count = 0
    except Exception:
        processing_count = pending_count = 0
    
    # Get active models (optional feature)
    try:
        TrainedModel = getattr(current_app, 'TrainedModel', None)
        if TrainedModel:
            active_models = TrainedModel.query.filter_by(is_active=True).count()
        else:
            active_models = 0
    except Exception:
        active_models = 0
    
    return jsonify({
        'database': db_status,
        'processing_queue': {
            'processing': processing_count,
            'pending': pending_count
        },
        'active_models': active_models,
        'timestamp': datetime.utcnow().isoformat()
    })

def get_dashboard_stats():
    """Helper function to get dashboard statistics"""
    Employee = current_app.Employee
    AttendanceRecord = current_app.AttendanceRecord
    
    # Employee statistics
    total_employees = Employee.query.count()
    active_employees = Employee.query.filter_by(status='active').count()
    
    # Video processing statistics (optional feature)
    try:
        Video = getattr(current_app, 'Video', None)
        if Video:
            total_videos = Video.query.count()
            processed_videos = Video.query.filter_by(status='completed').count()
            processing_videos = Video.query.filter_by(status='processing').count()
            queue_status = {
                'pending': Video.query.filter_by(status='uploaded').count(),
                'processing': processing_videos,
                'completed': processed_videos,
                'failed': Video.query.filter_by(status='failed').count()
            }
        else:
            total_videos = processed_videos = processing_videos = 0
            queue_status = {'pending': 0, 'processing': 0, 'completed': 0, 'failed': 0}
    except Exception:
        total_videos = processed_videos = processing_videos = 0
        queue_status = {'pending': 0, 'processing': 0, 'completed': 0, 'failed': 0}
    
    # Detection statistics (optional feature)
    try:
        DetectedPerson = getattr(current_app, 'DetectedPerson', None)
        if DetectedPerson:
            total_detections = DetectedPerson.query.count()
            identified_persons = DetectedPerson.query.filter_by(is_identified=True).count()
        else:
            total_detections = identified_persons = 0
    except Exception:
        total_detections = identified_persons = 0
    
    # Model statistics (optional feature)
    try:
        TrainedModel = getattr(current_app, 'TrainedModel', None)
        if TrainedModel:
            total_models = TrainedModel.query.count()
            active_models = TrainedModel.query.filter_by(is_active=True).count()
        else:
            total_models = active_models = 0
    except Exception:
        total_models = active_models = 0
    
    # Today's attendance
    today = datetime.now().date()
    today_attendance = AttendanceRecord.query.filter_by(date=today).count()
    
    return {
        'employees': {
            'total': total_employees,
            'active': active_employees,
            'inactive': total_employees - active_employees
        },
        'videos': {
            'total': total_videos,
            'processed': processed_videos,
            'processing': processing_videos,
            'success_rate': (processed_videos / total_videos * 100) if total_videos > 0 else 0
        },
        'detections': {
            'total': total_detections,
            'identified': identified_persons,
            'unidentified': total_detections - identified_persons,
            'identification_rate': (identified_persons / total_detections * 100) if total_detections > 0 else 0
        },
        'models': {
            'total': total_models,
            'active': active_models,
            'inactive': total_models - active_models
        },
        'attendance': {
            'today': today_attendance,
            'rate': (today_attendance / active_employees * 100) if active_employees > 0 else 0
        },
        'queue': queue_status
    }