from flask import Blueprint, jsonify, request
from flask_login import login_required
from models.employee import Employee, AttendanceRecord
from models.video import Video, DetectedPerson
from models.face_recognition import TrainedModel, FaceDataset, RecognitionResult
from models.base import db
from datetime import datetime, date
from sqlalchemy import func

api_bp = Blueprint('api', __name__)

@api_bp.route('/stats')
@login_required
def stats():
    """Get dashboard statistics"""
    return jsonify(get_dashboard_stats())

@api_bp.route('/processing-queue')
@login_required
def processing_queue():
    """Get processing queue status"""
    queue_count = Video.query.filter(Video.status.in_(['uploaded', 'processing'])).count()
    return jsonify({'count': queue_count})

@api_bp.route('/employees')
@login_required
def employees():
    """Get employees list"""
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
    
    query = Video.query
    if status:
        query = query.filter_by(status=status)
    
    videos = query.order_by(Video.created_at.desc()).all()
    return jsonify([video.to_dict() for video in videos])

@api_bp.route('/videos/<int:id>/detections')
@login_required
def video_detections(id):
    """Get detections for a specific video"""
    detections = DetectedPerson.query.filter_by(video_id=id).all()
    return jsonify([detection.to_dict() for detection in detections])

@api_bp.route('/attendance/today')
@login_required
def attendance_today():
    """Get today's attendance"""
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
    data = request.get_json()
    
    detection_id = data.get('detection_id')
    employee_id = data.get('employee_id')
    
    if not detection_id:
        return jsonify({'error': 'Missing detection_id'}), 400
    
    detection = DetectedPerson.query.get(detection_id)
    if not detection:
        return jsonify({'error': 'Detection not found'}), 404
    
    try:
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
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@api_bp.route('/models/performance')
@login_required
def models_performance():
    """Get model performance metrics"""
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

@api_bp.route('/system/health')
@login_required
def system_health():
    """Get system health status"""
    try:
        # Test database connection
        db.session.execute('SELECT 1')
        db_status = 'healthy'
    except Exception:
        db_status = 'error'
    
    # Get processing queue status
    processing_count = Video.query.filter_by(status='processing').count()
    pending_count = Video.query.filter_by(status='uploaded').count()
    
    # Get active models
    active_models = TrainedModel.query.filter_by(is_active=True).count()
    
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
    # Employee statistics
    total_employees = Employee.query.count()
    active_employees = Employee.query.filter_by(status='active').count()
    
    # Video processing statistics
    total_videos = Video.query.count()
    processed_videos = Video.query.filter_by(status='completed').count()
    processing_videos = Video.query.filter_by(status='processing').count()
    
    # Detection statistics
    total_detections = DetectedPerson.query.count()
    identified_persons = DetectedPerson.query.filter_by(is_identified=True).count()
    
    # Model statistics
    total_models = TrainedModel.query.count()
    active_models = TrainedModel.query.filter_by(is_active=True).count()
    
    # Today's attendance
    today = datetime.now().date()
    today_attendance = AttendanceRecord.query.filter_by(date=today).count()
    
    # Processing queue status
    queue_status = {
        'pending': Video.query.filter_by(status='uploaded').count(),
        'processing': processing_videos,
        'completed': processed_videos,
        'failed': Video.query.filter_by(status='failed').count()
    }
    
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