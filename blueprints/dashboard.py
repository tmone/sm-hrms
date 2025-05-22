from flask import Blueprint, render_template, jsonify
from flask_login import login_required
from models.employee import Employee, AttendanceRecord
from models.video import Video, DetectedPerson
from models.face_recognition import TrainedModel, FaceDataset
from models.base import db
from datetime import datetime, timedelta
from sqlalchemy import func

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/')
@login_required
def index():
    # Get summary statistics
    stats = get_dashboard_stats()
    
    # Get recent activities
    recent_videos = Video.query.order_by(Video.created_at.desc()).limit(5).all()
    recent_employees = Employee.query.order_by(Employee.created_at.desc()).limit(5).all()
    
    return render_template('dashboard/index.html', 
                         stats=stats,
                         recent_videos=recent_videos,
                         recent_employees=recent_employees)

@dashboard_bp.route('/api/stats')
@login_required
def api_stats():
    return jsonify(get_dashboard_stats())

def get_dashboard_stats():
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

@dashboard_bp.route('/processing-status')
@login_required
def processing_status():
    # Get current processing status for real-time updates
    processing_videos = Video.query.filter_by(status='processing').all()
    
    status_data = []
    for video in processing_videos:
        status_data.append({
            'id': video.id,
            'filename': video.filename,
            'progress': video.processing_progress,
            'status': video.status
        })
    
    return jsonify(status_data)