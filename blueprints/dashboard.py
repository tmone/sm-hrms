from flask import Blueprint, render_template, jsonify, current_app
from flask_login import login_required
from datetime import datetime, timedelta
from sqlalchemy import func

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/')
@login_required
def index():
    # Get models from current app
    Employee = current_app.Employee
    
    # Get summary statistics
    stats = get_dashboard_stats()
    
    # Get recent employees (simplified for now)
    recent_employees = Employee.query.order_by(Employee.created_at.desc()).limit(5).all()
    
    return render_template('dashboard/index.html', 
                         stats=stats,
                         recent_videos=[],  # Empty for now
                         recent_employees=recent_employees)

@dashboard_bp.route('/api/stats')
@login_required
def api_stats():
    return jsonify(get_dashboard_stats())

def get_dashboard_stats():
    # Get models from current app
    Employee = current_app.Employee
    AttendanceRecord = current_app.AttendanceRecord
    
    # Employee statistics
    total_employees = Employee.query.count()
    active_employees = Employee.query.filter_by(status='active').count()
    
    # Simplified stats for now (video features not implemented yet)
    total_videos = 0
    processed_videos = 0
    processing_videos = 0
    
    # Simplified detection stats
    total_detections = 0
    identified_persons = 0
    
    # Simplified model stats
    total_models = 0
    active_models = 0
    
    # Today's attendance
    today = datetime.now().date()
    today_attendance = AttendanceRecord.query.filter_by(date=today).count()
    
    # Processing queue status (simplified)
    queue_status = {
        'pending': 0,
        'processing': processing_videos,
        'completed': processed_videos,
        'failed': 0
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