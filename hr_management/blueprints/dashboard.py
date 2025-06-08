from flask import Blueprint, render_template, jsonify, current_app
from flask_login import login_required
from datetime import datetime, timedelta
from sqlalchemy import func
import os
import json

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/')
@login_required
def index():
    # Get models from current app
    Employee = current_app.Employee
    Video = current_app.Video
    
    # Get summary statistics
    stats = get_dashboard_stats()
    
    # Get recent videos
    recent_videos = Video.query.order_by(Video.created_at.desc()).limit(5).all()
    
    # Get recent employees
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
    # Get models from current app
    Employee = current_app.Employee
    AttendanceRecord = current_app.AttendanceRecord
    Video = current_app.Video
    DetectedPerson = current_app.DetectedPerson
    
    # Employee statistics
    total_employees = Employee.query.count()
    active_employees = Employee.query.filter_by(status='active').count()
    
    # Video statistics
    total_videos = Video.query.count()
    processed_videos = Video.query.filter_by(status='completed').count()
    processing_videos = Video.query.filter_by(status='processing').count()
    failed_videos = Video.query.filter_by(status='failed').count()
    
    # Detection statistics
    total_detections = DetectedPerson.query.count()
    identified_persons = DetectedPerson.query.filter_by(is_identified=True).count()
    
    # Count persons from filesystem (reviewed persons with images)
    persons_dir = os.path.join(current_app.root_path, 'processing', 'outputs', 'persons')
    unique_persons = 0
    
    if os.path.exists(persons_dir):
        for person_dir in os.listdir(persons_dir):
            if os.path.isdir(os.path.join(persons_dir, person_dir)) and person_dir.startswith('PERSON-'):
                metadata_path = os.path.join(persons_dir, person_dir, 'metadata.json')
                if os.path.exists(metadata_path):
                    unique_persons += 1
    
    # Person recognition model statistics
    models_path = os.path.join(current_app.root_path, 'models', 'person_recognition')
    total_models = 0
    active_models = 0
    
    if os.path.exists(models_path):
        for model_dir in os.listdir(models_path):
            model_metadata_path = os.path.join(models_path, model_dir, 'metadata.json')
            if os.path.exists(model_metadata_path):
                total_models += 1
                try:
                    with open(model_metadata_path, 'r') as f:
                        metadata = json.load(f)
                        if metadata.get('is_default', False):
                            active_models += 1
                except:
                    pass
    
    # Dataset statistics
    datasets_path = os.path.join(current_app.root_path, 'datasets', 'person_recognition')
    total_datasets = 0
    
    if os.path.exists(datasets_path):
        for dataset_dir in os.listdir(datasets_path):
            dataset_info_path = os.path.join(datasets_path, dataset_dir, 'dataset_info.json')
            if os.path.exists(dataset_info_path):
                total_datasets += 1
    
    # Today's attendance
    today = datetime.now().date()
    today_attendance = AttendanceRecord.query.filter_by(date=today).count()
    
    # Processing queue status
    pending_videos = Video.query.filter_by(status='uploaded').count()
    
    queue_status = {
        'pending': pending_videos,
        'processing': processing_videos,
        'completed': processed_videos,
        'failed': failed_videos
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
            'unique_persons': unique_persons,
            'identification_rate': (identified_persons / total_detections * 100) if total_detections > 0 else 0
        },
        'models': {
            'total': total_models,
            'active': active_models,
            'inactive': total_models - active_models
        },
        'datasets': {
            'total': total_datasets
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
    Video = current_app.Video
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