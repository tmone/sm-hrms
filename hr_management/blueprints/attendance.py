"""
Attendance Reports Blueprint
Generate attendance reports based on OCR extracted timestamps and locations
"""

from flask import Blueprint, render_template, request, jsonify, current_app, send_file
from flask_login import login_required
from datetime import datetime, timedelta, date
from sqlalchemy import func, and_
import pandas as pd
import io

attendance_bp = Blueprint('attendance', __name__, url_prefix='/attendance')

@attendance_bp.route('/')
@login_required
def index():
    """Display attendance reports dashboard"""
    return render_template('attendance/index.html')

@attendance_bp.route('/daily')
@login_required
def daily_report():
    """Generate daily attendance report"""
    db = current_app.db
    AttendanceSummary = current_app.AttendanceSummary if hasattr(current_app, 'AttendanceSummary') else None
    Video = current_app.Video
    DetectedPerson = current_app.DetectedPerson
    
    # Get date from query params or use today
    report_date = request.args.get('date')
    if report_date:
        report_date = datetime.strptime(report_date, '%Y-%m-%d').date()
    else:
        report_date = date.today()
    
    # Get location filter if any
    location_filter = request.args.get('location', '')
    
    # Query videos with OCR data for the date
    videos_query = Video.query.filter(
        Video.ocr_video_date == report_date,
        Video.ocr_extraction_done == True
    )
    
    if location_filter:
        videos_query = videos_query.filter(Video.ocr_location == location_filter)
    
    videos = videos_query.all()
    
    # Get all unique locations
    locations = db.session.query(Video.ocr_location).filter(
        Video.ocr_location.isnot(None)
    ).distinct().all()
    locations = [loc[0] for loc in locations]
    
    # Get attendance data
    attendance_data = []
    
    for video in videos:
        # Get all detected persons in this video
        detections = DetectedPerson.query.filter_by(video_id=video.id).all()
        
        # Group by person_id
        person_attendance = {}
        
        for detection in detections:
            person_id = f"PERSON-{detection.person_id:04d}" if detection.person_id else f"UNKNOWN-{detection.id}"
            
            if person_id not in person_attendance:
                person_attendance[person_id] = {
                    'person_id': person_id,
                    'employee_id': detection.employee_id,
                    'location': video.ocr_location,
                    'date': video.ocr_video_date,
                    'first_seen': None,
                    'last_seen': None,
                    'detections': []
                }
            
            # Calculate time from video timestamp and detection frame time
            if video.ocr_video_date and detection.start_time is not None:
                detection_time = datetime.combine(video.ocr_video_date, datetime.min.time()) + timedelta(seconds=detection.start_time)
                
                if not person_attendance[person_id]['first_seen'] or detection_time < person_attendance[person_id]['first_seen']:
                    person_attendance[person_id]['first_seen'] = detection_time
                    
                if detection.end_time:
                    end_time = datetime.combine(video.ocr_video_date, datetime.min.time()) + timedelta(seconds=detection.end_time)
                    if not person_attendance[person_id]['last_seen'] or end_time > person_attendance[person_id]['last_seen']:
                        person_attendance[person_id]['last_seen'] = end_time
            
            person_attendance[person_id]['detections'].append({
                'start_time': detection.start_time,
                'end_time': detection.end_time,
                'confidence': detection.confidence
            })
        
        # Add to attendance data
        for person_data in person_attendance.values():
            if person_data['first_seen'] and person_data['last_seen']:
                duration = (person_data['last_seen'] - person_data['first_seen']).total_seconds() / 60  # in minutes
                person_data['duration_minutes'] = round(duration, 2)
                person_data['detection_count'] = len(person_data['detections'])
                person_data['avg_confidence'] = sum(d['confidence'] for d in person_data['detections']) / len(person_data['detections'])
                attendance_data.append(person_data)
    
    # Sort by location and first seen time
    attendance_data.sort(key=lambda x: (x['location'] or '', x['first_seen'] or datetime.min))
    
    return render_template('attendance/daily_report.html',
                         report_date=report_date,
                         locations=locations,
                         location_filter=location_filter,
                         attendance_data=attendance_data)

@attendance_bp.route('/export')
@login_required
def export_report():
    """Export attendance report to Excel"""
    db = current_app.db
    Video = current_app.Video
    DetectedPerson = current_app.DetectedPerson
    
    # Get parameters
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    location = request.args.get('location', '')
    format = request.args.get('format', 'excel')
    
    if not start_date or not end_date:
        return jsonify({'error': 'Start and end dates are required'}), 400
    
    start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
    
    # Query videos in date range
    videos_query = Video.query.filter(
        Video.ocr_video_date >= start_date,
        Video.ocr_video_date <= end_date,
        Video.ocr_extraction_done == True
    )
    
    if location:
        videos_query = videos_query.filter(Video.ocr_location == location)
    
    videos = videos_query.all()
    
    # Collect all attendance data
    all_attendance = []
    
    for video in videos:
        detections = DetectedPerson.query.filter_by(video_id=video.id).all()
        
        for detection in detections:
            if video.ocr_video_date and detection.start_time is not None:
                record = {
                    'Date': video.ocr_video_date.strftime('%Y-%m-%d'),
                    'Location': video.ocr_location or 'Unknown',
                    'Person ID': f"PERSON-{detection.person_id:04d}" if detection.person_id else f"UNKNOWN-{detection.id}",
                    'Employee ID': detection.employee_id or 'Not Assigned',
                    'First Seen': datetime.combine(video.ocr_video_date, datetime.min.time()) + timedelta(seconds=detection.start_time),
                    'Last Seen': datetime.combine(video.ocr_video_date, datetime.min.time()) + timedelta(seconds=detection.end_time) if detection.end_time else None,
                    'Duration (minutes)': round((detection.end_time - detection.start_time) / 60, 2) if detection.end_time else 0,
                    'Confidence': round(detection.confidence, 2) if detection.confidence else 0,
                    'Video': video.filename
                }
                all_attendance.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(all_attendance)
    
    if df.empty:
        return jsonify({'error': 'No attendance data found for the specified criteria'}), 404
    
    # Sort by date and location
    df = df.sort_values(['Date', 'Location', 'First Seen'])
    
    # Generate file
    if format == 'excel':
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Attendance Report', index=False)
            
            # Auto-adjust column widths
            worksheet = writer.sheets['Attendance Report']
            for column in worksheet.columns:
                max_length = 0
                column = [cell for cell in column]
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column[0].column_letter].width = adjusted_width
        
        output.seek(0)
        
        filename = f"attendance_report_{start_date}_{end_date}.xlsx"
        return send_file(output, 
                        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        as_attachment=True,
                        download_name=filename)
    
    elif format == 'csv':
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        filename = f"attendance_report_{start_date}_{end_date}.csv"
        return send_file(io.BytesIO(output.getvalue().encode()),
                        mimetype='text/csv',
                        as_attachment=True,
                        download_name=filename)
    
    else:
        return jsonify({'error': 'Invalid format. Use excel or csv'}), 400

@attendance_bp.route('/summary')
@login_required
def attendance_summary():
    """Get attendance summary statistics"""
    db = current_app.db
    Video = current_app.Video
    DetectedPerson = current_app.DetectedPerson
    
    # Get date range
    days = int(request.args.get('days', 7))
    end_date = date.today()
    start_date = end_date - timedelta(days=days-1)
    
    # Get videos with OCR data
    videos = Video.query.filter(
        Video.ocr_video_date >= start_date,
        Video.ocr_video_date <= end_date,
        Video.ocr_extraction_done == True
    ).all()
    
    # Calculate statistics
    stats = {
        'total_days': days,
        'total_videos': len(videos),
        'locations': {},
        'daily_stats': {},
        'unique_persons': set()
    }
    
    for video in videos:
        date_str = video.ocr_video_date.strftime('%Y-%m-%d')
        location = video.ocr_location or 'Unknown'
        
        # Initialize daily stats
        if date_str not in stats['daily_stats']:
            stats['daily_stats'][date_str] = {
                'total_persons': 0,
                'total_detections': 0,
                'locations': set()
            }
        
        # Initialize location stats
        if location not in stats['locations']:
            stats['locations'][location] = {
                'total_days': 0,
                'total_persons': 0,
                'total_detections': 0
            }
        
        # Get detections
        detections = DetectedPerson.query.filter_by(video_id=video.id).all()
        unique_persons_in_video = set()
        
        for detection in detections:
            person_id = f"PERSON-{detection.person_id:04d}" if detection.person_id else f"UNKNOWN-{detection.id}"
            unique_persons_in_video.add(person_id)
            stats['unique_persons'].add(person_id)
        
        # Update stats
        stats['daily_stats'][date_str]['total_persons'] += len(unique_persons_in_video)
        stats['daily_stats'][date_str]['total_detections'] += len(detections)
        stats['daily_stats'][date_str]['locations'].add(location)
        
        stats['locations'][location]['total_persons'] += len(unique_persons_in_video)
        stats['locations'][location]['total_detections'] += len(detections)
    
    # Convert sets to counts
    stats['total_unique_persons'] = len(stats['unique_persons'])
    stats['unique_persons'] = None  # Remove the set from response
    
    # Convert location sets to lists
    for date_str in stats['daily_stats']:
        stats['daily_stats'][date_str]['locations'] = list(stats['daily_stats'][date_str]['locations'])
    
    return jsonify(stats)