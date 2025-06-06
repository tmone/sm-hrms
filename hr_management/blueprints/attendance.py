"""
Attendance Reports Blueprint
Generate attendance reports based on OCR extracted timestamps and locations
"""

from flask import Blueprint, render_template, request, jsonify, current_app, send_file, flash
from flask_login import login_required
from datetime import datetime, timedelta, date
from sqlalchemy import func, and_, distinct
import pandas as pd
import io

attendance_bp = Blueprint('attendance', __name__, url_prefix='/attendance')

@attendance_bp.route('/')
@login_required
def index():
    """Display attendance reports dashboard"""
    return render_template('attendance/index.html')

@attendance_bp.route('/test')
def test():
    """Test page for attendance UI (no login required)"""
    return render_template('attendance/test.html')

@attendance_bp.route('/demo')
def demo():
    """Demo page showing what attendance should look like (no login required)"""
    return render_template('attendance/demo.html')

@attendance_bp.route('/summary')
@login_required
def summary():
    """Get attendance summary statistics"""
    db = current_app.db
    Video = current_app.Video
    DetectedPerson = current_app.DetectedPerson
    
    # Get days parameter (default 7 days)
    days = request.args.get('days', 7, type=int)
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    
    # Get videos with OCR data
    total_videos = Video.query.filter(
        Video.ocr_extraction_done == True
    ).count()
    
    # Get unique locations
    locations_query = db.session.query(
        Video.ocr_location,
        func.count(Video.id).label('count')
    ).filter(
        Video.ocr_location.isnot(None)
    ).group_by(Video.ocr_location).all()
    
    locations = {loc: count for loc, count in locations_query}
    
    # Get daily statistics
    daily_stats = {}
    current_date = start_date
    while current_date <= end_date:
        # Count unique persons for this date
        person_count = db.session.query(
            func.count(func.distinct(DetectedPerson.person_id))
        ).filter(
            DetectedPerson.attendance_date == current_date
        ).scalar() or 0
        
        daily_stats[current_date.isoformat()] = {
            'total_persons': person_count,
            'date': current_date.isoformat()
        }
        
        current_date += timedelta(days=1)
    
    return jsonify({
        'total_videos': total_videos,
        'locations': locations,
        'daily_stats': daily_stats,
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat()
    })

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
            # Use person_code if available, otherwise use person_id
            if hasattr(detection, 'person_code') and detection.person_code:
                person_id = detection.person_code
            elif detection.person_id:
                # person_id might be a string, so convert to int first
                try:
                    person_id = f"PERSON-{int(detection.person_id):04d}"
                except (ValueError, TypeError):
                    # If conversion fails, use as-is
                    person_id = f"PERSON-{detection.person_id}"
            else:
                person_id = f"UNKNOWN-{detection.id}"
            
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
            
            # Calculate time from video timestamp and detection timestamp
            if video.ocr_video_date and detection.timestamp is not None:
                # If we have OCR time, use it as base
                if video.ocr_video_time:
                    base_time = datetime.combine(video.ocr_video_date, video.ocr_video_time)
                else:
                    base_time = datetime.combine(video.ocr_video_date, datetime.min.time())
                
                detection_time = base_time + timedelta(seconds=detection.timestamp)
                
                if not person_attendance[person_id]['first_seen'] or detection_time < person_attendance[person_id]['first_seen']:
                    person_attendance[person_id]['first_seen'] = detection_time
                
                # For last seen, use the same timestamp (we'll update this for each detection)
                if not person_attendance[person_id]['last_seen'] or detection_time > person_attendance[person_id]['last_seen']:
                    person_attendance[person_id]['last_seen'] = detection_time
            
            person_attendance[person_id]['detections'].append({
                'timestamp': detection.timestamp,
                'confidence': detection.confidence
            })
        
        # Add video info and attendance data
        for person_data in person_attendance.values():
            if person_data['first_seen'] and person_data['last_seen']:
                duration = (person_data['last_seen'] - person_data['first_seen']).total_seconds()
                person_data['duration_seconds'] = int(duration)
                person_data['duration_minutes'] = round(duration / 60, 2)
                person_data['detection_count'] = len(person_data['detections'])
                person_data['avg_confidence'] = sum(d['confidence'] for d in person_data['detections']) / len(person_data['detections'])
                person_data['video_filename'] = video.filename
                person_data['video_id'] = video.id
                person_data['ocr_time'] = video.ocr_video_time
                # Calculate actual clock times if OCR time is available
                if video.ocr_video_time and person_data['detections']:
                    first_detection = min(person_data['detections'], key=lambda x: x['timestamp'])
                    last_detection = max(person_data['detections'], key=lambda x: x['timestamp'])
                    
                    # Store first timestamp for video navigation
                    person_data['first_timestamp'] = first_detection['timestamp']
                    
                    # Convert OCR time to datetime and add seconds
                    base_datetime = datetime.combine(video.ocr_video_date, video.ocr_video_time)
                    person_data['clock_in'] = (base_datetime + timedelta(seconds=first_detection['timestamp'])).time()
                    person_data['clock_out'] = (base_datetime + timedelta(seconds=last_detection['timestamp'])).time()
                attendance_data.append(person_data)
    
    # Sort by location and first seen time
    attendance_data.sort(key=lambda x: (x['location'] or '', x['first_seen'] or datetime.min))
    
    # Check if JSON format requested
    if request.args.get('format') == 'json':
        # Convert datetime objects to strings for JSON serialization
        for record in attendance_data:
            if record['first_seen']:
                record['first_seen'] = record['first_seen'].isoformat()
            if record['last_seen']:
                record['last_seen'] = record['last_seen'].isoformat()
            if record['date']:
                record['date'] = record['date'].isoformat()
            # Remove detections array from JSON response
            record.pop('detections', None)
        
        return jsonify({
            'report_date': report_date.isoformat(),
            'locations': locations,
            'location_filter': location_filter,
            'attendance_data': attendance_data
        })
    
    def format_duration(seconds):
        """Format duration in seconds to human readable format"""
        if not seconds:
            return "0 seconds"
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        parts = []
        if hours > 0:
            parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
        if minutes > 0:
            parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
        if secs > 0 or not parts:
            parts.append(f"{secs} second{'s' if secs != 1 else ''}")
        
        return " ".join(parts)
    
    return render_template('attendance/daily_report.html',
                         report_date=report_date,
                         locations=locations,
                         location_filter=location_filter,
                         attendance_data=attendance_data,
                         timedelta=timedelta,
                         format_duration=format_duration)

@attendance_bp.route('/list')
@login_required
def attendance_list():
    """Get paginated attendance list with filters"""
    db = current_app.db
    Video = current_app.Video
    DetectedPerson = current_app.DetectedPerson
    
    # Get parameters
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    filter_type = request.args.get('filter', 'all')  # today, week, month, custom, all
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    location = request.args.get('location', '')
    person_id_filter = request.args.get('person_id', '')
    sort = request.args.get('sort', 'desc')  # desc or asc
    
    # Build date filter
    today = date.today()
    if filter_type == 'today':
        date_filter = DetectedPerson.attendance_date == today
    elif filter_type == 'week':
        week_start = today - timedelta(days=today.weekday())
        date_filter = and_(
            DetectedPerson.attendance_date >= week_start,
            DetectedPerson.attendance_date <= today
        )
    elif filter_type == 'month':
        month_start = today.replace(day=1)
        date_filter = and_(
            DetectedPerson.attendance_date >= month_start,
            DetectedPerson.attendance_date <= today
        )
    elif filter_type == 'custom' and start_date and end_date:
        start = datetime.strptime(start_date, '%Y-%m-%d').date()
        end = datetime.strptime(end_date, '%Y-%m-%d').date()
        date_filter = and_(
            DetectedPerson.attendance_date >= start,
            DetectedPerson.attendance_date <= end
        )
    else:
        # Show all data
        date_filter = DetectedPerson.attendance_date.isnot(None)
    
    # Build query - aggregate by person, date, and location
    base_query = db.session.query(
        DetectedPerson.person_id,
        DetectedPerson.attendance_date,
        DetectedPerson.attendance_location,
        func.min(DetectedPerson.attendance_time).label('check_in'),
        func.max(DetectedPerson.attendance_time).label('check_out'),
        func.count(DetectedPerson.id).label('detection_count'),
        func.avg(DetectedPerson.confidence).label('avg_confidence'),
        DetectedPerson.video_id,
        func.min(DetectedPerson.timestamp).label('first_timestamp')
    ).filter(
        date_filter,
        DetectedPerson.attendance_date.isnot(None)
    ).group_by(
        DetectedPerson.person_id,
        DetectedPerson.attendance_date,
        DetectedPerson.attendance_location,
        DetectedPerson.video_id
    )
    
    # Apply location filter if specified
    if location:
        base_query = base_query.filter(DetectedPerson.attendance_location == location)
    
    # Apply person ID filter if specified
    if person_id_filter:
        # Extract numeric ID from PERSON-XXXX format if provided
        if person_id_filter.startswith('PERSON-'):
            try:
                person_id_num = person_id_filter.split('-')[1]
                base_query = base_query.filter(DetectedPerson.person_id == person_id_num)
            except:
                # If format is invalid, try exact match
                base_query = base_query.filter(DetectedPerson.person_id == person_id_filter)
        else:
            # Direct ID match
            base_query = base_query.filter(DetectedPerson.person_id == person_id_filter)
    
    # Apply sorting
    if sort == 'desc':
        base_query = base_query.order_by(
            DetectedPerson.attendance_date.desc(),
            DetectedPerson.attendance_location,
            DetectedPerson.person_id
        )
    else:
        base_query = base_query.order_by(
            DetectedPerson.attendance_date.asc(),
            DetectedPerson.attendance_location,
            DetectedPerson.person_id
        )
    
    # Paginate
    pagination = base_query.paginate(page=page, per_page=per_page, error_out=False)
    
    # Define format_duration helper function
    def format_duration(seconds):
        """Format duration helper"""
        if not seconds:
            return "0m"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"
    
    # Format results
    attendance_records = []
    for record in pagination.items:
        # Get video info
        video = Video.query.get(record.video_id)
        
        # Calculate duration
        if record.check_in and record.check_out:
            check_in_time = datetime.combine(record.attendance_date, record.check_in)
            check_out_time = datetime.combine(record.attendance_date, record.check_out)
            duration = (check_out_time - check_in_time).total_seconds()
        else:
            duration = 0
        
        # Format person ID
        try:
            person_id_str = f"PERSON-{int(record.person_id):04d}"
        except (ValueError, TypeError):
            person_id_str = str(record.person_id)
        
        attendance_records.append({
            'id': f"{record.person_id}_{record.attendance_date}_{record.video_id}",
            'person_id': person_id_str,
            'date': record.attendance_date.isoformat(),
            'location': record.attendance_location or 'Unknown',
            'check_in': record.check_in.isoformat() if record.check_in else None,
            'check_out': record.check_out.isoformat() if record.check_out else None,
            'duration_seconds': int(duration),
            'duration_formatted': format_duration(int(duration)),
            'detection_count': record.detection_count,
            'confidence': round(record.avg_confidence, 2) if record.avg_confidence else 0,
            'video_filename': video.filename if video else 'Unknown',
            'video_id': record.video_id,
            'first_timestamp': record.first_timestamp
        })
    
    # Get available locations for filter
    all_locations = db.session.query(
        func.distinct(DetectedPerson.attendance_location)
    ).filter(
        DetectedPerson.attendance_location.isnot(None)
    ).all()
    locations = [loc[0] for loc in all_locations]
    
    return jsonify({
        'records': attendance_records,
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total': pagination.total,
            'pages': pagination.pages,
            'has_prev': pagination.has_prev,
            'has_next': pagination.has_next
        },
        'filters': {
            'filter_type': filter_type,
            'start_date': start_date,
            'end_date': end_date,
            'location': location,
            'person_id': person_id_filter,
            'locations': locations
        }
    })

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
        
        # Group detections by person_id to get first/last seen times
        person_detections = {}
        for detection in detections:
            if detection.person_id:
                if detection.person_id not in person_detections:
                    person_detections[detection.person_id] = []
                person_detections[detection.person_id].append(detection)
        
        # Process each person's detections
        for person_id, person_dets in person_detections.items():
            if video.ocr_video_date and len(person_dets) > 0:
                # Get timestamps
                timestamps = [d.timestamp for d in person_dets if d.timestamp is not None]
                if not timestamps:
                    continue
                
                first_timestamp = min(timestamps)
                last_timestamp = max(timestamps)
                
                # Calculate actual times
                base_time = datetime.combine(video.ocr_video_date, video.ocr_video_time) if video.ocr_video_time else datetime.combine(video.ocr_video_date, datetime.min.time())
                first_seen = base_time + timedelta(seconds=first_timestamp)
                last_seen = base_time + timedelta(seconds=last_timestamp)
                duration_minutes = round((last_timestamp - first_timestamp) / 60, 2)
                
                # Format person ID
                try:
                    person_id_str = f"PERSON-{int(person_id):04d}"
                except (ValueError, TypeError):
                    person_id_str = f"PERSON-{person_id}"
                
                # Get average confidence
                confidences = [d.confidence for d in person_dets if d.confidence is not None]
                avg_confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0
                
                record = {
                    'Date': video.ocr_video_date.strftime('%Y-%m-%d'),
                    'Location': video.ocr_location or 'Unknown',
                    'Person ID': person_id_str,
                    'Employee ID': person_dets[0].employee_id or 'Not Assigned',
                    'First Seen': first_seen,
                    'Last Seen': last_seen,
                    'Duration (minutes)': duration_minutes,
                    'Detections': len(person_dets),
                    'Avg Confidence': avg_confidence,
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
        try:
            # Try to use openpyxl for Excel export
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
        except ImportError:
            # If openpyxl is not installed, fall back to CSV
            flash('Excel export requires openpyxl. Exporting as CSV instead.', 'warning')
            format = 'csv'
    
    if format == 'csv' or format == 'excel':  # Excel falls through to here if openpyxl missing
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

@attendance_bp.route('/export-selected', methods=['POST'])
@login_required
def export_selected():
    """Export selected attendance records to CSV"""
    db = current_app.db
    Video = current_app.Video
    DetectedPerson = current_app.DetectedPerson
    
    # Get selected IDs from request
    data = request.get_json()
    selected_ids = data.get('selected_ids', [])
    
    if not selected_ids:
        return jsonify({'error': 'No records selected'}), 400
    
    # Parse IDs and build query
    records = []
    for record_id in selected_ids:
        try:
            # ID format: personId_date_videoId
            parts = record_id.split('_')
            if len(parts) >= 3:
                person_id = parts[0]
                date_str = parts[1]
                video_id = parts[2]
                
                # Get aggregated data for this person/date/video
                record_data = db.session.query(
                    DetectedPerson.person_id,
                    DetectedPerson.attendance_date,
                    DetectedPerson.attendance_location,
                    func.min(DetectedPerson.attendance_time).label('check_in'),
                    func.max(DetectedPerson.attendance_time).label('check_out'),
                    func.count(DetectedPerson.id).label('detection_count'),
                    func.avg(DetectedPerson.confidence).label('avg_confidence')
                ).filter(
                    DetectedPerson.person_id == person_id,
                    DetectedPerson.attendance_date == datetime.strptime(date_str, '%Y-%m-%d').date(),
                    DetectedPerson.video_id == int(video_id)
                ).group_by(
                    DetectedPerson.person_id,
                    DetectedPerson.attendance_date,
                    DetectedPerson.attendance_location
                ).first()
                
                if record_data:
                    # Calculate duration
                    if record_data.check_in and record_data.check_out:
                        check_in_time = datetime.combine(record_data.attendance_date, record_data.check_in)
                        check_out_time = datetime.combine(record_data.attendance_date, record_data.check_out)
                        duration_minutes = round((check_out_time - check_in_time).total_seconds() / 60, 2)
                    else:
                        duration_minutes = 0
                    
                    # Format person ID
                    try:
                        person_id_str = f"PERSON-{int(record_data.person_id):04d}"
                    except (ValueError, TypeError):
                        person_id_str = str(record_data.person_id)
                    
                    records.append({
                        'Date': record_data.attendance_date.strftime('%Y-%m-%d'),
                        'Person ID': person_id_str,
                        'Location': record_data.attendance_location or 'Unknown',
                        'Check In': record_data.check_in.strftime('%H:%M:%S') if record_data.check_in else '',
                        'Check Out': record_data.check_out.strftime('%H:%M:%S') if record_data.check_out else '',
                        'Duration (minutes)': duration_minutes,
                        'Detections': record_data.detection_count,
                        'Avg Confidence': round(record_data.avg_confidence, 2) if record_data.avg_confidence else 0
                    })
        except Exception as e:
            print(f"Error processing record {record_id}: {e}")
            continue
    
    if not records:
        return jsonify({'error': 'No valid records found'}), 404
    
    # Create CSV
    df = pd.DataFrame(records)
    df = df.sort_values(['Date', 'Location', 'Person ID'])
    
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    # Create response
    response = send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'attendance_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )
    
    return response