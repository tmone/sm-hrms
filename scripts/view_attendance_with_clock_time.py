#!/usr/bin/env python3
"""
View person attendance with actual clock times from OCR data
"""

import sys
import os
from datetime import datetime, date, timedelta
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app

def calculate_actual_time(base_time, video_offset_seconds):
    """Calculate actual clock time from OCR base time + video offset"""
    if not base_time:
        return None
    
    # Create datetime with base time
    base_datetime = datetime.combine(date.today(), base_time)
    # Add video offset
    actual_datetime = base_datetime + timedelta(seconds=video_offset_seconds)
    
    return actual_datetime.time()

def view_attendance_with_clock_time(date_filter=None, location_filter=None):
    """View attendance logs with actual clock times"""
    app = create_app()
    
    with app.app_context():
        Video = app.Video
        DetectedPerson = app.DetectedPerson
        db = app.db
        
        # Get videos with OCR data
        videos_query = Video.query.filter(
            Video.ocr_extraction_done == True,
            Video.ocr_video_time.isnot(None)
        )
        
        if date_filter:
            videos_query = videos_query.filter(Video.ocr_video_date == date_filter)
        
        if location_filter:
            videos_query = videos_query.filter(Video.ocr_location == location_filter)
        
        videos = videos_query.all()
        
        if not videos:
            print("No videos found with OCR time data.")
            return
        
        print("\n" + "="*80)
        print("PERSON ATTENDANCE WITH ACTUAL CLOCK TIMES")
        print("="*80)
        
        for video in videos:
            print(f"\nVideo: {video.filename}")
            print(f"Location: {video.ocr_location}")
            print(f"Date: {video.ocr_video_date}")
            print(f"OCR Base Time: {video.ocr_video_time}")
            print("-" * 60)
            
            # Get all person detections for this video
            detections_by_person = defaultdict(list)
            
            detections = DetectedPerson.query.filter_by(
                video_id=video.id
            ).filter(
                DetectedPerson.person_id.isnot(None)
            ).order_by(
                DetectedPerson.timestamp
            ).all()
            
            for detection in detections:
                detections_by_person[detection.person_id].append(detection)
            
            # Display each person's attendance
            for person_id, person_detections in sorted(detections_by_person.items()):
                if not person_detections:
                    continue
                
                # Get first and last detection
                first_detection = person_detections[0]
                last_detection = person_detections[-1]
                
                # Calculate actual clock times
                first_clock_time = calculate_actual_time(
                    video.ocr_video_time, 
                    first_detection.timestamp
                )
                last_clock_time = calculate_actual_time(
                    video.ocr_video_time,
                    last_detection.timestamp
                )
                
                print(f"\n  Person: {person_id}")
                print(f"  First Seen: {first_clock_time} (video time: {first_detection.timestamp:.1f}s)")
                print(f"  Last Seen: {last_clock_time} (video time: {last_detection.timestamp:.1f}s)")
                print(f"  Total Detections: {len(person_detections)}")
                
                # Calculate presence duration
                if first_detection.timestamp != last_detection.timestamp:
                    duration_seconds = last_detection.timestamp - first_detection.timestamp
                    hours, remainder = divmod(duration_seconds, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    print(f"  Presence Duration: {int(hours)}h {int(minutes)}m {int(seconds)}s")
                
                # Show confidence
                avg_confidence = sum(d.confidence for d in person_detections) / len(person_detections)
                print(f"  Average Confidence: {avg_confidence:.1%}")

def create_attendance_report(date_filter=None):
    """Create a formatted attendance report"""
    app = create_app()
    
    with app.app_context():
        from sqlalchemy import func
        Video = app.Video
        DetectedPerson = app.DetectedPerson
        db = app.db
        
        # Get all attendance records for the date
        query = db.session.query(
            DetectedPerson.person_id,
            DetectedPerson.attendance_location,
            Video.ocr_video_time,
            func.min(DetectedPerson.timestamp).label('first_seen'),
            func.max(DetectedPerson.timestamp).label('last_seen'),
            func.count(DetectedPerson.id).label('detection_count'),
            func.avg(DetectedPerson.confidence).label('avg_confidence')
        ).join(Video).filter(
            Video.ocr_video_time.isnot(None)
        ).group_by(
            DetectedPerson.person_id,
            DetectedPerson.attendance_location,
            Video.ocr_video_time
        )
        
        if date_filter:
            query = query.filter(Video.ocr_video_date == date_filter)
        
        results = query.all()
        
        if not results:
            print("No attendance data found.")
            return
        
        print("\n" + "="*80)
        print(f"ATTENDANCE REPORT - {date_filter if date_filter else 'All Dates'}")
        print("="*80)
        print(f"{'Person ID':<12} {'Location':<15} {'Check In':<10} {'Check Out':<10} {'Duration':<10} {'Detections':<12} {'Confidence':<10}")
        print("-" * 80)
        
        for row in results:
            person_id, location, base_time, first_seen, last_seen, count, confidence = row
            
            # Calculate actual times
            check_in = calculate_actual_time(base_time, first_seen)
            check_out = calculate_actual_time(base_time, last_seen)
            
            # Calculate duration
            duration_seconds = last_seen - first_seen
            if duration_seconds > 0:
                hours, remainder = divmod(duration_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                duration_str = f"{int(hours)}h {int(minutes)}m"
            else:
                duration_str = "-"
            
            print(f"{person_id:<12} {location:<15} {check_in.strftime('%H:%M:%S') if check_in else '-':<10} "
                  f"{check_out.strftime('%H:%M:%S') if check_out else '-':<10} {duration_str:<10} "
                  f"{count:<12} {confidence:.1%}" if confidence else "N/A")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='View attendance with actual clock times')
    parser.add_argument('--date', type=str, help='Filter by date (YYYY-MM-DD)')
    parser.add_argument('--location', type=str, help='Filter by location')
    parser.add_argument('--report', action='store_true', help='Generate summary report')
    
    args = parser.parse_args()
    
    # Parse date if provided
    date_filter = None
    if args.date:
        try:
            date_filter = datetime.strptime(args.date, '%Y-%m-%d').date()
        except ValueError:
            print(f"Invalid date format: {args.date}. Use YYYY-MM-DD")
            return
    
    if args.report:
        create_attendance_report(date_filter)
    else:
        view_attendance_with_clock_time(date_filter, args.location)

if __name__ == '__main__':
    main()