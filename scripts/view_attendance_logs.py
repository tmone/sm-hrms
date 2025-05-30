#!/usr/bin/env python3
"""
View person attendance time logs from OCR-extracted data
"""

import sys
import os
from datetime import datetime, date
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app

def view_attendance_logs(date_filter=None, location_filter=None, person_filter=None):
    """View attendance logs with optional filters"""
    app = create_app()
    
    with app.app_context():
        Video = app.Video
        DetectedPerson = app.DetectedPerson
        db = app.db
        
        # Build query
        query = DetectedPerson.query.join(Video)
        
        # Apply filters
        if date_filter:
            query = query.filter(DetectedPerson.attendance_date == date_filter)
        
        if location_filter:
            query = query.filter(DetectedPerson.attendance_location == location_filter)
            
        if person_filter:
            query = query.filter(DetectedPerson.person_id == person_filter)
        
        # Get detections with attendance data
        detections = query.filter(
            DetectedPerson.attendance_date.isnot(None)
        ).order_by(
            DetectedPerson.attendance_date.desc(),
            DetectedPerson.attendance_time
        ).all()
        
        if not detections:
            print("No attendance records found with the given filters.")
            return
        
        # Group by date and person
        attendance_by_date = defaultdict(lambda: defaultdict(list))
        
        for detection in detections:
            date_key = detection.attendance_date
            person_key = detection.person_id or f"Unknown-{detection.id}"
            attendance_by_date[date_key][person_key].append(detection)
        
        # Display results
        print("\n" + "="*80)
        print("PERSON ATTENDANCE TIME LOGS")
        print("="*80)
        
        for att_date, persons in sorted(attendance_by_date.items(), reverse=True):
            print(f"\nDate: {att_date}")
            print("-" * 60)
            
            for person_id, person_detections in sorted(persons.items()):
                # Get first and last detection times
                times = sorted([d.attendance_time for d in person_detections if d.attendance_time])
                if times:
                    first_seen = times[0]
                    last_seen = times[-1]
                    location = person_detections[0].attendance_location
                    video_name = person_detections[0].video.filename
                    
                    print(f"\n  Person: {person_id}")
                    print(f"  Location: {location}")
                    print(f"  First Seen: {first_seen}")
                    print(f"  Last Seen: {last_seen}")
                    print(f"  Total Detections: {len(person_detections)}")
                    print(f"  Video: {video_name}")
                    
                    # Calculate duration if different times
                    if first_seen != last_seen:
                        # Convert times to datetime for calculation
                        dt1 = datetime.combine(date.today(), first_seen)
                        dt2 = datetime.combine(date.today(), last_seen)
                        duration = dt2 - dt1
                        hours, remainder = divmod(duration.total_seconds(), 3600)
                        minutes, seconds = divmod(remainder, 60)
                        print(f"  Duration: {int(hours)}h {int(minutes)}m {int(seconds)}s")

def get_summary_stats():
    """Get summary statistics"""
    app = create_app()
    
    with app.app_context():
        Video = app.Video
        DetectedPerson = app.DetectedPerson
        db = app.db
        
        # Get unique dates with attendance
        dates = db.session.query(
            DetectedPerson.attendance_date
        ).filter(
            DetectedPerson.attendance_date.isnot(None)
        ).distinct().count()
        
        # Get unique persons with attendance
        persons = db.session.query(
            DetectedPerson.person_id
        ).filter(
            DetectedPerson.attendance_date.isnot(None)
        ).distinct().count()
        
        # Get unique locations
        locations = db.session.query(
            DetectedPerson.attendance_location
        ).filter(
            DetectedPerson.attendance_location.isnot(None)
        ).distinct().all()
        
        print("\nATTENDANCE SUMMARY:")
        print(f"- Total Days with Records: {dates}")
        print(f"- Total Unique Persons: {persons}")
        print(f"- Locations: {', '.join([loc[0] for loc in locations if loc[0]])}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='View person attendance logs')
    parser.add_argument('--date', type=str, help='Filter by date (YYYY-MM-DD)')
    parser.add_argument('--location', type=str, help='Filter by location')
    parser.add_argument('--person', type=str, help='Filter by person ID')
    parser.add_argument('--summary', action='store_true', help='Show summary statistics')
    
    args = parser.parse_args()
    
    if args.summary:
        get_summary_stats()
    else:
        # Parse date if provided
        date_filter = None
        if args.date:
            try:
                date_filter = datetime.strptime(args.date, '%Y-%m-%d').date()
            except ValueError:
                print(f"Invalid date format: {args.date}. Use YYYY-MM-DD")
                return
        
        view_attendance_logs(
            date_filter=date_filter,
            location_filter=args.location,
            person_filter=args.person
        )

if __name__ == '__main__':
    main()