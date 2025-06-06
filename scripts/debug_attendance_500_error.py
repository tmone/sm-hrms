#!/usr/bin/env python3
"""
Debug the 500 error in attendance list endpoint
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from datetime import date, datetime, timedelta
from sqlalchemy import func, and_
import traceback

def debug_attendance_500():
    """Debug the 500 error"""
    app = create_app()
    
    with app.app_context():
        db = app.db
        Video = app.Video
        DetectedPerson = app.DetectedPerson
        
        print("="*80)
        print("DEBUGGING ATTENDANCE LIST 500 ERROR")
        print("="*80)
        
        try:
            # Simulate the exact query from the endpoint
            page = 1
            per_page = 20
            filter_type = 'all'
            location = ''
            person_id_filter = ''
            sort = 'desc'
            
            # Build date filter
            today = date.today()
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
            
            # Apply sorting
            if sort == 'desc':
                base_query = base_query.order_by(
                    DetectedPerson.attendance_date.desc(),
                    DetectedPerson.attendance_location,
                    DetectedPerson.person_id
                )
            
            print("1. Testing pagination...")
            # Test pagination
            try:
                pagination = base_query.paginate(page=page, per_page=per_page, error_out=False)
                print(f"   ✅ Pagination successful: {pagination.total} total records")
            except Exception as e:
                print(f"   ❌ Pagination error: {e}")
                traceback.print_exc()
                return
            
            print("\n2. Testing record processing...")
            # Format results
            attendance_records = []
            for i, record in enumerate(pagination.items):
                try:
                    print(f"\n   Processing record {i+1}:")
                    print(f"   - Person ID: {record.person_id}")
                    print(f"   - Video ID: {record.video_id}")
                    
                    # Get video info - THIS MIGHT BE THE ISSUE
                    video = Video.query.get(record.video_id)
                    if not video:
                        print(f"   ⚠️  WARNING: Video {record.video_id} not found!")
                    else:
                        print(f"   - Video filename: {video.filename}")
                    
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
                    
                    # Test the format_duration function inline
                    def format_duration(seconds):
                        if not seconds:
                            return "0m"
                        hours = int(seconds // 3600)
                        minutes = int((seconds % 3600) // 60)
                        if hours > 0:
                            return f"{hours}h {minutes}m"
                        return f"{minutes}m"
                    
                    attendance_record = {
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
                    }
                    
                    attendance_records.append(attendance_record)
                    print(f"   ✅ Record processed successfully")
                    
                except Exception as e:
                    print(f"   ❌ Error processing record: {e}")
                    traceback.print_exc()
            
            print(f"\n3. Successfully processed {len(attendance_records)} records")
            
            # Test locations query
            print("\n4. Testing locations query...")
            try:
                all_locations = db.session.query(
                    func.distinct(DetectedPerson.attendance_location)
                ).filter(
                    DetectedPerson.attendance_location.isnot(None)
                ).all()
                locations = [loc[0] for loc in all_locations]
                print(f"   ✅ Found locations: {locations}")
            except Exception as e:
                print(f"   ❌ Locations query error: {e}")
                traceback.print_exc()
                
        except Exception as e:
            print(f"\n❌ MAIN ERROR: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    debug_attendance_500()