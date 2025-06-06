#!/usr/bin/env python3
"""
Test the attendance aggregation query
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from datetime import date
from sqlalchemy import func, and_

def test_attendance_aggregation():
    """Test the attendance aggregation query"""
    app = create_app()
    
    with app.app_context():
        db = app.db
        DetectedPerson = app.DetectedPerson
        
        print("="*80)
        print("TESTING ATTENDANCE AGGREGATION QUERY")
        print("="*80)
        
        # First, check raw data
        print("\n1. Checking raw DetectedPerson data:")
        total_records = DetectedPerson.query.count()
        print(f"   Total DetectedPerson records: {total_records}")
        
        # Sample records
        sample_records = DetectedPerson.query.limit(5).all()
        print("\n   Sample records:")
        for record in sample_records:
            print(f"   - ID: {record.id}, Person: {record.person_id}, Date: {record.attendance_date}, Time: {record.attendance_time}, Location: {record.attendance_location}")
        
        # Test the aggregation query used in attendance list
        print("\n2. Testing aggregation query:")
        
        # Build the same query as in the attendance list endpoint
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
            DetectedPerson.attendance_date.isnot(None)
        ).group_by(
            DetectedPerson.person_id,
            DetectedPerson.attendance_date,
            DetectedPerson.attendance_location,
            DetectedPerson.video_id
        ).order_by(
            DetectedPerson.attendance_date.desc(),
            DetectedPerson.attendance_location,
            DetectedPerson.person_id
        )
        
        # Execute query
        try:
            results = base_query.limit(10).all()
            print(f"   Aggregated records found: {len(results)}")
            
            if results:
                print("\n   Aggregated data:")
                for record in results[:5]:
                    print(f"   - Person: {record.person_id}, Date: {record.attendance_date}, Location: {record.attendance_location}")
                    print(f"     Check In: {record.check_in}, Check Out: {record.check_out}, Count: {record.detection_count}")
            else:
                print("   ⚠️  No aggregated records found!")
                
            # Try to count total aggregated records
            total_aggregated = base_query.count()
            print(f"\n   Total aggregated records: {total_aggregated}")
            
        except Exception as e:
            print(f"   ❌ Error in aggregation query: {e}")
            import traceback
            traceback.print_exc()
        
        # Test date filtering
        print("\n3. Testing date filters:")
        
        # Check specific date
        specific_date = date(2025, 5, 12)
        date_filtered = base_query.filter(
            DetectedPerson.attendance_date == specific_date
        ).all()
        print(f"   Records for {specific_date}: {len(date_filtered)}")

if __name__ == "__main__":
    test_attendance_aggregation()