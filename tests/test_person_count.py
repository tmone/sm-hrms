#!/usr/bin/env python3
"""Test script to verify person counting is consistent"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app

def test_person_count():
    """Test person counting in dashboard vs persons page"""
    app = create_app()
    
    with app.app_context():
        # Import required models
        db = app.db
        DetectedPerson = app.DetectedPerson
        
        # Dashboard method - count unique person_id from detected_persons
        dashboard_count = db.session.query(DetectedPerson.person_id)\
            .filter(DetectedPerson.person_id.isnot(None))\
            .distinct()\
            .count()
        
        print(f"Dashboard count (unique person_id in detected_persons): {dashboard_count}")
        
        # Get the actual person IDs to debug
        person_ids = db.session.query(DetectedPerson.person_id)\
            .filter(DetectedPerson.person_id.isnot(None))\
            .distinct()\
            .all()
        
        print("\nPerson IDs in database:")
        for pid in person_ids:
            print(f"  - {pid[0]}")
            
        # Persons page method - same query but formatted
        unique_person_ids = db.session.query(DetectedPerson.person_id)\
            .filter(DetectedPerson.person_id.isnot(None))\
            .distinct()\
            .all()
        unique_person_ids_formatted = {f"PERSON-{int(pid[0]):04d}" for pid in unique_person_ids if pid[0] is not None}
        
        print(f"\nPersons page count (after formatting): {len(unique_person_ids_formatted)}")
        print("Formatted person IDs:")
        for pid in sorted(unique_person_ids_formatted):
            print(f"  - {pid}")
            
        # Check filesystem
        from pathlib import Path
        persons_dir = Path('processing/outputs/persons')
        filesystem_count = 0
        if persons_dir.exists():
            filesystem_persons = [d.name for d in persons_dir.iterdir() if d.is_dir() and d.name.startswith('PERSON-')]
            filesystem_count = len(filesystem_persons)
            print(f"\nFilesystem person directories: {filesystem_count}")
            for person in sorted(filesystem_persons):
                print(f"  - {person}")
                
        # Summary
        print("\n" + "="*50)
        print("SUMMARY:")
        print(f"  Dashboard shows: {dashboard_count} unique persons")
        print(f"  Persons page should show: {len(unique_person_ids_formatted)} persons")
        print(f"  Filesystem has: {filesystem_count} person directories")
        
        if dashboard_count == len(unique_person_ids_formatted):
            print("\n[SUCCESS] Counts match!")
        else:
            print("\n[ERROR] Counts don't match!")

if __name__ == '__main__':
    test_person_count()