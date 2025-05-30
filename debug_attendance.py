#!/usr/bin/env python3
"""
Debug attendance functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app import create_app
    print("✓ App module imported successfully")
    
    app = create_app()
    print("✓ App created successfully")
    
    with app.app_context():
        # Check if models are available
        try:
            Video = app.Video
            DetectedPerson = app.DetectedPerson
            print("✓ Models loaded successfully")
            
            # Check for videos with OCR data
            ocr_videos = Video.query.filter(Video.ocr_extraction_done == True).count()
            print(f"✓ Videos with OCR data: {ocr_videos}")
            
            # Check for attendance records
            attendance_records = DetectedPerson.query.filter(
                DetectedPerson.attendance_date.isnot(None)
            ).count()
            print(f"✓ Attendance records: {attendance_records}")
            
        except Exception as e:
            print(f"✗ Error accessing models: {e}")
            
        # Check if attendance blueprint is registered
        if 'attendance' in [bp.name for bp in app.blueprints.values()]:
            print("✓ Attendance blueprint is registered")
            
            # List all attendance routes
            print("\nAttendance Routes:")
            for rule in app.url_map.iter_rules():
                if 'attendance' in rule.rule:
                    print(f"  - {rule.rule} [{', '.join(rule.methods - {'HEAD', 'OPTIONS'})}]")
        else:
            print("✗ Attendance blueprint not registered!")
            
except ImportError as e:
    print(f"✗ Failed to import app: {e}")
    print("\nPlease ensure all dependencies are installed:")
    print("  pip install -r requirements.txt")
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    import traceback
    traceback.print_exc()