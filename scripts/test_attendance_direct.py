#!/usr/bin/env python3
"""
Test attendance list function directly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from flask import Flask
from werkzeug.test import EnvironBuilder
from flask.ctx import RequestContext

def test_attendance_direct():
    """Test the attendance list function directly"""
    app = create_app()
    
    with app.app_context():
        print("="*80)
        print("TESTING ATTENDANCE LIST DIRECTLY")
        print("="*80)
        
        # Import the blueprint function
        from hr_management.blueprints.attendance import attendance_list
        
        # Create a test request context
        with app.test_request_context('/attendance/list?page=1&per_page=20&sort=desc&filter=all'):
            # Skip login for now - just test the function
            print("[WARNING]  Testing without login (may fail if login_required)")
            
            try:
                # Call the function directly
                result = attendance_list()
                
                # Check the response
                print(f"Response type: {type(result)}")
                print(f"Response status: {result.status if hasattr(result, 'status') else 'N/A'}")
                
                # Try to get JSON data
                if hasattr(result, 'get_json'):
                    data = result.get_json()
                elif hasattr(result, 'json'):
                    data = result.json
                else:
                    # Try to parse the data
                    import json
                    data = json.loads(result.data) if hasattr(result, 'data') else None
                
                if data:
                    print(f"\n[OK] SUCCESS! Got JSON response")
                    print(f"Total records: {data['pagination']['total']}")
                    print(f"Records on this page: {len(data['records'])}")
                    print(f"Available locations: {data['filters']['locations']}")
                    
                    if data['records']:
                        print("\nSample records:")
                        for i, record in enumerate(data['records'][:3]):
                            print(f"{i+1}. Person: {record['person_id']}, Date: {record['date']}, Location: {record['location']}")
                            print(f"   Check In: {record['check_in']}, Check Out: {record['check_out']}, Duration: {record['duration_formatted']}")
                else:
                    print(f"[ERROR] Could not extract JSON data")
                    print(f"Response: {result}")
                    if hasattr(result, 'data'):
                        print(f"Response data: {result.data}")
                    
            except Exception as e:
                print(f"\n[ERROR] ERROR: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    test_attendance_direct()