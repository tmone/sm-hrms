#!/usr/bin/env python3
"""
Test the attendance list API endpoint
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from flask import json

def test_attendance_list_api():
    """Test the /attendance/list endpoint"""
    app = create_app()
    
    with app.app_context():
        with app.test_client() as client:
            # First, we need to login (if required)
            # For testing, we'll try without login first
            
            print("="*80)
            print("TESTING ATTENDANCE LIST API")
            print("="*80)
            
            # Test 1: Basic request without filters
            print("\n1. Testing basic request (no filters):")
            response = client.get('/attendance/list?page=1&per_page=20&sort=desc&filter=all')
            print(f"   Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = json.loads(response.data)
                print(f"   Total Records: {data['pagination']['total']}")
                print(f"   Records on Page: {len(data['records'])}")
                if data['records']:
                    print(f"   First Record: {data['records'][0]}")
            else:
                print(f"   Error: {response.data.decode()}")
            
            # Test 2: Check if login is required
            if response.status_code == 401 or response.status_code == 302:
                print("\n   ⚠️  Login required. Attempting to login...")
                
                # Try to login with default credentials
                login_data = {
                    'username': 'admin',
                    'password': 'admin123'
                }
                login_response = client.post('/auth/login', 
                                           data=login_data,
                                           follow_redirects=False)
                
                if login_response.status_code in [302, 200]:
                    print("   ✅ Login successful")
                    
                    # Retry the API request
                    print("\n2. Retrying API request after login:")
                    response = client.get('/attendance/list?page=1&per_page=20&sort=desc&filter=all')
                    print(f"   Status Code: {response.status_code}")
                    
                    if response.status_code == 200:
                        data = json.loads(response.data)
                        print(f"   Total Records: {data['pagination']['total']}")
                        print(f"   Records on Page: {len(data['records'])}")
                        print(f"   Available Locations: {data['filters']['locations']}")
                        
                        if data['records']:
                            print("\n   Sample Records:")
                            for i, record in enumerate(data['records'][:3]):
                                print(f"   {i+1}. Person: {record['person_id']}, Date: {record['date']}, Location: {record['location']}")
                    else:
                        print(f"   Error: {response.data.decode()}")
                else:
                    print(f"   ❌ Login failed: {login_response.status_code}")

if __name__ == "__main__":
    test_attendance_list_api()