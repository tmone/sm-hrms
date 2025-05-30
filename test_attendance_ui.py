#!/usr/bin/env python3
"""
Test script to check attendance UI accessibility
"""

import requests
from urllib.parse import urljoin

# Base URL of your application
BASE_URL = "http://localhost:5001"

def test_attendance_endpoints():
    """Test all attendance endpoints"""
    endpoints = [
        "/attendance/",
        "/attendance/summary?days=7",
        "/attendance/daily?format=json",
    ]
    
    print("Testing Attendance UI Endpoints")
    print("=" * 50)
    
    for endpoint in endpoints:
        url = urljoin(BASE_URL, endpoint)
        print(f"\nTesting: {url}")
        
        try:
            response = requests.get(url, timeout=5)
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                print("✓ Endpoint accessible")
                if endpoint.endswith("json"):
                    try:
                        data = response.json()
                        print(f"✓ Valid JSON response with {len(data)} keys")
                    except:
                        print("✗ Invalid JSON response")
            elif response.status_code == 302:
                print("→ Redirect (likely to login page)")
                print(f"  Location: {response.headers.get('Location', 'Unknown')}")
            else:
                print(f"✗ Error: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print("✗ Connection Error - Is the server running?")
        except requests.exceptions.Timeout:
            print("✗ Request Timeout")
        except Exception as e:
            print(f"✗ Error: {str(e)}")

if __name__ == "__main__":
    test_attendance_endpoints()