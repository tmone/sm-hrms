#!/usr/bin/env python3
"""
Quick test script to verify all routes are working properly
"""
import requests
import sys

def test_route(url, expected_status=200, description=""):
    """Test a single route"""
    try:
        response = requests.get(url, allow_redirects=False, timeout=5)
        status = "[OK]" if response.status_code == expected_status else "[ERROR]"
        print(f"{status} {url} - {response.status_code} ({description})")
        return response.status_code == expected_status
    except Exception as e:
        print(f"[ERROR] {url} - ERROR: {e}")
        return False

def main():
    base_url = "http://localhost:5000"
    
    print("Testing StepMedia HRM Routes...")
    print("=" * 50)
    
    # Test public routes (should redirect to login)
    routes_to_test = [
        (f"{base_url}/", 302, "Home - should redirect to login"),
        (f"{base_url}/auth/login", 200, "Login page"),
        (f"{base_url}/auth/demo-login", 302, "Demo login - should redirect"),
        
        # These should redirect to login for unauthenticated users
        (f"{base_url}/dashboard", 302, "Dashboard - should redirect to login"),
        (f"{base_url}/employees", 302, "Employees - should redirect to login"),
        (f"{base_url}/videos", 302, "Videos - should redirect to login"),
        (f"{base_url}/face-recognition", 302, "Face Recognition - should redirect to login"),
        
        # API endpoints should also redirect or return 401
        (f"{base_url}/api/stats", 302, "API Stats - should redirect to login"),
        (f"{base_url}/api/employees", 302, "API Employees - should redirect to login"),
    ]
    
    passed = 0
    total = len(routes_to_test)
    
    for url, expected_status, description in routes_to_test:
        if test_route(url, expected_status, description):
            passed += 1
    
    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All routes are working correctly!")
        return 0
    else:
        print("[WARNING] Some routes have issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())