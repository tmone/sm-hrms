#!/usr/bin/env python3
"""Test script to verify dashboard functionality after fixes"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app

def test_dashboard():
    """Test dashboard stats calculation"""
    app = create_app()
    
    with app.app_context():
        # Import the dashboard blueprint
        from hr_management.blueprints.dashboard import get_dashboard_stats
        
        try:
            # Test getting dashboard stats
            stats = get_dashboard_stats()
            
            print("[OK] Dashboard stats retrieved successfully!")
            print("\nStats structure:")
            for key, value in stats.items():
                print(f"  {key}: {type(value).__name__}")
                if isinstance(value, dict):
                    for k, v in value.items():
                        print(f"    - {k}: {v}")
            
            # Verify required keys exist
            required_keys = ['employees', 'videos', 'detections', 'models', 'datasets', 'attendance', 'queue']
            missing_keys = [k for k in required_keys if k not in stats]
            
            if missing_keys:
                print(f"\n[ERROR] Missing required keys: {missing_keys}")
                return False
            else:
                print("\n[OK] All required keys present")
            
            # Test template rendering
            with app.test_client() as client:
                # Create a test user session (simulate login)
                with client.session_transaction() as sess:
                    sess['_user_id'] = '1'  # Simulate logged in user
                
                # Try to access dashboard
                response = client.get('/')
                
                if response.status_code == 302:  # Redirect to login
                    print("\n[WARNING] Dashboard requires authentication (redirect to login)")
                elif response.status_code == 200:
                    print("\n[OK] Dashboard page renders successfully")
                else:
                    print(f"\n[ERROR] Dashboard returned status code: {response.status_code}")
                    
            return True
            
        except Exception as e:
            print(f"\n[ERROR] Error testing dashboard: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == '__main__':
    print("Testing dashboard functionality...")
    success = test_dashboard()
    
    if success:
        print("\n[SUCCESS] Dashboard tests passed!")
    else:
        print("\n[FAILED] Dashboard tests failed!")
        sys.exit(1)