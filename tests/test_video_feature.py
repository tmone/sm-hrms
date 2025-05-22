#!/usr/bin/env python3
"""
Test script to verify video management feature is working
"""
import requests
import sys

def test_video_management():
    """Test if video management is available"""
    base_url = "http://localhost:5000"
    
    print("Testing Video Management Feature...")
    print("=" * 50)
    
    # First, get demo login session
    session = requests.Session()
    
    try:
        # Try demo login
        login_response = session.post(f"{base_url}/auth/demo-login", allow_redirects=False)
        print(f"Demo login: {login_response.status_code}")
        
        if login_response.status_code != 302:
            print("❌ Failed to login")
            return False
        
        # Test video management page
        video_response = session.get(f"{base_url}/videos", allow_redirects=False)
        print(f"Video management page: {video_response.status_code}")
        
        if video_response.status_code == 200:
            print("✅ Video management is ENABLED and working!")
            
            # Check if it contains upload functionality
            if "Upload Video" in video_response.text or "upload" in video_response.text.lower():
                print("✅ Video upload functionality detected")
                
            # Test upload page
            upload_response = session.get(f"{base_url}/videos/upload")
            if upload_response.status_code == 200:
                print("✅ Video upload page accessible")
            else:
                print(f"⚠️ Video upload page returned: {upload_response.status_code}")
                
            return True
            
        elif video_response.status_code == 302:
            print("⚠️ Video management redirected (might be disabled)")
            return False
        else:
            # Check if we get the "not available" page
            if "not available" in video_response.text.lower():
                print("❌ Video management is DISABLED")
                print("    The application is showing 'Video Management Not Available' page")
                return False
            else:
                print(f"❌ Unexpected response: {video_response.status_code}")
                return False
                
    except Exception as e:
        print(f"❌ Error testing video management: {e}")
        return False

def test_face_recognition():
    """Test if face recognition is available"""
    base_url = "http://localhost:5000"
    
    print("\nTesting Face Recognition Feature...")
    print("=" * 50)
    
    session = requests.Session()
    
    try:
        # Demo login
        session.post(f"{base_url}/auth/demo-login", allow_redirects=False)
        
        # Test face recognition page
        face_response = session.get(f"{base_url}/face-recognition/datasets")
        print(f"Face recognition datasets page: {face_response.status_code}")
        
        if face_response.status_code == 200:
            if "not available" not in face_response.text.lower():
                print("✅ Face recognition is ENABLED and working!")
                return True
            else:
                print("❌ Face recognition is DISABLED")
                return False
        else:
            print(f"❌ Face recognition returned: {face_response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing face recognition: {e}")
        return False

def main():
    print("StepMedia HRM Feature Test")
    print("=" * 50)
    
    video_working = test_video_management()
    face_working = test_face_recognition()
    
    print("\n" + "=" * 50)
    print("RESULTS:")
    print(f"✅ Video Management: {'ENABLED' if video_working else 'DISABLED'}")
    print(f"✅ Face Recognition: {'ENABLED' if face_working else 'DISABLED'}")
    
    if video_working and face_working:
        print("\n🎉 All advanced features are working!")
        return 0
    elif video_working or face_working:
        print("\n⚠️ Some features are working")
        return 0
    else:
        print("\n❌ Advanced features are disabled")
        print("\nTo enable them:")
        print("1. Make sure the application models are loaded correctly")
        print("2. Restart the application")
        print("3. Check for any error messages in the console")
        return 1

if __name__ == "__main__":
    sys.exit(main())