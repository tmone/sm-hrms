#!/usr/bin/env python3
"""
Test script to verify video serving functionality
"""
import os
import sys

def test_video_file_access():
    """Test if video file can be accessed directly"""
    video_file = "static/uploads/4a5b80c6-1959-4032-8ad1-f375408b1f43_TANG_TRET_84A_Tret_84A_Tret_20250512085459_20250512091458_472401.mp4"
    
    print("=== Video File Access Test ===")
    
    if os.path.exists(video_file):
        file_size = os.path.getsize(video_file)
        print(f"✅ Video file exists: {video_file}")
        print(f"📁 File size: {file_size:,} bytes ({file_size / 1024 / 1024:.1f} MB)")
        
        # Test reading first few bytes
        try:
            with open(video_file, 'rb') as f:
                header = f.read(32)
                print(f"📄 File header: {header.hex()}")
                print(f"📄 File readable: ✅")
        except Exception as e:
            print(f"❌ Error reading file: {e}")
    else:
        print(f"❌ Video file not found: {video_file}")
    
    print()

def test_flask_imports():
    """Test if Flask can be imported and app created"""
    print("=== Flask Import Test ===")
    
    try:
        from flask import Flask
        print("✅ Flask imported successfully")
        
        # Try to create basic app
        app = Flask(__name__)
        print("✅ Flask app created successfully")
        
        # Test if our app module works
        sys.path.append('.')
        from app import create_app
        hrm_app = create_app()
        print("✅ HRM app created successfully")
        
        # Test app configuration
        print(f"📁 Upload folder: {hrm_app.config.get('UPLOAD_FOLDER')}")
        print(f"🔧 Debug mode: {hrm_app.config.get('DEBUG')}")
        
    except ImportError as e:
        print(f"❌ Flask import error: {e}")
        print("💡 Solution: pip install flask flask-sqlalchemy flask-login")
    except Exception as e:
        print(f"❌ App creation error: {e}")
    
    print()

def test_url_generation():
    """Test URL generation for video endpoints"""
    print("=== URL Generation Test ===")
    
    try:
        sys.path.append('.')
        from app import create_app
        
        app = create_app()
        with app.app_context():
            from flask import url_for
            
            filename = "4a5b80c6-1959-4032-8ad1-f375408b1f43_TANG_TRET_84A_Tret_84A_Tret_20250512085459_20250512091458_472401.mp4"
            
            # Test different URL generation methods
            try:
                stream_url = url_for('videos.stream_video', filename=filename)
                print(f"🔗 Stream URL: {stream_url}")
            except Exception as e:
                print(f"❌ Stream URL error: {e}")
            
            try:
                serve_url = url_for('videos.serve_video_static', filename=filename)
                print(f"🔗 Serve URL: {serve_url}")
            except Exception as e:
                print(f"❌ Serve URL error: {e}")
            
            try:
                download_url = url_for('videos.download_video', filename=filename)
                print(f"🔗 Download URL: {download_url}")
            except Exception as e:
                print(f"❌ Download URL error: {e}")
            
            try:
                static_url = url_for('static', filename=f'uploads/{filename}')
                print(f"🔗 Static URL: {static_url}")
            except Exception as e:
                print(f"❌ Static URL error: {e}")
                
    except Exception as e:
        print(f"❌ URL generation failed: {e}")
    
    print()

def create_test_html():
    """Create a test HTML file to debug video playback"""
    filename = "4a5b80c6-1959-4032-8ad1-f375408b1f43_TANG_TRET_84A_Tret_84A_Tret_20250512085459_20250512091458_472401.mp4"
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Video Test</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .test-section {{ margin: 20px 0; padding: 15px; border: 1px solid #ccc; }}
        video {{ width: 100%; max-width: 600px; }}
        .error {{ color: red; }}
        .success {{ color: green; }}
    </style>
</head>
<body>
    <h1>Video Playback Test</h1>
    
    <div class="test-section">
        <h2>Test 1: Direct File Access</h2>
        <video controls>
            <source src="/static/uploads/{filename}" type="video/mp4">
            <source src="/static/uploads/{filename}" type="application/octet-stream">
            Your browser does not support the video tag.
        </video>
        <p>URL: /static/uploads/{filename}</p>
    </div>
    
    <div class="test-section">
        <h2>Test 2: Stream Endpoint</h2>
        <video controls>
            <source src="/videos/stream/{filename}" type="video/mp4">
            <source src="/videos/stream/{filename}" type="application/octet-stream">
            Your browser does not support the video tag.
        </video>
        <p>URL: /videos/stream/{filename}</p>
    </div>
    
    <div class="test-section">
        <h2>Test 3: Serve Endpoint</h2>
        <video controls>
            <source src="/videos/serve/{filename}" type="video/mp4">
            <source src="/videos/serve/{filename}" type="application/octet-stream">
            Your browser does not support the video tag.
        </video>
        <p>URL: /videos/serve/{filename}</p>
    </div>
    
    <div class="test-section">
        <h2>Download Links</h2>
        <ul>
            <li><a href="/videos/download/{filename}" target="_blank">Download via endpoint</a></li>
            <li><a href="/static/uploads/{filename}" target="_blank">Direct file access</a></li>
        </ul>
    </div>
    
    <script>
        // Add error handling to all videos
        document.addEventListener('DOMContentLoaded', function() {{
            const videos = document.querySelectorAll('video');
            videos.forEach((video, index) => {{
                video.addEventListener('error', function(e) {{
                    console.error(`Video ${{index + 1}} error:`, e);
                    const error = document.createElement('p');
                    error.className = 'error';
                    error.textContent = `❌ Video failed to load (Error: ${{e.target.error?.code || 'Unknown'}})`;
                    video.parentNode.appendChild(error);
                }});
                
                video.addEventListener('loadstart', function() {{
                    console.log(`Video ${{index + 1}} load started`);
                }});
                
                video.addEventListener('canplay', function() {{
                    console.log(`Video ${{index + 1}} can play`);
                    const success = document.createElement('p');
                    success.className = 'success';
                    success.textContent = '✅ Video loaded successfully';
                    video.parentNode.appendChild(success);
                }});
            }});
        }});
    </script>
</body>
</html>"""
    
    with open('video_test.html', 'w') as f:
        f.write(html_content)
    
    print("📄 Created video_test.html for browser testing")
    print("💡 Copy this file to your Flask static folder and access via browser")

if __name__ == "__main__":
    print("🔍 Video Serving Diagnostic Tool")
    print("=" * 50)
    
    test_video_file_access()
    test_flask_imports()
    test_url_generation()
    create_test_html()
    
    print("🎯 Next Steps:")
    print("1. Start your Flask app: python app.py")
    print("2. Open browser to http://localhost:5000/static/video_test.html")
    print("3. Check browser console for detailed error messages")
    print("4. Test each video source and note which ones work")