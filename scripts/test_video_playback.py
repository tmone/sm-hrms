#!/usr/bin/env python3
"""
Test video playback by creating a simple HTML file
"""

import os

video_filename = "6b0e2a6e-8ec4-406b-b604-bf7ef7470596_test.mp4"
video_path = f"static/uploads/{video_filename}"

# Check video properties
if os.path.exists(video_path):
    print(f"✅ Video exists: {video_path}")
    print(f"Size: {os.path.getsize(video_path) / (1024*1024):.1f} MB")
else:
    print(f"❌ Video not found: {video_path}")
    exit(1)

# Create test HTML
html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Video Test</title>
    <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; }}
        video {{ max-width: 100%; height: auto; }}
        .info {{ margin: 20px 0; padding: 10px; background: #f0f0f0; }}
    </style>
</head>
<body>
    <h1>Video Playback Test</h1>
    
    <h2>Test 1: Direct file path</h2>
    <video controls width="640" height="360">
        <source src="/{video_path}" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    
    <h2>Test 2: Flask serve route</h2>
    <video controls width="640" height="360">
        <source src="/videos/serve/{video_filename}" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    
    <h2>Test 3: Stream route</h2>
    <video controls width="640" height="360">
        <source src="/videos/stream/{video_filename}" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    
    <div class="info">
        <h3>Debug Info:</h3>
        <p>Video file: {video_filename}</p>
        <p>File size: {os.path.getsize(video_path) / (1024*1024):.1f} MB</p>
        <p>Direct path: /{video_path}</p>
        <p>Serve URL: /videos/serve/{video_filename}</p>
        <p>Stream URL: /videos/stream/{video_filename}</p>
    </div>
    
    <script>
        // Log video events
        document.querySelectorAll('video').forEach((video, index) => {{
            video.addEventListener('loadstart', () => console.log(`Video ${{index+1}}: loadstart`));
            video.addEventListener('loadedmetadata', () => console.log(`Video ${{index+1}}: metadata loaded`));
            video.addEventListener('canplay', () => console.log(`Video ${{index+1}}: can play`));
            video.addEventListener('error', (e) => {{
                console.error(`Video ${{index+1}} error:`, e);
                console.error('Error code:', video.error?.code);
                console.error('Error message:', video.error?.message);
            }});
        }});
    </script>
</body>
</html>"""

# Save test HTML
with open("test_video_playback.html", "w") as f:
    f.write(html_content)
    
print("✅ Created test_video_playback.html")
print("📌 Open this file in your browser to test video playback")
print("📌 Check browser console (F12) for error messages")