#!/usr/bin/env python3
"""
Fix video preview template URLs
"""

import re

template_path = "templates/videos/detail.html"

# Read template
with open(template_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix serve URLs
replacements = [
    # Fix annotated video serve URL
    (r"url_for\('videos\.serve_video_static', filename=video\.annotated_video_path\)", 
     "url_for('videos.serve_annotated', filename=video.annotated_video_path)"),
    
    # Fix detected video serve URL  
    (r"url_for\('videos\.serve_detected_video', filename=video\.annotated_video_path\)",
     "url_for('videos.serve_annotated', filename=video.annotated_video_path)"),
    
    # Fix processed video serve URL
    (r"url_for\('videos\.serve_video_static', filename=video\.processed_path\)",
     "url_for('videos.serve', filename=video.processed_path)"),
     
    # Fix original video serve URL
    (r"url_for\('videos\.serve_video_static', filename=video\.file_path\)",
     "url_for('videos.serve', filename=video.file_path)")
]

# Apply replacements
for old, new in replacements:
    content = re.sub(old, new, content)
    
# Write back
with open(template_path, 'w', encoding='utf-8') as f:
    f.write(content)
    
print("[OK] Fixed video preview template URLs")