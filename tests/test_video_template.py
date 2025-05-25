#!/usr/bin/env python3
"""Test video detail template rendering"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
from flask import render_template_string
from jinja2 import Environment, FileSystemLoader

def test_template_syntax():
    """Test that the video detail template has valid syntax"""
    with app.app_context():
        try:
            # Create a mock video object with all required attributes
            mock_video = {
                'id': 1,
                'filename': 'test.mp4',
                'file_path': 'test.mp4',
                'original_filename': 'test.mp4',
                'file_size': 1024,
                'duration': 60.0,
                'fps': 30.0,
                'width': 1920,
                'height': 1080,
                'upload_date': '2025-01-01',
                'status': 'completed',
                'processed_path': None,
                'annotated_video_path': None,
                'processing_started_at': None,
                'processing_completed_at': None,
                'error_message': None,
                'detected_persons': [],
                'processing_log': None
            }
            
            # Test with empty detections
            detections = []
            view_mode = 'individual'
            total_detections = 0
            
            # Load and render template
            env = Environment(loader=FileSystemLoader('templates'))
            template = env.get_template('videos/detail.html')
            
            # This will raise an exception if there's a syntax error
            rendered = template.render(
                video=mock_video,
                detections=detections,
                view_mode=view_mode,
                total_detections=total_detections
            )
            
            print("✅ Template renders successfully with mock data!")
            return True
            
        except Exception as e:
            print(f"❌ Template rendering failed: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == '__main__':
    success = test_template_syntax()
    sys.exit(0 if success else 1)