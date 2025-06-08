#!/usr/bin/env python3
"""
Create test detection data with proper numeric values for testing jumpToDetection.
"""

import os
import sys
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_app():
    """Create Flask app for database access"""
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///instance/hr_management.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    db = SQLAlchemy()
    db.init_app(app)
    
    return app, db

def create_test_detections(video_id):
    """Create test detection data with proper numeric values"""
    app, db = create_app()
    
    with app.app_context():
        # Import the model
        from app import DetectedPerson
        
        print(f"🧪 Creating test detections for video {video_id}...")
        
        # Clear existing detections for this video
        existing = DetectedPerson.query.filter_by(video_id=video_id).all()
        for detection in existing:
            db.session.delete(detection)
        
        print(f"[DELETE] Cleared {len(existing)} existing detections")
        
        # Create test detections with proper numeric data
        test_detections = [
            {
                'timestamp': 1.5,
                'frame_number': 45,
                'confidence': 0.85,
                'bbox_x': 10,      # 10% from left
                'bbox_y': 15,      # 15% from top
                'bbox_width': 25,  # 25% width
                'bbox_height': 40  # 40% height
            },
            {
                'timestamp': 4.2,
                'frame_number': 126,
                'confidence': 0.92,
                'bbox_x': 60,      # 60% from left
                'bbox_y': 20,      # 20% from top
                'bbox_width': 30,  # 30% width
                'bbox_height': 35  # 35% height
            },
            {
                'timestamp': 7.8,
                'frame_number': 234,
                'confidence': 0.78,
                'bbox_x': 35,      # 35% from left
                'bbox_y': 25,      # 25% from top
                'bbox_width': 20,  # 20% width
                'bbox_height': 45  # 45% height
            },
            {
                'timestamp': 12.1,
                'frame_number': 363,
                'confidence': 0.89,
                'bbox_x': 70,      # 70% from left
                'bbox_y': 10,      # 10% from top
                'bbox_width': 25,  # 25% width
                'bbox_height': 50  # 50% height
            },
            {
                'timestamp': 18.5,
                'frame_number': 555,
                'confidence': 0.83,
                'bbox_x': 15,      # 15% from left
                'bbox_y': 30,      # 30% from top
                'bbox_width': 35,  # 35% width
                'bbox_height': 40  # 40% height
            }
        ]
        
        created_count = 0
        for i, detection_data in enumerate(test_detections):
            detection = DetectedPerson(
                video_id=video_id,
                timestamp=detection_data['timestamp'],
                frame_number=detection_data['frame_number'],
                confidence=detection_data['confidence'],
                bbox_x=detection_data['bbox_x'],
                bbox_y=detection_data['bbox_y'],
                bbox_width=detection_data['bbox_width'],
                bbox_height=detection_data['bbox_height'],
                is_identified=False
            )
            
            db.session.add(detection)
            created_count += 1
            
            print(f"   [OK] Created detection {i+1}: {detection_data['timestamp']}s at [{detection_data['bbox_x']}, {detection_data['bbox_y']}, {detection_data['bbox_width']}x{detection_data['bbox_height']}]")
        
        try:
            db.session.commit()
            print(f"[OK] Successfully created {created_count} test detections for video {video_id}")
            return True
        except Exception as e:
            print(f"[ERROR] Error creating test detections: {e}")
            db.session.rollback()
            return False

def list_videos():
    """List available videos"""
    app, db = create_app()
    
    with app.app_context():
        from app import Video
        
        videos = Video.query.all()
        print("[VIDEO] Available videos:")
        for video in videos:
            print(f"   {video.id}: {video.title} ({video.status})")
        
        return videos

if __name__ == "__main__":
    print("🧪 Test Detection Data Creator")
    print("=" * 40)
    
    # List available videos
    videos = list_videos()
    
    if not videos:
        print("[ERROR] No videos found in database!")
        sys.exit(1)
    
    # Get video ID from user or use first video
    if len(sys.argv) > 1:
        try:
            video_id = int(sys.argv[1])
        except ValueError:
            print("[ERROR] Invalid video ID. Please provide a numeric video ID.")
            sys.exit(1)
    else:
        video_id = videos[0].id
        print(f"[TARGET] Using first video (ID: {video_id})")
    
    # Verify video exists
    video_exists = any(v.id == video_id for v in videos)
    if not video_exists:
        print(f"[ERROR] Video with ID {video_id} not found!")
        sys.exit(1)
    
    # Create test detections
    success = create_test_detections(video_id)
    
    if success:
        print(f"\n[OK] Test detections created successfully for video {video_id}!")
        print("Now you can test the jumpToDetection function in the web interface.")
    else:
        print(f"\n[ERROR] Failed to create test detections!")
        sys.exit(1)