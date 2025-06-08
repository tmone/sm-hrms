#!/usr/bin/env python3
"""Check unique person IDs in database."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from hr_management.models.face_recognition import DetectedPerson
from hr_management.models.video import Video

db = SQLAlchemy()

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///instance/hr_management.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

with app.app_context():
    # Get unique person IDs
    unique_persons = db.session.query(DetectedPerson.person_id).distinct().all()
    print(f"Unique person IDs in database: {len(unique_persons)}")
    
    for person_id, in unique_persons:
        count = DetectedPerson.query.filter_by(person_id=person_id).count()
        print(f"  Person ID {person_id}: {count} detections")
    
    # Check videos
    videos = Video.query.all()
    print(f"\nVideos in database:")
    for video in videos:
        print(f"  Video {video.id}: {video.filename} - {video.person_count} persons")