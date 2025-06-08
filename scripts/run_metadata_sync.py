#!/usr/bin/env python
"""
Script to run metadata synchronization
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from hr_management.models.video import Video
from hr_management.models.face_recognition import DetectedPerson
from hr_management.blueprints.persons import sync_metadata_with_database

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///instance/hr_management.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    db.init_app(app)
    
    # Make models available via current_app
    app.Video = Video
    app.DetectedPerson = DetectedPerson
    app.db = db
    
    return app

def main():
    print("Starting metadata synchronization...")
    
    app = create_app()
    
    with app.app_context():
        try:
            sync_metadata_with_database()
            print("\n[OK] Metadata synchronization completed successfully!")
        except Exception as e:
            print(f"\n[ERROR] Error during synchronization: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()