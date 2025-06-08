#!/usr/bin/env python3
"""
Debug script to check if models are properly loaded
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app

def debug_models():
    print("Debugging model availability...")
    print("=" * 50)
    
    app = create_app()
    with app.app_context():
        print("Current app:", app)
        print("App attributes:", [attr for attr in dir(app) if not attr.startswith('_')])
        
        # Check for models
        models_to_check = ['Employee', 'AttendanceRecord', 'Video', 'DetectedPerson', 'FaceDataset', 'TrainedModel', 'RecognitionResult']
        
        for model_name in models_to_check:
            model = getattr(app, model_name, None)
            if model:
                print(f"[OK] {model_name}: Available ({model})")
                try:
                    count = model.query.count()
                    print(f"   Records in database: {count}")
                except Exception as e:
                    print(f"   Error querying: {e}")
            else:
                print(f"[ERROR] {model_name}: Not Available")
        
        # Check db
        db = getattr(app, 'db', None)
        if db:
            print(f"OK Database: Available ({db})")
        else:
            print("ERROR Database: Not Available")

if __name__ == "__main__":
    debug_models()