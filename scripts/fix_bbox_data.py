#!/usr/bin/env python3
"""
Fix corrupted bounding box data in the database.
This script cleans up binary/bytes data that was incorrectly stored in bbox fields.
"""

import os
import sys
import struct
import numpy as np
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

def fix_bbox_data():
    """Fix corrupted bounding box data in detected_persons table"""
    app, db = create_app()
    
    with app.app_context():
        # Import the model
        from app import DetectedPerson
        
        print("🔍 Scanning for corrupted bounding box data...")
        
        # Get all detections
        detections = DetectedPerson.query.all()
        print(f"📊 Found {len(detections)} detection records")
        
        fixed_count = 0
        deleted_count = 0
        
        for detection in detections:
            needs_fix = False
            original_data = {
                'bbox_x': detection.bbox_x,
                'bbox_y': detection.bbox_y,
                'bbox_width': detection.bbox_width,
                'bbox_height': detection.bbox_height
            }
            
            # Check if any bbox field contains binary data
            for field_name, value in original_data.items():
                if isinstance(value, (bytes, bytearray)):
                    print(f"❌ Found binary data in {field_name} for detection {detection.id}: {value}")
                    needs_fix = True
                elif isinstance(value, str) and value.startswith("b'"):
                    print(f"❌ Found string-encoded binary data in {field_name} for detection {detection.id}: {value}")
                    needs_fix = True
                elif value is None:
                    print(f"⚠️ Found NULL value in {field_name} for detection {detection.id}")
                    needs_fix = True
            
            if needs_fix:
                try:
                    # Attempt to fix the data
                    fixed_bbox = fix_single_detection(detection, original_data)
                    
                    if fixed_bbox:
                        detection.bbox_x = fixed_bbox['x']
                        detection.bbox_y = fixed_bbox['y']
                        detection.bbox_width = fixed_bbox['width']
                        detection.bbox_height = fixed_bbox['height']
                        
                        print(f"✅ Fixed detection {detection.id}: {fixed_bbox}")
                        fixed_count += 1
                    else:
                        # Cannot fix, delete the record
                        print(f"🗑️ Deleting unfixable detection {detection.id}")
                        db.session.delete(detection)
                        deleted_count += 1
                        
                except Exception as e:
                    print(f"❌ Error fixing detection {detection.id}: {e}")
                    # Delete problematic records
                    db.session.delete(detection)
                    deleted_count += 1
        
        # Commit changes
        try:
            db.session.commit()
            print(f"✅ Database cleanup completed:")
            print(f"   🔧 Fixed: {fixed_count} records")
            print(f"   🗑️ Deleted: {deleted_count} records")
            print(f"   📊 Total processed: {len(detections)} records")
        except Exception as e:
            print(f"❌ Error committing changes: {e}")
            db.session.rollback()
            return False
    
    return True

def fix_single_detection(detection, original_data):
    """Attempt to fix a single detection's bounding box data"""
    
    # Try to decode binary data
    for field_name, value in original_data.items():
        if isinstance(value, (bytes, bytearray)):
            try:
                # Try to unpack as float32
                if len(value) == 4:
                    fixed_value = struct.unpack('f', value)[0]
                    if 0 <= fixed_value <= 100:  # Valid percentage
                        original_data[field_name] = int(round(fixed_value))
                        continue
                
                # Try to unpack as int32
                if len(value) == 4:
                    fixed_value = struct.unpack('i', value)[0]
                    if 0 <= fixed_value <= 100:  # Valid percentage
                        original_data[field_name] = fixed_value
                        continue
                        
            except (struct.error, ValueError):
                pass
        
        elif isinstance(value, str) and value.startswith("b'"):
            # Try to parse string-encoded binary
            try:
                # Remove b' and ' wrapper
                hex_data = value[2:-1]
                if hex_data:
                    # Convert hex to bytes and unpack
                    bytes_data = bytes.fromhex(hex_data.replace('\\x', ''))
                    if len(bytes_data) == 4:
                        fixed_value = struct.unpack('f', bytes_data)[0]
                        if 0 <= fixed_value <= 100:
                            original_data[field_name] = int(round(fixed_value))
                            continue
            except (ValueError, struct.error):
                pass
        
        elif value is None:
            # Set to default value
            if field_name in ['bbox_x', 'bbox_y']:
                original_data[field_name] = 10  # Default position
            else:  # bbox_width, bbox_height
                original_data[field_name] = 50  # Default size
    
    # Validate all values are now reasonable
    x = original_data.get('bbox_x', 0)
    y = original_data.get('bbox_y', 0)
    width = original_data.get('bbox_width', 0)
    height = original_data.get('bbox_height', 0)
    
    if (isinstance(x, int) and isinstance(y, int) and 
        isinstance(width, int) and isinstance(height, int) and
        0 <= x <= 100 and 0 <= y <= 100 and 
        1 <= width <= 100 and 1 <= height <= 100):
        
        return {
            'x': x,
            'y': y, 
            'width': width,
            'height': height
        }
    
    return None

if __name__ == "__main__":
    print("🔧 Starting database bounding box repair...")
    success = fix_bbox_data()
    
    if success:
        print("\n✅ Database repair completed successfully!")
        print("You can now test the jumpToDetection function.")
    else:
        print("\n❌ Database repair failed!")
        sys.exit(1)