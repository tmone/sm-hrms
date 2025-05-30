#!/usr/bin/env python3
"""
Initialize system settings
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app

def initialize_settings():
    """Initialize system settings"""
    app = create_app()
    
    with app.app_context():
        SystemSettings = app.SystemSettings
        db = app.db
        
        print("Initializing system settings...")
        
        # Check if settings already exist
        existing_count = SystemSettings.query.count()
        print(f"Existing settings: {existing_count}")
        
        # Initialize defaults
        SystemSettings.initialize_defaults()
        
        # List all settings
        print("\nCurrent settings:")
        settings_groups = SystemSettings.get_all_by_category()
        
        for category, settings in settings_groups.items():
            print(f"\n{category.upper()}:")
            for setting in settings:
                print(f"  - {setting.key}: {setting.value} ({setting.value_type})")
        
        print("\n✅ Settings initialized successfully!")
        print("\nYou can now access settings at: http://localhost:5001/settings/")

if __name__ == '__main__':
    initialize_settings()