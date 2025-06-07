#!/usr/bin/env python3
"""Add detailed logging to show recognition process during video processing"""

import sys
import os
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def add_recognition_logging():
    """Add logging to key recognition points"""
    
    print("📝 Adding detailed recognition logging...\n")
    
    # 1. Update shared_state_manager_improved.py to add more logging
    state_manager_file = Path('processing/shared_state_manager_improved.py')
    
    if state_manager_file.exists():
        print("✅ Found shared_state_manager_improved.py")
        
        # Read the file
        with open(state_manager_file) as f:
            content = f.read()
            
        # Check if we have the assign_temporary_id method
        if 'def assign_temporary_id' in content:
            print("   - Has assign_temporary_id method")
            
        # Check if we have resolve_person_ids
        if 'def resolve_person_ids' in content:
            print("   - Has resolve_person_ids method")
            
    # 2. Create a simple monitoring script
    monitor_script = Path('processing/recognition_monitor.py')
    
    monitor_content = '''"""Simple recognition monitoring for debugging"""

import logging
from datetime import datetime
from pathlib import Path

# Create a special logger for recognition
recognition_logger = logging.getLogger('recognition_monitor')
recognition_logger.setLevel(logging.INFO)

# Create file handler
log_dir = Path('processing/recognition_logs')
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f'recognition_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

fh = logging.FileHandler(log_file)
fh.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(message)s')
fh.setFormatter(formatter)

# Add handler
recognition_logger.addHandler(fh)

# Also log to console
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
recognition_logger.addHandler(ch)

def log_recognition_event(event_type, data):
    """Log a recognition event"""
    recognition_logger.info(f"[{event_type}] {data}")

# Make it globally available
print(f"Recognition logging to: {log_file}")
'''
    
    with open(monitor_script, 'w') as f:
        f.write(monitor_content)
        
    print(f"✅ Created {monitor_script}")
    
    # 3. Show how to use it
    print("\n📋 To enable recognition logging during video processing:")
    print("\n1. The system will log recognition events to:")
    print("   processing/recognition_logs/recognition_TIMESTAMP.log")
    print("\n2. Key events logged:")
    print("   - When recognition model loads/fails")
    print("   - Each recognition attempt")
    print("   - Person ID assignment decisions")
    print("   - Final ID mappings")
    
    # 4. Check current video processing logs
    print("\n📂 Checking for existing debug logs...")
    
    debug_log_dir = Path('processing/debug_logs')
    if debug_log_dir.exists():
        log_files = list(debug_log_dir.glob('*.log'))
        if log_files:
            print(f"   Found {len(log_files)} debug logs")
            
            # Show latest log
            latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
            print(f"   Latest: {latest_log.name}")
            
            # Show last few lines
            with open(latest_log) as f:
                lines = f.readlines()
                
            print(f"\n   Last 10 lines from {latest_log.name}:")
            for line in lines[-10:]:
                print(f"   {line.rstrip()}")
    else:
        print("   No debug logs found")
        
    # 5. Show what needs to be fixed
    print("\n🔧 Issues to fix:")
    print("\n1. Recognition model is missing files:")
    print("   - label_encoder.pkl")
    print("   - persons.json")
    print("\n2. The model needs these files to map predictions to person IDs")
    print("\n3. Without these files, recognition cannot work properly")
    
    print("\n💡 Solution:")
    print("1. Retrain the model to generate all required files")
    print("2. Or manually create the missing mappings")
    print("3. Or use a different model that has all files")

if __name__ == "__main__":
    add_recognition_logging()