#!/usr/bin/env python3
"""Trace where recognition should happen in video processing"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("[SEARCH] Tracing Video Processing Flow\n")

# Find which processing script is being used
print("1. Checking processing files...")

processing_files = [
    'processing/gpu_enhanced_detection.py',
    'processing/enhanced_detection.py', 
    'processing/chunked_video_processor.py',
    'processing/tasks.py',
    'hr_management/processing/tasks.py',
    'hr_management/processing/enhanced_detection.py'
]

for file_path in processing_files:
    if Path(file_path).exists():
        print(f"   [CHECK] Found: {file_path}")
        
        # Check if it has recognition
        with open(file_path) as f:
            content = f.read()
            
        if 'recogniz' in content.lower():
            print(f"     Has recognition code")
        else:
            print(f"     NO recognition code")
            
        # Check what it logs
        if 'Created PERSON-' in content:
            print(f"     Creates PERSON folders")

print("\n2. Based on your output:")
print("   - Creates PERSON-0022, PERSON-0023 (new IDs)")
print("   - Shows '[CAMERA] Extracting person images'")
print("   - Shows '[TARGET] GPU detection completed'")
print("   - NO recognition messages")

print("\n3. This suggests GPU detection is being used...")

# Check GPU detection
gpu_file = Path('processing/gpu_enhanced_detection.py')
if gpu_file.exists():
    print(f"\n   Checking {gpu_file}...")
    with open(gpu_file) as f:
        lines = f.readlines()
        
    # Find where PERSON IDs are assigned
    for i, line in enumerate(lines):
        if 'PERSON-' in line and ('format' in line or 'f"PERSON' in line):
            print(f"   Line {i+1}: {line.strip()}")
            
            # Show context
            if i > 0:
                print(f"   Line {i}: {lines[i-1].strip()}")

print("\n4. The issue is likely:")
print("   - GPU detection creates new PERSON IDs directly")
print("   - It doesn't attempt recognition before assigning IDs")
print("   - Recognition code exists but isn't being called")

print("\n5. To fix this, GPU detection needs to:")
print("   1. Extract person image")
print("   2. Try recognition BEFORE creating new ID")
print("   3. Only create new ID if recognition fails")