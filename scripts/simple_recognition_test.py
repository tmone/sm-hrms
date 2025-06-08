#!/usr/bin/env python3
"""Simple test to check if recognition is working"""

import json
from pathlib import Path

print("🧪 Simple Recognition Test\n")

# 1. Check model files
model_dir = Path('models/person_recognition/refined_quick_20250606_054446')
print(f"1. Model directory: {model_dir}")
print(f"   Exists: {model_dir.exists()}")

if model_dir.exists():
    files = list(model_dir.iterdir())
    print(f"   Files: {[f.name for f in files]}")
    
    # Check persons.json
    persons_file = model_dir / 'persons.json'
    if persons_file.exists():
        with open(persons_file) as f:
            persons_data = json.load(f)
        print(f"\n2. persons.json contains {len(persons_data)} persons:")
        for pid in list(persons_data.keys())[:5]:
            print(f"   - {pid}")
            
# 2. Check PersonIDManager mappings
mapping_file = Path('processing/outputs/person_id_mappings.json')
if mapping_file.exists():
    print(f"\n3. Found person_id_mappings.json")
    with open(mapping_file) as f:
        mappings = json.load(f)
    print(f"   Contains {len(mappings.get('recognized_to_person_id', {}))} mappings")
else:
    print(f"\n3. No person_id_mappings.json found")
    
# 3. Check if we can import modules
print("\n4. Import test:")
try:
    from processing.person_id_manager import PersonIDManager
    print("   [OK] Can import PersonIDManager")
except Exception as e:
    print(f"   [ERROR] Cannot import PersonIDManager: {e}")
    
try:
    from processing.shared_state_manager_improved import ImprovedSharedStateManagerV3
    print("   [OK] Can import ImprovedSharedStateManagerV3")
except Exception as e:
    print(f"   [ERROR] Cannot import ImprovedSharedStateManagerV3: {e}")

print("\n[OK] Basic checks complete!")