#!/usr/bin/env python3
"""Debug person recognition during video processing"""

import sys
import os
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_recognition_setup():
    """Check if recognition is properly set up"""
    
    print("🔍 Checking Person Recognition Setup\n")
    
    # 1. Check default model configuration
    config_path = Path('models/person_recognition/config.json')
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print(f"✅ Config file exists")
        print(f"   Default model: {config.get('default_model', 'NOT SET')}")
        
        # Check if default model directory exists
        if config.get('default_model'):
            model_dir = Path('models/person_recognition') / config['default_model']
            if model_dir.exists():
                print(f"✅ Default model directory exists: {model_dir}")
                
                # Check model files
                model_files = ['model.pkl', 'label_encoder.pkl', 'scaler.pkl', 'persons.json']
                for file in model_files:
                    file_path = model_dir / file
                    if file_path.exists():
                        print(f"   ✅ {file} exists")
                        
                        # Show persons.json content if it exists
                        if file == 'persons.json':
                            with open(file_path) as f:
                                persons_data = json.load(f)
                            print(f"   📋 Trained persons: {list(persons_data.keys())}")
                    else:
                        print(f"   ❌ {file} MISSING")
            else:
                print(f"❌ Default model directory NOT FOUND: {model_dir}")
        else:
            print("❌ No default model configured")
    else:
        print("❌ Config file NOT FOUND")
    
    print("\n" + "="*60 + "\n")
    
    # 2. Check which detection module is being used
    print("🔍 Checking Detection Modules\n")
    
    # Check imports in chunked_video_processor
    chunked_processor_path = Path('processing/chunked_video_processor.py')
    if chunked_processor_path.exists():
        with open(chunked_processor_path) as f:
            content = f.read()
        
        if 'from processing.shared_state_manager_improved import ImprovedSharedStateManagerV3' in content:
            print("✅ ChunkedVideoProcessor is using ImprovedSharedStateManagerV3")
        elif 'from processing.shared_state_manager_v2 import SharedStateManagerV2' in content:
            print("⚠️  ChunkedVideoProcessor is using OLD SharedStateManagerV2")
        else:
            print("❓ Cannot determine which SharedStateManager is being used")
    
    # Check if improved manager exists
    improved_manager_path = Path('processing/shared_state_manager_improved.py')
    if improved_manager_path.exists():
        print("✅ ImprovedSharedStateManagerV3 file exists")
    else:
        print("❌ ImprovedSharedStateManagerV3 file NOT FOUND")
    
    # Check if PersonIDManager exists
    person_id_manager_path = Path('processing/person_id_manager.py')
    if person_id_manager_path.exists():
        print("✅ PersonIDManager file exists")
    else:
        print("❌ PersonIDManager file NOT FOUND")
    
    print("\n" + "="*60 + "\n")
    
    # 3. Test recognition directly
    print("🧪 Testing Recognition Module\n")
    
    try:
        from hr_management.processing.person_recognition_inference_simple import PersonRecognitionInferenceSimple
        
        # Load default model
        config_path = Path('models/person_recognition/config.json')
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            
            if config.get('default_model'):
                print(f"Loading model: {config['default_model']}")
                model = PersonRecognitionInferenceSimple(
                    config['default_model'],
                    confidence_threshold=0.7
                )
                print("✅ Model loaded successfully")
                
                # Show model info
                if hasattr(model, 'persons'):
                    print(f"   Persons in model: {list(model.persons.keys())}")
                
                # Test with a sample image if available
                test_images_dir = Path('processing/outputs/persons')
                if test_images_dir.exists():
                    # Find a person with images
                    for person_dir in test_images_dir.iterdir():
                        if person_dir.is_dir() and person_dir.name.startswith('PERSON-'):
                            images = list(person_dir.glob('*.jpg'))
                            if images:
                                test_image = images[0]
                                print(f"\n🧪 Testing with image: {test_image}")
                                
                                result = model.process_cropped_image(str(test_image))
                                if result.get('persons'):
                                    for person in result['persons']:
                                        print(f"   Result: {person['person_id']} (confidence: {person['confidence']:.2f})")
                                else:
                                    print("   No persons detected")
                                break
        else:
            print("❌ No config file to load model")
            
    except Exception as e:
        print(f"❌ Error testing recognition: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_recognition_setup()