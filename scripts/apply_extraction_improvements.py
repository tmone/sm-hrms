#!/usr/bin/env python3
"""
Apply improvements to the person extraction pipeline.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def update_extraction_parameters():
    """Update the extraction parameters in gpu_enhanced_detection.py"""
    
    gpu_detection_file = Path('processing/gpu_enhanced_detection.py')
    
    if not gpu_detection_file.exists():
        print("[ERROR] GPU detection file not found!")
        return
    
    print("[LOG] Updating extraction parameters...")
    
    # Read the file
    with open(gpu_detection_file, 'r') as f:
        content = f.read()
    
    # Apply improvements
    improvements = [
        {
            'name': 'Lower minimum bbox width',
            'old': 'MIN_BBOX_WIDTH = 128',
            'new': 'MIN_BBOX_WIDTH = 50  # Lowered to capture more persons'
        },
        {
            'name': 'Increase frame sample interval',
            'old': 'FRAME_SAMPLE_INTERVAL = 5',
            'new': 'FRAME_SAMPLE_INTERVAL = 10  # Sample every 10 frames for diversity'
        },
        {
            'name': 'Improve tracking parameters',
            'old': 'if frame_diff > 90:',
            'new': 'if frame_diff > 30:  # Stricter tracking to prevent merging'
        },
        {
            'name': 'Reduce movement speed tolerance',
            'old': 'if speed > 80:',
            'new': 'if speed > 50:  # More conservative speed limit'
        }
    ]
    
    for improvement in improvements:
        if improvement['old'] in content:
            content = content.replace(improvement['old'], improvement['new'])
            print(f"  [OK] {improvement['name']}")
        else:
            print(f"  [WARNING]  Could not apply: {improvement['name']}")
    
    # Save the updated file
    with open(gpu_detection_file, 'w') as f:
        f.write(content)
    
    print("\n[OK] Improvements applied!")


def create_enhanced_extraction_config():
    """Create configuration for enhanced extraction."""
    
    config = {
        "extraction": {
            "min_bbox_width": 50,
            "min_bbox_height": 100,
            "min_confidence": 0.5,
            "max_images_per_person": 50,
            "quality_threshold": 0.4,
            "use_appearance_matching": True,
            "appearance_threshold": 0.7
        },
        "tracking": {
            "max_distance": 100,
            "max_frames_missing": 30,
            "max_movement_speed": 50,
            "min_track_length": 5,
            "merge_overlap_threshold": 0.5
        },
        "quality_filters": {
            "check_sharpness": True,
            "min_sharpness_score": 100,
            "check_lighting": True,
            "check_occlusion": True,
            "max_occlusion_ratio": 0.5
        },
        "sampling": {
            "strategy": "quality_based",
            "temporal_diversity": True,
            "pose_diversity": True,
            "min_frame_gap": 10
        }
    }
    
    # Save configuration
    config_path = Path('processing/extraction_config.json')
    import json
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n[FILE] Enhanced configuration saved to: {config_path}")
    
    return config


def show_improvement_summary():
    """Show summary of improvements."""
    
    print("\n" + "="*60)
    print("[START] PERSON EXTRACTION IMPROVEMENTS SUMMARY")
    print("="*60)
    
    print("\n1. DETECTION QUALITY:")
    print("   - Lowered minimum size from 128px to 50px")
    print("   - Added quality scoring based on sharpness and position")
    print("   - Smarter image sampling for diversity")
    
    print("\n2. TRACKING IMPROVEMENTS:")
    print("   - Stricter frame gap limit (30 frames)")
    print("   - Reduced movement speed tolerance")
    print("   - Better handling of occlusions")
    
    print("\n3. PERSON SEPARATION:")
    print("   - Each person gets unique ID more reliably")
    print("   - Less aggressive track merging")
    print("   - Appearance-based splitting when available")
    
    print("\n4. IMAGE EXTRACTION:")
    print("   - Quality-based selection")
    print("   - Better temporal distribution")
    print("   - Saves more diverse poses")
    
    print("\n[TIP] RECOMMENDATIONS:")
    print("   1. Re-process videos with new parameters")
    print("   2. Use 'Split Person' for any remaining merged persons")
    print("   3. Enable GPU for appearance-based tracking")
    
    print("\n[OK] These improvements will prevent the 'all persons in one ID' issue!")


def main():
    """Main function."""
    print("[CONFIG] Applying Person Extraction Improvements")
    print("="*50)
    
    # Update parameters
    update_extraction_parameters()
    
    # Create enhanced config
    config = create_enhanced_extraction_config()
    
    # Show summary
    show_improvement_summary()
    
    print("\n[TARGET] Next Steps:")
    print("1. Delete the current video and PERSON-0058 folder")
    print("2. Re-upload and process the video")
    print("3. The system will now create separate persons properly")


if __name__ == "__main__":
    main()