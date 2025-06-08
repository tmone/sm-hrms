#!/usr/bin/env python3
"""
Fix the aggressive track merging that causes all persons to be grouped as one.
"""

import os
import sys
from pathlib import Path

def fix_track_merging():
    """Fix the aggressive merging in validate_and_merge_tracks function."""
    
    gpu_detection_file = Path('processing/gpu_enhanced_detection.py')
    
    if not gpu_detection_file.exists():
        print("[ERROR] GPU detection file not found!")
        return False
    
    print("[CONFIG] Fixing aggressive track merging...")
    
    # Read the file
    with open(gpu_detection_file, 'r') as f:
        lines = f.readlines()
    
    # Find and fix the problematic line
    fixed = False
    for i, line in enumerate(lines):
        # Fix the overlap threshold
        if "if overlap > min(len(existing_frames), len(current_frames)) * 0.3:" in line:
            # Change from 0.3 (30%) to 0.8 (80%) - much stricter
            lines[i] = line.replace("* 0.3:", "* 0.8:  # Increased from 0.3 to prevent merging different people")
            print(f"  [OK] Fixed line {i+1}: Changed overlap threshold from 30% to 80%")
            fixed = True
            break
    
    if fixed:
        # Save the file
        with open(gpu_detection_file, 'w') as f:
            f.writelines(lines)
        print("  [OK] File updated successfully!")
        return True
    else:
        print("  [WARNING]  Could not find the line to fix")
        return False


def create_better_merge_function():
    """Create an improved track validation function."""
    
    improved_function = '''def validate_and_merge_tracks_improved(person_tracks):
    """
    Improved track validation that prevents merging different people.
    Only merges tracks that are truly the same person.
    """
    merged_tracks = {}
    
    for person_id, detections in person_tracks.items():
        if not detections:
            continue
            
        # Sort detections by frame number
        detections.sort(key=lambda d: d['frame_number'])
        
        # Check if this track might be a duplicate
        is_duplicate = False
        
        for existing_id, existing_detections in merged_tracks.items():
            # Get frame ranges
            existing_start = existing_detections[0]['frame_number']
            existing_end = existing_detections[-1]['frame_number']
            current_start = detections[0]['frame_number']
            current_end = detections[-1]['frame_number']
            
            # Only consider merging if tracks are consecutive (not overlapping)
            # This prevents merging different people in the same frame
            if current_start > existing_end + 5 or existing_start > current_end + 5:
                # Tracks are separated in time - could be same person reappearing
                
                # Check spatial proximity at boundaries
                if current_start > existing_end:
                    # Check if end of existing matches start of current
                    last_existing = existing_detections[-1]
                    first_current = detections[0]
                    
                    # Calculate distance
                    dx = first_current['x'] - last_existing['x']
                    dy = first_current['y'] - last_existing['y']
                    distance = (dx**2 + dy**2)**0.5
                    
                    # Only merge if very close and gap is small
                    frame_gap = current_start - existing_end
                    if distance < 50 and frame_gap < 30:
                        print(f"[PROCESSING] Merging consecutive track {person_id} into {existing_id}")
                        existing_detections.extend(detections)
                        is_duplicate = True
                        break
            
            # Never merge overlapping tracks - they are different people!
        
        if not is_duplicate:
            merged_tracks[person_id] = detections
    
    # Re-sort all detections
    for person_id in merged_tracks:
        merged_tracks[person_id].sort(key=lambda d: d['frame_number'])
    
    print(f"[OK] Track validation complete: {len(person_tracks)} tracks -> {len(merged_tracks)} tracks")
    
    return merged_tracks
'''
    
    # Save the improved function
    with open('processing/improved_track_validation.py', 'w') as f:
        f.write(improved_function)
    
    print("\n[FILE] Created improved track validation function")
    print("   This function only merges tracks that are:")
    print("   - Consecutive in time (not overlapping)")
    print("   - Spatially close at boundaries")
    print("   - Within a small time gap")


def explain_the_issue():
    """Explain why all persons were merged into one."""
    
    print("\n" + "="*60)
    print("❓ WHY ALL PERSONS WERE GROUPED AS ONE")
    print("="*60)
    
    print("\nThe issue is in the `validate_and_merge_tracks` function:")
    print("\n1. It merges tracks with only 30% frame overlap")
    print("2. In a crowded scene, many people appear in the same frames")
    print("3. Person A overlaps 30% with Person B")
    print("4. Person B overlaps 30% with Person C")
    print("5. Result: A, B, and C all get merged into one person!")
    
    print("\n[CONFIG] THE FIX:")
    print("- Increased overlap threshold from 30% to 80%")
    print("- This prevents merging different people")
    print("- Only truly duplicate tracks will be merged")
    
    print("\n[TIP] BETTER SOLUTION:")
    print("- Never merge overlapping tracks (they're different people!)")
    print("- Only merge consecutive tracks that are close in space")
    print("- Use appearance features when available")


def main():
    """Main function."""
    print("[CONFIG] Fixing Aggressive Track Merging")
    print("="*50)
    
    # Apply the fix
    if fix_track_merging():
        print("\n[OK] Fix applied successfully!")
        
        # Create improved function
        create_better_merge_function()
        
        # Explain the issue
        explain_the_issue()
        
        print("\n[TARGET] NEXT STEPS:")
        print("1. The merging issue is now fixed")
        print("2. Re-process your video to get proper person separation")
        print("3. Or use the Split Person feature to manually separate PERSON-0058")
    else:
        print("\n[ERROR] Could not apply fix automatically")
        print("Please manually edit gpu_enhanced_detection.py")
        print("Change the line with '* 0.3:' to '* 0.8:'")


if __name__ == "__main__":
    main()