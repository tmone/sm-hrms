def validate_and_merge_tracks_improved(person_tracks):
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
                        print(f"🔄 Merging consecutive track {person_id} into {existing_id}")
                        existing_detections.extend(detections)
                        is_duplicate = True
                        break
            
            # Never merge overlapping tracks - they are different people!
        
        if not is_duplicate:
            merged_tracks[person_id] = detections
    
    # Re-sort all detections
    for person_id in merged_tracks:
        merged_tracks[person_id].sort(key=lambda d: d['frame_number'])
    
    print(f"✅ Track validation complete: {len(person_tracks)} tracks → {len(merged_tracks)} tracks")
    
    return merged_tracks
