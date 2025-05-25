#!/usr/bin/env python
"""
Verify and fix person_id consistency between database and video annotations
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db

def verify_person_ids(video_id):
    """Verify person IDs are consistent"""
    app = create_app()
    
    with app.app_context():
        # Import models from app context
        Video = app.Video
        DetectedPerson = app.DetectedPerson
        
        video = Video.query.get(video_id)
        if not video:
            print(f"Video {video_id} not found")
            return
        
        print(f"Verifying person IDs for video: {video.filename}")
        print("-" * 60)
        
        # Get all detections ordered by timestamp
        detections = DetectedPerson.query.filter_by(video_id=video_id).order_by(
            DetectedPerson.timestamp.asc()
        ).all()
        
        print(f"Total detections: {len(detections)}")
        
        # Group by person_id
        person_groups = {}
        for det in detections:
            pid = det.person_id
            if pid not in person_groups:
                person_groups[pid] = []
            person_groups[pid].append(det)
        
        print(f"\nUnique person IDs: {len(person_groups)}")
        print("\nPerson ID Summary:")
        print("-" * 40)
        
        for pid in sorted(person_groups.keys()):
            dets = person_groups[pid]
            first_time = min(d.timestamp for d in dets)
            last_time = max(d.timestamp for d in dets)
            print(f"Person ID {pid}: {len(dets)} detections, {first_time:.1f}s - {last_time:.1f}s")
        
        # Check for issues
        print("\nChecking for issues...")
        
        # Issue 1: Missing person_id
        missing_pid = [d for d in detections if d.person_id is None]
        if missing_pid:
            print(f"⚠️  {len(missing_pid)} detections have no person_id")
        
        # Issue 2: Non-numeric person_id
        non_numeric = [d for d in detections if d.person_id and not str(d.person_id).isdigit()]
        if non_numeric:
            print(f"⚠️  {len(non_numeric)} detections have non-numeric person_id")
        
        # Show sample detections
        print("\nSample detections (first 10):")
        print("-" * 60)
        print(f"{'ID':<6} {'Time':<8} {'Person':<12} {'Track':<15} {'Pos':<20} {'Conf':<6}")
        print("-" * 60)
        
        for i, det in enumerate(detections[:10]):
            pos = f"{det.bbox_x},{det.bbox_y} {det.bbox_width}x{det.bbox_height}"
            print(f"{det.id:<6} {det.timestamp:<8.2f} {str(det.person_id):<12} {str(det.track_id):<15} {pos:<20} {det.confidence:<6.2f}")

def fix_person_ids(video_id):
    """Fix person IDs to ensure they're numeric and sequential"""
    app = create_app()
    
    with app.app_context():
        DetectedPerson = app.DetectedPerson
        
        print(f"\nFixing person IDs for video {video_id}...")
        
        # Get all detections
        detections = DetectedPerson.query.filter_by(video_id=video_id).order_by(
            DetectedPerson.timestamp.asc()
        ).all()
        
        # Fix person_ids to be numeric
        fixed_count = 0
        for det in detections:
            if det.person_id:
                # Extract numeric part if it's a string like "PERSON-0001"
                if isinstance(det.person_id, str):
                    if det.person_id.startswith('PERSON-'):
                        try:
                            numeric_id = int(det.person_id.replace('PERSON-', ''))
                            det.person_id = str(numeric_id)
                            fixed_count += 1
                        except:
                            pass
                # Ensure it's stored as string
                elif isinstance(det.person_id, int):
                    det.person_id = str(det.person_id)
                    fixed_count += 1
        
        if fixed_count > 0:
            db.session.commit()
            print(f"✅ Fixed {fixed_count} person IDs")
        else:
            print("✅ All person IDs are already in correct format")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_person_ids.py <video_id> [--fix]")
        sys.exit(1)
    
    video_id = int(sys.argv[1])
    
    verify_person_ids(video_id)
    
    if len(sys.argv) > 2 and sys.argv[2] == '--fix':
        fix_person_ids(video_id)