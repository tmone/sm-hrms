"""
Person management blueprint for viewing and merging detected persons
"""
from flask import Blueprint, render_template, jsonify, request, redirect, url_for, flash, send_file
from flask_login import login_required
import os
import json
from pathlib import Path
import shutil
from datetime import datetime
from sqlalchemy import func

persons_bp = Blueprint('persons', __name__, url_prefix='/persons')

@persons_bp.route('/')
@login_required
def index():
    """Display all detected persons from all videos"""
    from flask import current_app
    db = current_app.db
    Video = current_app.Video
    DetectedPerson = current_app.DetectedPerson
    
    # Get all videos with persons
    videos = Video.query.filter(Video.status == 'completed').all()
    
    persons_data = []
    # Check for persons in the main outputs directory
    persons_dir = Path('processing/outputs') / "persons"
    
    if persons_dir.exists():
        for person_dir in persons_dir.iterdir():
            if person_dir.is_dir() and person_dir.name.startswith('PERSON-'):
                metadata_path = person_dir / 'metadata.json'
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    
                    # Get sample images
                    images = []
                    for img_data in metadata.get('images', [])[:3]:  # First 3 images
                        img_path = person_dir / img_data['filename']
                        if img_path.exists():
                            images.append({
                                'filename': img_data['filename'],
                                'path': str(img_path.relative_to('processing/outputs')),
                                'confidence': img_data['confidence']
                            })
                    
                    # Skip persons with no images
                    if not images:
                        # Also check if any actual image files exist in the folder
                        image_files = list(person_dir.glob('*.jpg')) + list(person_dir.glob('*.png'))
                        if not image_files:
                            continue
                    
                    # Get video info from metadata
                    video_info_list = metadata.get('videos', [])
                    
                    # For display purposes, we'll show the first video this person appears in
                    # If person appears in multiple videos, we could enhance this later
                    video_id = None
                    video_filename = "Unknown"
                    if video_info_list:
                        first_video = video_info_list[0]
                        video_id = first_video.get('video_id')
                        video_filename = first_video.get('filename', 'Unknown')
                    
                    # Calculate total detections and other stats from metadata
                    total_detections = metadata.get('total_detections', 0)
                    if not total_detections and video_info_list:
                        # Calculate from videos if not in metadata
                        total_detections = sum(len(v.get('frames', [])) for v in video_info_list)
                    
                    # Get first/last appearance times
                    first_appearance = metadata.get('first_appearance', 0)
                    last_appearance = metadata.get('last_appearance', 0)
                    
                    # Calculate duration (handle case where times might be frames or seconds)
                    duration = last_appearance - first_appearance if last_appearance > first_appearance else 0
                    
                    persons_data.append({
                        'person_id': metadata['person_id'],
                        'video_id': video_id,
                        'video_filename': video_filename,
                        'video_count': len(video_info_list),  # How many videos this person appears in
                        'total_detections': total_detections,
                        'first_appearance': first_appearance,
                        'last_appearance': last_appearance,
                        'duration': duration,
                        'avg_confidence': metadata.get('avg_confidence', metadata.get('confidence', 0)),
                        'images': images,
                        'image_count': len(metadata.get('images', [])),
                        'person_dir': str(person_dir.relative_to('processing/outputs'))
                    })
    
    # Sort by person_id
    persons_data.sort(key=lambda x: x['person_id'])
    
    return render_template('persons/index.html', persons=persons_data)


@persons_bp.route('/merge', methods=['GET', 'POST'])
@login_required
def merge():
    """Merge multiple person IDs into one"""
    if request.method == 'POST':
        primary_person = request.form.get('primary_person')
        persons_to_merge = request.form.getlist('persons_to_merge')
        
        if not primary_person or not persons_to_merge:
            flash('Please select a primary person and at least one person to merge', 'error')
            return redirect(url_for('persons.merge'))
        
        if primary_person in persons_to_merge:
            persons_to_merge.remove(primary_person)
        
        if not persons_to_merge:
            flash('Please select different persons to merge', 'error')
            return redirect(url_for('persons.merge'))
        
        # Perform the merge
        merge_result = merge_persons(primary_person, persons_to_merge)
        
        if merge_result['success']:
            flash(f"Successfully merged {len(persons_to_merge)} persons into {primary_person}", 'success')
            return redirect(url_for('persons.index'))
        else:
            flash(f"Error merging persons: {merge_result['error']}", 'error')
    
    # Get all persons for selection
    from flask import current_app
    Video = current_app.Video
    videos = Video.query.filter(Video.status == 'completed').all()
    
    persons_data = []
    # Check for persons in the main outputs directory
    persons_dir = Path('processing/outputs') / "persons"
    
    if persons_dir.exists():
        for person_dir in persons_dir.iterdir():
            if person_dir.is_dir() and person_dir.name.startswith('PERSON-'):
                metadata_path = person_dir / 'metadata.json'
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    
                    # Get first image
                    first_image = None
                    if metadata.get('images'):
                        img_filename = metadata['images'][0]['filename']
                        img_path = person_dir / img_filename
                        if img_path.exists():
                            first_image = str(img_path.relative_to('processing/outputs'))
                    
                    # Find video info
                    video_filename = "Unknown"
                    if videos:
                        video_filename = videos[0].filename  # Simple heuristic
                    
                    persons_data.append({
                        'person_id': metadata['person_id'],
                        'video_filename': video_filename,
                        'total_detections': metadata['total_detections'],
                        'avg_confidence': metadata['avg_confidence'],
                        'first_image': first_image,
                        'person_path': str(person_dir)
                    })
    
    persons_data.sort(key=lambda x: x['person_id'])
    
    return render_template('persons/merge.html', persons=persons_data)


@persons_bp.route('/api/<person_id>/images')
@login_required
def get_person_images(person_id):
    """Get all images for a person"""
    # Persons are now in a single directory
    persons_dir = Path('processing/outputs/persons')
    person_dir = persons_dir / person_id
    
    if not person_dir.exists():
        return jsonify({'error': 'Person not found'}), 404
    
    metadata_path = person_dir / 'metadata.json'
    if not metadata_path.exists():
        return jsonify({'error': 'Metadata not found'}), 404
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    images = []
    for img_data in metadata.get('images', []):
        img_path = person_dir / img_data['filename']
        if img_path.exists():
            images.append({
                'filename': img_data['filename'],
                'path': str(img_path.relative_to('processing/outputs')),
                'frame_number': img_data['frame_number'],
                'timestamp': img_data['timestamp'],
                'confidence': img_data['confidence']
            })
    
    return jsonify({
        'person_id': person_id,
        'total_images': len(images),
        'images': images
    })


@persons_bp.route('/serve/<path:filepath>')
@login_required
def serve_person_image(filepath):
    """Serve person images from processing outputs"""
    file_path = Path('processing/outputs') / filepath
    
    if not file_path.exists() or not file_path.is_file():
        return "File not found", 404
    
    # Security check - ensure path doesn't escape outputs directory
    try:
        file_path.resolve().relative_to(Path('processing/outputs').resolve())
    except ValueError:
        return "Invalid file path", 403
    
    return send_file(str(file_path))


def merge_persons(primary_person_id, persons_to_merge):
    """
    Merge multiple persons into one primary person
    
    Args:
        primary_person_id: The person ID to keep
        persons_to_merge: List of person IDs to merge into primary
        
    Returns:
        dict with success status and message
    """
    print(f"=== Starting merge: Primary={primary_person_id}, To merge={persons_to_merge}")
    try:
        # Persons are now in a single directory
        persons_dir = Path('processing/outputs/persons')
        primary_dir = persons_dir / primary_person_id
        
        if not primary_dir.exists():
            print(f"❌ Primary directory not found: {primary_dir}")
            return {'success': False, 'error': f'Primary person {primary_person_id} not found at {primary_dir}'}
        
        print(f"✅ Found primary directory: {primary_dir}")
        
        # Load primary metadata
        primary_metadata_path = primary_dir / 'metadata.json'
        with open(primary_metadata_path) as f:
            primary_metadata = json.load(f)
        
        print(f"✅ Loaded primary metadata: {primary_metadata.get('total_detections', 0)} detections")
        
        merged_count = 0
        total_images_added = 0
        
        # Merge each person
        for person_id in persons_to_merge:
            # Find person directory in the main persons folder
            person_dir = persons_dir / person_id
            
            if not person_dir.exists():
                print(f"⚠️ Warning: Person {person_id} not found at {person_dir}, skipping")
                continue
            
            print(f"✅ Processing merge for {person_id} from {person_dir}")
            
            # Load metadata
            metadata_path = person_dir / 'metadata.json'
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            # Copy images to primary directory
            person_images_added = 0
            for img_data in metadata.get('images', []):
                src_img = person_dir / img_data['filename']
                if src_img.exists():
                    # Rename to avoid conflicts
                    new_filename = f"{person_id}_{img_data['filename']}"
                    dst_img = primary_dir / new_filename
                    shutil.copy2(src_img, dst_img)
                    print(f"  📁 Copied: {src_img.name} → {dst_img.name}")
                    
                    # Update image data
                    img_data['filename'] = new_filename
                    img_data['original_person_id'] = person_id
                    primary_metadata['images'].append(img_data)
                    total_images_added += 1
                    person_images_added += 1
            
            # Update detection counts
            primary_metadata['total_detections'] += metadata['total_detections']
            
            # Update time ranges
            if metadata['first_appearance'] < primary_metadata['first_appearance']:
                primary_metadata['first_appearance'] = metadata['first_appearance']
            if metadata['last_appearance'] > primary_metadata['last_appearance']:
                primary_metadata['last_appearance'] = metadata['last_appearance']
            
            # Remove the merged person directory
            print(f"🗑️ Removing merged directory: {person_dir}")
            shutil.rmtree(person_dir)
            merged_count += 1
            print(f"✅ Merged {person_id} into {primary_person_id}: {person_images_added} images moved")
        
        # Recalculate average confidence
        if primary_metadata['images']:
            primary_metadata['avg_confidence'] = sum(
                img['confidence'] for img in primary_metadata['images']
            ) / len(primary_metadata['images'])
        
        # Add merge history
        if 'merge_history' not in primary_metadata:
            primary_metadata['merge_history'] = []
        
        primary_metadata['merge_history'].append({
            'merged_at': datetime.now().isoformat(),
            'merged_persons': persons_to_merge,
            'images_added': total_images_added
        })
        
        # Save updated metadata
        with open(primary_metadata_path, 'w') as f:
            json.dump(primary_metadata, f, indent=2)
        
        # Update database
        from flask import current_app
        db = current_app.db
        DetectedPerson = current_app.DetectedPerson
        
        # Extract numeric IDs for database update
        primary_numeric_id = int(primary_person_id.replace('PERSON-', '')) if primary_person_id.startswith('PERSON-') else primary_person_id
        
        # Update DetectedPerson records
        for person_id in persons_to_merge:
            # Convert to numeric ID for database
            numeric_id = int(person_id.replace('PERSON-', '')) if person_id.startswith('PERSON-') else person_id
            
            updated = DetectedPerson.query.filter_by(person_id=numeric_id).update({
                'person_id': primary_numeric_id
            })
            print(f"📝 Updated {updated} database records: person_id {numeric_id} → {primary_numeric_id}")
        
        db.session.commit()
        
        # Update video person counts
        print("📊 Updating video person counts...")
        
        # Get Video model from current_app
        Video = current_app.Video
        
        # Get all videos and recalculate their person counts based on actual detections
        videos = Video.query.filter(Video.status == 'completed').all()
        for video in videos:
            # Count unique persons detected in this specific video
            unique_persons = db.session.query(DetectedPerson.person_id)\
                .filter(DetectedPerson.video_id == video.id)\
                .distinct()\
                .count()
            
            # Update video person count
            old_count = video.person_count
            video.person_count = unique_persons
            print(f"📹 Video {video.id} ({video.filename}): {old_count} → {video.person_count} persons")
        
        db.session.commit()
        
        print(f"✅ Merge complete: {merged_count} persons merged, {total_images_added} images added")
        print(f"✅ Updated video person counts")
        
        # Sync all metadata files with database
        print("🔄 Synchronizing metadata files...")
        sync_metadata_with_database()
        
        return {
            'success': True,
            'merged_count': merged_count,
            'images_added': total_images_added
        }
        
    except Exception as e:
        print(f"❌ Merge error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def sync_metadata_with_database():
    """Synchronize all person metadata files with the database"""
    from flask import current_app
    db = current_app.db
    DetectedPerson = current_app.DetectedPerson
    Video = current_app.Video
    
    persons_dir = Path('processing/outputs/persons')
    if not persons_dir.exists():
        print("❌ Persons directory not found")
        return
    
    print("🔄 Starting metadata synchronization...")
    
    # Get all person folders
    for person_dir in persons_dir.iterdir():
        if person_dir.is_dir() and person_dir.name.startswith('PERSON-'):
            person_id = person_dir.name
            numeric_id = int(person_id.replace('PERSON-', ''))
            metadata_path = person_dir / 'metadata.json'
            
            if not metadata_path.exists():
                print(f"⚠️  No metadata for {person_id}, creating...")
                metadata = {
                    'person_id': person_id,
                    'first_seen': None,
                    'last_seen': None,
                    'confidence': 0,
                    'images': [],
                    'videos': []
                }
            else:
                with open(metadata_path) as f:
                    metadata = json.load(f)
            
            # Get all detections for this person from database
            detections = DetectedPerson.query.filter_by(person_id=numeric_id).all()
            
            if not detections:
                print(f"⚠️  {person_id} has no database records")
                continue
            
            # Update metadata based on database
            videos_info = {}
            all_timestamps = []
            
            for detection in detections:
                video = Video.query.get(detection.video_id)
                if not video:
                    continue
                
                # Group detections by video
                if video.id not in videos_info:
                    videos_info[video.id] = {
                        'video_id': video.id,
                        'filename': video.filename,
                        'upload_path': video.upload_path,
                        'frames': []
                    }
                
                videos_info[video.id]['frames'].append({
                    'frame_num': detection.frame_num,
                    'confidence': detection.confidence,
                    'bbox': detection.bbox,
                    'timestamp': detection.timestamp.isoformat() if detection.timestamp else None
                })
                
                if detection.timestamp:
                    all_timestamps.append(detection.timestamp)
            
            # Update metadata
            metadata['videos'] = list(videos_info.values())
            
            # Update first/last seen based on all detections
            if all_timestamps:
                metadata['first_seen'] = min(all_timestamps).isoformat()
                metadata['last_seen'] = max(all_timestamps).isoformat()
            
            # Update confidence as average
            if detections:
                avg_confidence = sum(d.confidence for d in detections) / len(detections)
                metadata['confidence'] = round(avg_confidence, 3)
            
            # Count actual images in folder
            image_files = list(person_dir.glob('*.jpg')) + list(person_dir.glob('*.png'))
            metadata['total_images'] = len(image_files)
            
            # Update images list if needed
            if len(metadata.get('images', [])) != len(image_files):
                metadata['images'] = []
                for img_file in sorted(image_files)[:100]:  # Limit to first 100
                    metadata['images'].append({
                        'filename': img_file.name,
                        'confidence': 0.95  # Default if not available
                    })
            
            # Save updated metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Updated metadata for {person_id}: {len(detections)} detections, {len(image_files)} images")
    
    print("✅ Metadata synchronization complete")


@persons_bp.route('/sync-metadata')
@login_required
def sync_metadata():
    """Endpoint to trigger metadata synchronization"""
    try:
        sync_metadata_with_database()
        flash('Metadata synchronized successfully', 'success')
    except Exception as e:
        flash(f'Error synchronizing metadata: {str(e)}', 'error')
    
    return redirect(url_for('persons.index'))