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
    for video in videos:
        # Check if persons directory exists
        persons_dir = Path('processing/outputs') / f"detected_{Path(video.file_path).stem}" / "persons"
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
                        
                        persons_data.append({
                            'person_id': metadata['person_id'],
                            'video_id': video.id,
                            'video_filename': video.filename,
                            'total_detections': metadata['total_detections'],
                            'first_appearance': metadata['first_appearance'],
                            'last_appearance': metadata['last_appearance'],
                            'duration': metadata['last_appearance'] - metadata['first_appearance'],
                            'avg_confidence': metadata['avg_confidence'],
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
    for video in videos:
        persons_dir = Path('processing/outputs') / f"detected_{Path(video.file_path).stem}" / "persons"
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
                        
                        persons_data.append({
                            'person_id': metadata['person_id'],
                            'video_filename': video.filename,
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
    # Find the person directory
    outputs_dir = Path('processing/outputs')
    person_dir = None
    
    for video_dir in outputs_dir.iterdir():
        if video_dir.is_dir() and video_dir.name.startswith('detected_'):
            possible_person_dir = video_dir / 'persons' / person_id
            if possible_person_dir.exists():
                person_dir = possible_person_dir
                break
    
    if not person_dir:
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
    try:
        outputs_dir = Path('processing/outputs')
        primary_dir = None
        
        # Find primary person directory
        for video_dir in outputs_dir.iterdir():
            if video_dir.is_dir() and video_dir.name.startswith('detected_'):
                possible_dir = video_dir / 'persons' / primary_person_id
                if possible_dir.exists():
                    primary_dir = possible_dir
                    break
        
        if not primary_dir:
            return {'success': False, 'error': f'Primary person {primary_person_id} not found'}
        
        # Load primary metadata
        primary_metadata_path = primary_dir / 'metadata.json'
        with open(primary_metadata_path) as f:
            primary_metadata = json.load(f)
        
        merged_count = 0
        total_images_added = 0
        
        # Merge each person
        for person_id in persons_to_merge:
            # Find person directory
            person_dir = None
            for video_dir in outputs_dir.iterdir():
                if video_dir.is_dir() and video_dir.name.startswith('detected_'):
                    possible_dir = video_dir / 'persons' / person_id
                    if possible_dir.exists():
                        person_dir = possible_dir
                        break
            
            if not person_dir:
                print(f"Warning: Person {person_id} not found, skipping")
                continue
            
            # Load metadata
            metadata_path = person_dir / 'metadata.json'
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            # Copy images to primary directory
            for img_data in metadata.get('images', []):
                src_img = person_dir / img_data['filename']
                if src_img.exists():
                    # Rename to avoid conflicts
                    new_filename = f"{person_id}_{img_data['filename']}"
                    dst_img = primary_dir / new_filename
                    shutil.copy2(src_img, dst_img)
                    
                    # Update image data
                    img_data['filename'] = new_filename
                    img_data['original_person_id'] = person_id
                    primary_metadata['images'].append(img_data)
                    total_images_added += 1
            
            # Update detection counts
            primary_metadata['total_detections'] += metadata['total_detections']
            
            # Update time ranges
            if metadata['first_appearance'] < primary_metadata['first_appearance']:
                primary_metadata['first_appearance'] = metadata['first_appearance']
            if metadata['last_appearance'] > primary_metadata['last_appearance']:
                primary_metadata['last_appearance'] = metadata['last_appearance']
            
            # Remove the merged person directory
            shutil.rmtree(person_dir)
            merged_count += 1
        
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
        
        # Update DetectedPerson records
        for person_id in persons_to_merge:
            DetectedPerson.query.filter_by(person_id=person_id).update({
                'person_id': primary_person_id
            })
        
        db.session.commit()
        
        return {
            'success': True,
            'merged_count': merged_count,
            'images_added': total_images_added
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}