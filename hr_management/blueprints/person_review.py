"""
Person image review blueprint for confirming/rejecting misrecognized images
"""
from flask import Blueprint, render_template, jsonify, request, redirect, url_for, flash
from flask_login import login_required
import os
import json
from pathlib import Path
import shutil
from datetime import datetime

person_review_bp = Blueprint('person_review', __name__, url_prefix='/persons/review')

@person_review_bp.route('/<person_id>')
@login_required
def review_person(person_id):
    """Review unconfirmed images for a person"""
    person_dir = Path('processing/outputs/persons') / person_id
    
    if not person_dir.exists():
        flash(f'Person {person_id} not found', 'error')
        return redirect(url_for('persons.index'))
    
    # Load metadata
    metadata_path = person_dir / 'metadata.json'
    if not metadata_path.exists():
        flash(f'No metadata for {person_id}', 'error')
        return redirect(url_for('persons.index'))
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    # Load review status
    review_status_path = person_dir / 'review_status.json'
    if review_status_path.exists():
        with open(review_status_path) as f:
            review_status = json.load(f)
    else:
        review_status = {
            'person_id': person_id,
            'images': {},
            'summary': {
                'total_images': 0,
                'confirmed': 0,
                'unconfirmed': 0
            }
        }
    
    # Get all images with their status
    images_data = []
    for img_file in sorted(person_dir.glob('*.jpg')):
        img_name = img_file.name
        
        # Get image metadata
        img_meta = None
        for img in metadata.get('images', []):
            if img.get('filename') == img_name:
                img_meta = img
                break
        
        # Get review status
        img_status = review_status['images'].get(img_name, {})
        
        images_data.append({
            'filename': img_name,
            'path': str(img_file.relative_to('processing/outputs')),
            'status': img_status.get('status', 'unconfirmed'),
            'in_dataset': img_status.get('in_dataset', False),
            'frame_number': img_meta.get('frame_number') if img_meta else None,
            'confidence': img_meta.get('confidence', 0) if img_meta else 0,
            'recognized_as': img_meta.get('recognized_as') if img_meta else None,
            'recognition_confidence': img_meta.get('recognition_confidence', 0) if img_meta else 0
        })
    
    # Sort unconfirmed first
    images_data.sort(key=lambda x: (x['status'] == 'confirmed', x['filename']))
    
    return render_template('persons/review.html',
                         person_id=person_id,
                         metadata=metadata,
                         review_status=review_status,
                         images=images_data)

@person_review_bp.route('/<person_id>/confirm', methods=['POST'])
@login_required
def confirm_images(person_id):
    """Confirm images as correctly belonging to this person and add to dataset"""
    data = request.get_json()
    image_names = data.get('images', [])
    
    if not image_names:
        return jsonify({'error': 'No images specified'}), 400
    
    person_dir = Path('processing/outputs/persons') / person_id
    review_status_path = person_dir / 'review_status.json'
    
    # Create dataset directory for this person
    dataset_dir = Path('datasets') / person_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Load current status
    if review_status_path.exists():
        with open(review_status_path) as f:
            review_status = json.load(f)
    else:
        review_status = {
            'person_id': person_id,
            'images': {},
            'summary': {}
        }
    
    confirmed_count = 0
    added_to_dataset = 0
    
    for img_name in image_names:
        source_path = person_dir / img_name
        
        if source_path.exists():
            # Copy to dataset
            dest_path = dataset_dir / img_name
            try:
                shutil.copy2(str(source_path), str(dest_path))
                added_to_dataset += 1
                
                # Update review status
                if img_name not in review_status['images']:
                    review_status['images'][img_name] = {}
                
                review_status['images'][img_name]['status'] = 'confirmed'
                review_status['images'][img_name]['in_dataset'] = True
                review_status['images'][img_name]['confirmed_at'] = datetime.now().isoformat()
                review_status['images'][img_name]['confirmed_by'] = 'user'
                review_status['images'][img_name]['dataset_path'] = str(dest_path.relative_to('datasets'))
                confirmed_count += 1
                
            except Exception as e:
                print(f"Failed to copy {img_name} to dataset: {e}")
    
    # Update dataset metadata
    dataset_meta_path = dataset_dir / 'dataset_info.json'
    if dataset_meta_path.exists():
        with open(dataset_meta_path) as f:
            dataset_meta = json.load(f)
    else:
        dataset_meta = {
            'person_id': person_id,
            'created_at': datetime.now().isoformat(),
            'images': []
        }
    
    # Add new images to dataset metadata
    existing_images = set(dataset_meta.get('images', []))
    for img_name in image_names:
        if img_name not in existing_images:
            dataset_meta['images'].append(img_name)
    
    dataset_meta['updated_at'] = datetime.now().isoformat()
    dataset_meta['total_images'] = len(dataset_meta['images'])
    
    with open(dataset_meta_path, 'w') as f:
        json.dump(dataset_meta, f, indent=2)
    
    # Update summary
    total = 0
    confirmed = 0
    in_dataset = 0
    for img_data in review_status['images'].values():
        total += 1
        if img_data.get('status') == 'confirmed':
            confirmed += 1
        if img_data.get('in_dataset'):
            in_dataset += 1
    
    review_status['summary'] = {
        'total_images': total,
        'confirmed': confirmed,
        'unconfirmed': total - confirmed,
        'in_dataset': in_dataset,
        'last_updated': datetime.now().isoformat()
    }
    
    # Save updated status
    with open(review_status_path, 'w') as f:
        json.dump(review_status, f, indent=2)
    
    return jsonify({
        'success': True,
        'confirmed': confirmed_count,
        'added_to_dataset': added_to_dataset,
        'summary': review_status['summary']
    })

@person_review_bp.route('/<person_id>/move', methods=['POST'])
@login_required
def move_images(person_id):
    """Move images to a different person"""
    data = request.get_json()
    image_names = data.get('images', [])
    target_person_id = data.get('target_person_id')
    
    if not image_names or not target_person_id:
        return jsonify({'error': 'Missing required parameters'}), 400
    
    source_dir = Path('processing/outputs/persons') / person_id
    target_dir = Path('processing/outputs/persons') / target_person_id
    
    if not source_dir.exists():
        return jsonify({'error': f'Source person {person_id} not found'}), 404
    
    # Create target if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)
    
    moved_count = 0
    errors = []
    
    for img_name in image_names:
        source_path = source_dir / img_name
        if source_path.exists():
            # Generate unique name if needed
            target_path = target_dir / img_name
            if target_path.exists():
                base, ext = os.path.splitext(img_name)
                counter = 1
                while target_path.exists():
                    new_name = f"{base}_{counter}{ext}"
                    target_path = target_dir / new_name
                    counter += 1
            
            try:
                shutil.move(str(source_path), str(target_path))
                moved_count += 1
                
                # Update metadata for both persons
                # TODO: Update metadata.json files
                
            except Exception as e:
                errors.append(f"Failed to move {img_name}: {str(e)}")
    
    # Update review status for source person
    review_status_path = source_dir / 'review_status.json'
    if review_status_path.exists():
        with open(review_status_path) as f:
            review_status = json.load(f)
    else:
        review_status = {'person_id': person_id, 'images': {}, 'summary': {}}
    
    # Mark images as moved
    for img_name in image_names:
        if img_name in review_status['images']:
            review_status['images'][img_name]['status'] = 'moved'
            review_status['images'][img_name]['moved_to'] = target_person_id
            review_status['images'][img_name]['moved_at'] = datetime.now().isoformat()
    
    # Recalculate summary for source
    remaining_images = set(img.name for img in source_dir.glob('*.jpg'))
    total = len(remaining_images)
    confirmed = 0
    unconfirmed = 0
    
    for img_name in remaining_images:
        img_data = review_status['images'].get(img_name, {})
        if img_data.get('status') == 'confirmed':
            confirmed += 1
        else:
            unconfirmed += 1
    
    review_status['summary'] = {
        'total_images': total,
        'confirmed': confirmed,
        'unconfirmed': unconfirmed,
        'last_updated': datetime.now().isoformat()
    }
    
    with open(review_status_path, 'w') as f:
        json.dump(review_status, f, indent=2)
    
    return jsonify({
        'success': True,
        'moved': moved_count,
        'errors': errors
    })

@person_review_bp.route('/<person_id>/delete', methods=['POST'])
@login_required
def delete_images(person_id):
    """Delete images from a person"""
    data = request.get_json()
    image_names = data.get('images', [])
    
    if not image_names:
        return jsonify({'error': 'No images specified'}), 400
    
    person_dir = Path('processing/outputs/persons') / person_id
    deleted_dir = person_dir / 'deleted'
    deleted_dir.mkdir(exist_ok=True)
    
    deleted_count = 0
    
    for img_name in image_names:
        img_path = person_dir / img_name
        if img_path.exists():
            # Move to deleted folder instead of permanent deletion
            dest_path = deleted_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{img_name}"
            shutil.move(str(img_path), str(dest_path))
            deleted_count += 1
    
    # Update review status
    review_status_path = person_dir / 'review_status.json'
    if review_status_path.exists():
        with open(review_status_path) as f:
            review_status = json.load(f)
    else:
        review_status = {
            'person_id': person_id,
            'images': {},
            'summary': {}
        }
    
    # Update image statuses
    for img_name in image_names:
        if img_name in review_status['images']:
            review_status['images'][img_name]['status'] = 'deleted'
            review_status['images'][img_name]['deleted_at'] = datetime.now().isoformat()
        else:
            # Add entry for deleted image
            review_status['images'][img_name] = {
                'status': 'deleted',
                'deleted_at': datetime.now().isoformat()
            }
    
    # Recalculate summary based on actual remaining images
    total = 0
    confirmed = 0
    unconfirmed = 0
    deleted = 0
    in_dataset = 0
    
    # Get all remaining images in folder
    remaining_images = set(img.name for img in person_dir.glob('*.jpg'))
    
    # First, ensure all remaining images have entries in review_status
    for img_name in remaining_images:
        if img_name not in review_status['images']:
            review_status['images'][img_name] = {
                'status': 'unconfirmed',
                'in_dataset': False
            }
    
    # Now count based on actual remaining images
    for img_name in remaining_images:
        total += 1
        img_data = review_status['images'].get(img_name, {})
        status = img_data.get('status', 'unconfirmed')
        
        if status == 'confirmed':
            confirmed += 1
        else:
            # If not explicitly confirmed, it's unconfirmed
            unconfirmed += 1
        
        if img_data.get('in_dataset'):
            in_dataset += 1
    
    # Count deleted images (those in review_status but not in folder)
    for img_name, img_data in review_status['images'].items():
        if img_name not in remaining_images and img_data.get('status') == 'deleted':
            deleted += 1
    
    review_status['summary'] = {
        'total_images': total,
        'confirmed': confirmed,
        'unconfirmed': unconfirmed,
        'deleted': deleted,
        'in_dataset': in_dataset,
        'last_updated': datetime.now().isoformat()
    }
    
    # Save updated status
    with open(review_status_path, 'w') as f:
        json.dump(review_status, f, indent=2)
    
    return jsonify({
        'success': True,
        'deleted': deleted_count,
        'summary': review_status['summary']
    })