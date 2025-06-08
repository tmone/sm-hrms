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
import cv2
import numpy as np
from collections import defaultdict

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
                    
                    # Check for review status
                    review_status_path = person_dir / 'review_status.json'
                    unconfirmed_count = 0
                    confirmed_count = 0
                    if review_status_path.exists():
                        with open(review_status_path) as f:
                            review_status = json.load(f)
                        summary = review_status.get('summary', {})
                        unconfirmed_count = summary.get('unconfirmed', 0)
                        confirmed_count = summary.get('confirmed', 0)
                    
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
                        'person_dir': str(person_dir.relative_to('processing/outputs')),
                        'unconfirmed_count': unconfirmed_count,
                        'confirmed_count': confirmed_count,
                        'recognized': metadata.get('recognized', False),
                        'recognition_confidence': metadata.get('recognition_confidence', 0)
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
    try:
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
                    'path': f"persons/{person_id}/{img_data['filename']}",
                    'frame_number': img_data.get('frame_number', 0),
                    'timestamp': img_data.get('timestamp', 0),
                    'confidence': img_data.get('confidence', 0)
                })
        
        return jsonify({
            'person_id': person_id,
            'total_images': len(images),
            'images': images
        })
    
    except Exception as e:
        print(f"[ERROR] Error getting person images: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


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


@persons_bp.route('/api/batch-test', methods=['POST'])
@login_required
def batch_test_persons():
    """Batch test selected persons for misidentification"""
    try:
        data = request.get_json()
        person_ids = data.get('person_ids', [])
        
        if not person_ids:
            return jsonify({'success': False, 'error': 'No persons selected'})
        
        print(f"[BATCH TEST] Testing {len(person_ids)} persons for misidentification")
        
        # Import recognition module
        try:
            from hr_management.processing.person_recognition_inference_simple import PersonRecognitionInferenceSimple as SimplePersonRecognitionInference
        except ImportError:
            # Fallback to the processing module version
            from processing.simple_person_recognition_inference import SimplePersonRecognitionInference
        
        # Load the default model
        config_path = Path("models/person_recognition/config.json")
        with open(config_path) as f:
            config = json.load(f)
        model_name = config.get('default_model', 'person_model_svm_20250607_181818')
        
        recognizer = SimplePersonRecognitionInference(
            model_name=model_name,
            confidence_threshold=0.5  # Lower threshold to catch more potential matches
        )
        
        # Get trained persons list
        trained_persons = list(recognizer.person_id_mapping.values())
        print(f"[BATCH TEST] Model trained on: {trained_persons}")
        
        results = []
        persons_dir = Path('processing/outputs/persons')
        
        for person_id in person_ids:
            person_dir = persons_dir / person_id
            if not person_dir.exists():
                continue
            
            print(f"[BATCH TEST] Testing {person_id}...")
            
            # Get all images for this person
            image_files = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
            
            # Test ALL images without limit
            test_images = image_files  # Process all images, no limit
            
            predictions = defaultdict(list)
            images_to_move = defaultdict(list)  # Track which images should move where
            
            print(f"[BATCH TEST] Testing {len(test_images)} images from {person_id}")
            
            for img_path in test_images:
                try:
                    # Read and process image
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Get prediction using process_cropped_image since images are already cropped
                    result = recognizer.process_cropped_image(str(img_path))
                    
                    # Extract the first person result (since it's already cropped, should only be one)
                    if result.get('persons') and len(result['persons']) > 0:
                        person_result = result['persons'][0]
                        predicted_id = person_result.get('person_id', 'unknown')
                        confidence = person_result.get('confidence', 0.0)
                    else:
                        predicted_id = 'unknown'
                        confidence = 0.0
                    
                    predictions[predicted_id].append({
                        'image': img_path.name,
                        'confidence': confidence
                    })
                    
                    # Track images that should be moved to trained persons
                    if predicted_id != 'unknown' and predicted_id in trained_persons and predicted_id != person_id:
                        images_to_move[predicted_id].append({
                            'image': img_path.name,
                            'confidence': confidence,
                            'full_path': str(img_path)
                        })
                    
                except Exception as e:
                    print(f"[BATCH TEST] Error processing {img_path.name}: {e}")
            
            # Analyze results for this person
            person_result = {
                'person_id': person_id,
                'total_images': len(image_files),
                'tested_images': len(test_images),
                'predictions': {},
                'misidentified': False,
                'split_suggestions': [],
                'images_to_move': images_to_move  # Include detailed move info
            }
            
            # Count predictions
            for pred_id, pred_list in predictions.items():
                avg_confidence = np.mean([p['confidence'] for p in pred_list])
                person_result['predictions'][pred_id] = {
                    'count': len(pred_list),
                    'percentage': len(pred_list) / len(test_images) * 100,
                    'avg_confidence': avg_confidence,
                    'images': pred_list
                }
            
            # Check for misidentification
            if person_id in trained_persons:
                # This is a trained person - check if recognized correctly
                if person_id not in predictions or len(predictions[person_id]) < len(test_images) * 0.5:
                    person_result['misidentified'] = True
                    person_result['issue'] = 'trained_not_recognized'
            else:
                # This is an untrained person - check if ANY images are recognized as trained persons
                for trained_id in trained_persons:
                    if trained_id in images_to_move and len(images_to_move[trained_id]) > 0:
                        person_result['misidentified'] = True
                        person_result['issue'] = 'untrained_recognized_as_trained'
                        # Create split suggestion for each trained person detected
                        person_result['split_suggestions'].append({
                            'split_to': trained_id,
                            'images': images_to_move[trained_id],
                            'count': len(images_to_move[trained_id]),
                            'confidence': np.mean([img['confidence'] for img in images_to_move[trained_id]])
                        })
            
            results.append(person_result)
        
        # Generate summary
        misidentified_count = sum(1 for r in results if r['misidentified'])
        total_images_to_move = sum(len(r['images_to_move'][tid]) for r in results for tid in r['images_to_move'])
        
        # Create a summary of all moves needed
        all_moves = {}
        for result in results:
            if result['images_to_move']:
                for target_id, images in result['images_to_move'].items():
                    if target_id not in all_moves:
                        all_moves[target_id] = []
                    all_moves[target_id].extend([{
                        'source_person': result['person_id'],
                        'image': img['image'],
                        'confidence': img['confidence']
                    } for img in images])
        
        return jsonify({
            'success': True,
            'tested_persons': len(results),
            'misidentified_count': misidentified_count,
            'total_images_to_move': total_images_to_move,
            'results': results,
            'model_info': {
                'name': model_name,
                'trained_persons': trained_persons
            },
            'move_summary': all_moves
        })
        
    except Exception as e:
        print(f"[BATCH TEST] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@persons_bp.route('/api/auto-split-merge', methods=['POST'])
@login_required
def auto_split_merge():
    """Automatically split and merge persons based on test results"""
    try:
        data = request.get_json()
        operations = data.get('operations', [])
        
        if not operations:
            return jsonify({'success': False, 'error': 'No operations specified'})
        
        print(f"[AUTO SPLIT/MERGE] Processing {len(operations)} operations")
        
        results = []
        persons_dir = Path('processing/outputs/persons')
        
        for op in operations:
            op_type = op.get('type')
            source_person = op.get('source_person')
            target_person = op.get('target_person')
            images = op.get('images', [])
            
            if op_type == 'split_merge':
                # Split images from source and merge to target
                print(f"[AUTO SPLIT/MERGE] Moving {len(images)} images from {source_person} to {target_person}")
                
                source_dir = persons_dir / source_person
                target_dir = persons_dir / target_person
                
                if not source_dir.exists():
                    results.append({
                        'operation': op,
                        'success': False,
                        'error': f'Source person {source_person} not found'
                    })
                    continue
                
                # Ensure target directory exists
                target_dir.mkdir(exist_ok=True)
                
                # Move images
                moved_count = 0
                for img_info in images:
                    img_filename = img_info['image']
                    src_path = source_dir / img_filename
                    dst_path = target_dir / img_filename
                    
                    if src_path.exists():
                        shutil.move(str(src_path), str(dst_path))
                        moved_count += 1
                
                # Update metadata for both persons
                update_person_metadata(source_person)
                update_person_metadata(target_person)
                
                results.append({
                    'operation': op,
                    'success': True,
                    'moved_count': moved_count
                })
                
            elif op_type == 'merge_all':
                # Merge entire person to target
                merge_result = merge_persons(target_person, [source_person])
                results.append({
                    'operation': op,
                    'success': merge_result['success'],
                    'error': merge_result.get('error')
                })
        
        # Sync metadata with database
        sync_metadata_with_database()
        
        successful_ops = sum(1 for r in results if r['success'])
        
        return jsonify({
            'success': True,
            'total_operations': len(operations),
            'successful_operations': successful_ops,
            'results': results
        })
        
    except Exception as e:
        print(f"[AUTO SPLIT/MERGE] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


def update_person_metadata(person_id):
    """Update metadata for a person after changes"""
    persons_dir = Path('processing/outputs/persons')
    person_dir = persons_dir / person_id
    
    if not person_dir.exists():
        return
    
    metadata_path = person_dir / 'metadata.json'
    
    # Count actual images
    image_files = list(person_dir.glob('*.jpg')) + list(person_dir.glob('*.png'))
    
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = {
            'person_id': person_id,
            'videos': []
        }
    
    # Update image list
    metadata['images'] = []
    for img_file in sorted(image_files)[:1000]:  # Limit to 1000 images
        metadata['images'].append({
            'filename': img_file.name,
            'confidence': 0.95
        })
    
    metadata['total_detections'] = len(image_files)
    metadata['total_images'] = len(image_files)
    metadata['updated_at'] = datetime.now().isoformat()
    
    # Save updated metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


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
            print(f"[ERROR] Primary directory not found: {primary_dir}")
            return {'success': False, 'error': f'Primary person {primary_person_id} not found at {primary_dir}'}
        
        print(f"[OK] Found primary directory: {primary_dir}")
        
        # Load primary metadata
        primary_metadata_path = primary_dir / 'metadata.json'
        with open(primary_metadata_path) as f:
            primary_metadata = json.load(f)
        
        print(f"[OK] Loaded primary metadata: {primary_metadata.get('total_detections', 0)} detections")
        
        merged_count = 0
        total_images_added = 0
        
        # Merge each person
        for person_id in persons_to_merge:
            # Find person directory in the main persons folder
            person_dir = persons_dir / person_id
            
            if not person_dir.exists():
                print(f"[WARNING] Warning: Person {person_id} not found at {person_dir}, skipping")
                continue
            
            print(f"[OK] Processing merge for {person_id} from {person_dir}")
            
            # Load metadata
            metadata_path = person_dir / 'metadata.json'
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            # Copy images to primary directory
            person_images_added = 0
            for img_data in metadata.get('images', []):
                src_img = person_dir / img_data['filename']
                if src_img.exists():
                    # Since we use UUIDs, no need to rename
                    dst_img = primary_dir / img_data['filename']
                    shutil.copy2(src_img, dst_img)
                    print(f"  [FILE] Copied: {src_img.name} -> {dst_img.name}")
                    
                    # Update image data
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
            print(f"[DELETE] Removing merged directory: {person_dir}")
            shutil.rmtree(person_dir)
            merged_count += 1
            print(f"[OK] Merged {person_id} into {primary_person_id}: {person_images_added} images moved")
        
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
            print(f"[LOG] Updated {updated} database records: person_id {numeric_id} -> {primary_numeric_id}")
        
        db.session.commit()
        
        # Update video person counts
        print("[INFO] Updating video person counts...")
        
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
            print(f"[VIDEO] Video {video.id} ({video.filename}): {old_count} -> {video.person_count} persons")
        
        db.session.commit()
        
        print(f"[OK] Merge complete: {merged_count} persons merged, {total_images_added} images added")
        print(f"[OK] Updated video person counts")
        
        # Sync all metadata files with database
        print("[PROCESSING] Synchronizing metadata files...")
        sync_metadata_with_database()
        
        return {
            'success': True,
            'merged_count': merged_count,
            'images_added': total_images_added
        }
        
    except Exception as e:
        print(f"[ERROR] Merge error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def update_datasets_after_person_deletion(deleted_person_ids):
    """Update datasets to remove deleted persons"""
    datasets_dir = Path('datasets/person_recognition')
    updated_datasets = []
    
    if not datasets_dir.exists():
        return updated_datasets
    
    print(f"[PROCESSING] Checking datasets for deleted persons: {deleted_person_ids}")
    
    # Check each dataset
    for dataset_dir in datasets_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
            
        dataset_info_path = dataset_dir / 'dataset_info.json'
        if not dataset_info_path.exists():
            continue
            
        # Load dataset info
        with open(dataset_info_path) as f:
            dataset_info = json.load(f)
        
        # Check if any deleted persons are in this dataset
        dataset_persons = set(dataset_info.get('persons', {}).keys())
        deleted_set = set(deleted_person_ids)
        
        if dataset_persons & deleted_set:
            # This dataset contains deleted persons
            print(f"[INFO] Dataset {dataset_dir.name} contains deleted persons")
            
            # Remove deleted persons from dataset
            for person_id in deleted_person_ids:
                if 'persons' in dataset_info and person_id in dataset_info['persons']:
                    # Remove person data
                    person_data = dataset_info['persons'].pop(person_id)
                    
                    # Update totals if they exist
                    if 'total_images' in dataset_info:
                        dataset_info['total_images'] -= person_data.get('images_count', 0)
                    if 'total_faces' in dataset_info:
                        dataset_info['total_faces'] -= person_data.get('faces_count', 0)
                    if 'total_embeddings' in dataset_info:
                        dataset_info['total_embeddings'] -= person_data.get('embeddings_count', 0)
                    
                    # Also update train/val totals if they exist
                    if 'total_train_images' in dataset_info:
                        dataset_info['total_train_images'] -= person_data.get('train_images_count', 0)
                    if 'total_val_images' in dataset_info:
                        dataset_info['total_val_images'] -= person_data.get('val_images_count', 0)
                    if 'total_train_features' in dataset_info:
                        dataset_info['total_train_features'] -= person_data.get('train_features_count', 0)
                    if 'total_val_features' in dataset_info:
                        dataset_info['total_val_features'] -= person_data.get('val_features_count', 0)
                    if 'total_features' in dataset_info:
                        dataset_info['total_features'] -= person_data.get('features_count', 0)
                    
                    # Remove person's files
                    for subdir in ['images', 'faces', 'features']:
                        person_subdir = dataset_dir / subdir / person_id
                        if person_subdir.exists():
                            shutil.rmtree(person_subdir)
                            print(f"   [DELETE] Removed {person_id} from {subdir}")
            
            # Update dataset info
            dataset_info['modified_at'] = datetime.now().isoformat()
            dataset_info['person_count'] = len(dataset_info['persons'])
            
            # Check if dataset is now empty
            if dataset_info['person_count'] == 0:
                # Dataset has no persons left, delete it
                try:
                    shutil.rmtree(dataset_dir)
                    print(f"   [DELETE] Deleted empty dataset: {dataset_dir.name}")
                    updated_datasets.append(f"{dataset_dir.name} (deleted - empty)")
                except Exception as e:
                    print(f"   [WARNING]  Error deleting empty dataset {dataset_dir.name}: {e}")
                    # Still save the updated info even if we can't delete
                    with open(dataset_info_path, 'w') as f:
                        json.dump(dataset_info, f, indent=2)
                    updated_datasets.append(f"{dataset_dir.name} (emptied)")
            else:
                # Save updated dataset info
                with open(dataset_info_path, 'w') as f:
                    json.dump(dataset_info, f, indent=2)
                
                updated_datasets.append(dataset_dir.name)
                print(f"   [OK] Updated dataset: {dataset_dir.name}")
    
    # Also update the main person_features.pkl if it exists
    features_pkl = Path('datasets/person_features.pkl')
    if features_pkl.exists():
        try:
            import pickle
            with open(features_pkl, 'rb') as f:
                features_data = pickle.load(f)
            
            # Remove deleted persons from features
            original_count = len(features_data.get('person_ids', []))
            if 'person_ids' in features_data:
                features_data['person_ids'] = [pid for pid in features_data['person_ids'] 
                                               if pid not in deleted_person_ids]
            
            # Update features arrays to match
            if len(features_data['person_ids']) < original_count:
                print(f"   [PROCESSING] Updating person_features.pkl")
                # This is complex - we'd need to filter the feature vectors too
                # For now, just mark it as needing regeneration
                features_pkl.rename(features_pkl.with_suffix('.pkl.old'))
                print(f"   [WARNING]  Renamed person_features.pkl to .old - regeneration needed")
                
        except Exception as e:
            print(f"   [WARNING]  Error updating person_features.pkl: {e}")
    
    return updated_datasets


def sync_metadata_with_database():
    """Synchronize all person metadata files with the database"""
    from flask import current_app
    db = current_app.db
    DetectedPerson = current_app.DetectedPerson
    Video = current_app.Video
    
    persons_dir = Path('processing/outputs/persons')
    if not persons_dir.exists():
        print("[ERROR] Persons directory not found")
        return
    
    print("[PROCESSING] Starting metadata synchronization...")
    
    # Get all person folders
    for person_dir in persons_dir.iterdir():
        if person_dir.is_dir() and person_dir.name.startswith('PERSON-'):
            person_id = person_dir.name
            numeric_id = int(person_id.replace('PERSON-', ''))
            metadata_path = person_dir / 'metadata.json'
            
            if not metadata_path.exists():
                print(f"[WARNING]  No metadata for {person_id}, creating...")
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
                print(f"[WARNING]  {person_id} has no database records")
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
                        'file_path': video.file_path,
                        'frames': []
                    }
                
                videos_info[video.id]['frames'].append({
                    'frame_number': detection.frame_number,
                    'confidence': detection.confidence,
                    'bbox': {
                        'x': detection.bbox_x,
                        'y': detection.bbox_y,
                        'width': detection.bbox_width,
                        'height': detection.bbox_height
                    } if detection.bbox_x is not None else None,
                    'timestamp': detection.timestamp
                })
                
                if detection.timestamp is not None:
                    all_timestamps.append(detection.timestamp)
            
            # Update metadata
            metadata['videos'] = list(videos_info.values())
            
            # Update first/last seen based on all detections
            if all_timestamps:
                metadata['first_seen'] = min(all_timestamps)
                metadata['last_seen'] = max(all_timestamps)
            
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
                for img_file in sorted(image_files):  # No limit, process all images
                    metadata['images'].append({
                        'filename': img_file.name,
                        'confidence': 0.95  # Default if not available
                    })
            
            # Save updated metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"[OK] Updated metadata for {person_id}: {len(detections)} detections, {len(image_files)} images")
    
    print("[OK] Metadata synchronization complete")


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


@persons_bp.route('/split', methods=['POST'])
@login_required
def split_person():
    """Split selected images from one person into a new person"""
    try:
        data = request.get_json()
        person_id = data.get('person_id')
        image_paths = data.get('image_paths', [])
        
        if not person_id or not image_paths:
            return jsonify({'success': False, 'error': 'Missing person_id or image_paths'})
        
        # Get the next available person ID
        from processing.gpu_enhanced_detection import get_next_person_id
        new_person_id = f"PERSON-{get_next_person_id():04d}"
        
        # Create new person directory
        persons_dir = Path('processing/outputs/persons')
        old_person_dir = persons_dir / person_id
        new_person_dir = persons_dir / new_person_id
        new_person_dir.mkdir(exist_ok=True)
        
        # Move selected images to new person
        moved_images = []
        for img_path in image_paths:
            # Extract just the filename from the path
            img_filename = Path(img_path).name
            old_img_path = old_person_dir / img_filename
            new_img_path = new_person_dir / img_filename
            
            if old_img_path.exists():
                shutil.move(str(old_img_path), str(new_img_path))
                moved_images.append({
                    'filename': img_filename,
                    'confidence': 0.95  # Default confidence
                })
        
        # Create metadata for new person
        new_metadata = {
            'person_id': new_person_id,
            'total_detections': len(moved_images),
            'first_appearance': 0,
            'last_appearance': 0,
            'avg_confidence': 0.95,
            'confidence': 0.95,
            'images': moved_images,
            'videos': [],
            'split_from': person_id,
            'split_date': datetime.now().isoformat()
        }
        
        with open(new_person_dir / 'metadata.json', 'w') as f:
            json.dump(new_metadata, f, indent=2)
        
        # Update old person's metadata
        old_metadata_path = old_person_dir / 'metadata.json'
        if old_metadata_path.exists():
            with open(old_metadata_path) as f:
                old_metadata = json.load(f)
            
            # Remove split images from old metadata
            remaining_images = []
            split_filenames = {img['filename'] for img in moved_images}
            for img in old_metadata.get('images', []):
                if img['filename'] not in split_filenames:
                    remaining_images.append(img)
            
            old_metadata['images'] = remaining_images
            old_metadata['total_detections'] = len(remaining_images)
            
            with open(old_metadata_path, 'w') as f:
                json.dump(old_metadata, f, indent=2)
        
        # Update the person ID counter
        from processing.gpu_enhanced_detection import update_person_id_counter
        update_person_id_counter(int(new_person_id.replace('PERSON-', '')))
        
        # Sync metadata with database
        sync_metadata_with_database()
        
        return jsonify({
            'success': True,
            'new_person_id': new_person_id,
            'images_moved': len(moved_images)
        })
        
    except Exception as e:
        print(f"[ERROR] Split error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})


@persons_bp.route('/remove-multiple', methods=['POST'])
@login_required
def remove_multiple():
    """Remove multiple persons and their associated data"""
    try:
        data = request.get_json()
        person_ids = data.get('person_ids', [])
        
        if not person_ids:
            return jsonify({'success': False, 'error': 'No persons selected'})
        
        deleted_count = 0
        persons_dir = Path('processing/outputs/persons')
        
        # Import database models
        from flask import current_app
        db = current_app.db
        DetectedPerson = current_app.DetectedPerson
        
        # Check and update datasets that use these persons
        datasets_updated = update_datasets_after_person_deletion(person_ids)
        
        for person_id in person_ids:
            try:
                # Remove person directory
                person_dir = persons_dir / person_id
                if person_dir.exists():
                    shutil.rmtree(person_dir)
                    print(f"[DELETE] Removed directory: {person_dir}")
                
                # Remove from database
                # Convert to numeric ID for database
                numeric_id = int(person_id.replace('PERSON-', '')) if person_id.startswith('PERSON-') else person_id
                deleted = DetectedPerson.query.filter_by(person_id=numeric_id).delete()
                print(f"[LOG] Deleted {deleted} database records for person_id {numeric_id}")
                
                deleted_count += 1
                
            except Exception as e:
                print(f"[ERROR] Error deleting {person_id}: {str(e)}")
                continue
        
        # Commit database changes
        db.session.commit()
        
        return jsonify({
            'success': True,
            'deleted_count': deleted_count,
            'datasets_updated': datasets_updated,
            'message': f'Successfully deleted {deleted_count} person(s)'
        })
        
    except Exception as e:
        print(f"[ERROR] Remove multiple error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@persons_bp.route('/remove-duplicates', methods=['POST'])
@login_required
def remove_duplicates_api():
    """Remove duplicate images from selected persons or all persons"""
    try:
        data = request.get_json()
        person_ids = data.get('person_ids', [])
        create_backup = data.get('create_backup', True)
        
        persons_dir = Path('processing/outputs/persons')
        
        # Optional: Create backup directory
        backup_dir = None
        if create_backup:
            backup_dir = Path('processing/outputs/duplicates_backup')
            backup_dir.mkdir(exist_ok=True)
        
        results = []
        total_removed = 0
        
        # Determine which persons to process
        if person_ids:
            # Process selected persons
            persons_to_process = [persons_dir / pid for pid in person_ids if (persons_dir / pid).exists()]
        else:
            # Process all persons
            persons_to_process = [d for d in persons_dir.iterdir() if d.is_dir() and d.name.startswith('PERSON-')]
        
        # Process each person
        for person_dir in persons_to_process:
            # Find duplicates
            duplicates = find_duplicates_in_person(person_dir)
            
            if duplicates:
                # Count duplicates
                duplicate_count = sum(len(files) - 1 for files in duplicates.values())
                
                # Remove duplicates
                removed_files = []
                for hash_val, files in duplicates.items():
                    files.sort(key=lambda x: x.name)
                    
                    # Keep first file, remove rest
                    for remove_file in files[1:]:
                        try:
                            if backup_dir:
                                # Create backup
                                backup_path = backup_dir / person_dir.name
                                backup_path.mkdir(exist_ok=True)
                                backup_file = backup_path / remove_file.name
                                shutil.copy2(remove_file, backup_file)
                            
                            # Remove the duplicate
                            remove_file.unlink()
                            removed_files.append(remove_file)
                            
                        except Exception as e:
                            print(f"Error removing {remove_file.name}: {e}")
                
                # Update metadata
                update_metadata_after_removal(person_dir, removed_files)
                
                results.append({
                    'person_id': person_dir.name,
                    'duplicates_found': duplicate_count,
                    'duplicates_removed': len(removed_files)
                })
                
                total_removed += len(removed_files)
        
        return jsonify({
            'success': True,
            'results': results,
            'total_removed': total_removed,
            'backup_created': create_backup
        })
        
    except Exception as e:
        print(f"[ERROR] Remove duplicates error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


def find_duplicates_in_person(person_dir):
    """Find visually similar duplicate images in a person folder"""
    try:
        # Try to use the lightweight visual duplicate detector
        from hr_management.processing.visual_duplicate_detector import VisualDuplicateDetector
        
        # Get all image files
        image_files = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
        
        if len(image_files) < 2:
            return {}
        
        print(f"[DUPLICATE CHECK] Checking {person_dir.name} for visual duplicates...")
        
        # Use visual duplicate detector
        detector = VisualDuplicateDetector(similarity_threshold=0.90)
        duplicates = detector.find_duplicates(image_files)
        
        # Sort groups by file size (keep largest)
        for group_key in duplicates:
            duplicates[group_key].sort(key=lambda x: x.stat().st_size, reverse=True)
        
        print(f"[DUPLICATE CHECK] Found {len(duplicates)} duplicate groups")
        
        return duplicates
        
    except ImportError:
        print("[DUPLICATE CHECK] Visual duplicate detector not available, using simple comparison")
        # Fallback to simpler method
        return find_duplicates_simple(person_dir)
    except Exception as e:
        print(f"[DUPLICATE CHECK] Error: {e}")
        import traceback
        traceback.print_exc()
        return {}


def find_duplicates_simple(person_dir):
    """Simple duplicate detection using basic image comparison"""
    # Get all image files
    image_files = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
    
    if len(image_files) < 2:
        return {}
    
    duplicates = {}
    checked = set()
    
    for i in range(len(image_files)):
        if image_files[i] in checked:
            continue
            
        current_group = [image_files[i]]
        
        for j in range(i + 1, len(image_files)):
            if image_files[j] in checked:
                continue
            
            # Compare using simple visual similarity
            similarity = calculate_visual_similarity_simple(image_files[i], image_files[j])
            
            if similarity >= 0.90:  # 90% threshold
                current_group.append(image_files[j])
                checked.add(image_files[j])
        
        if len(current_group) > 1:
            # Sort by file size
            current_group.sort(key=lambda x: x.stat().st_size, reverse=True)
            duplicates[f"group_{len(duplicates)}"] = current_group
            for img in current_group:
                checked.add(img)
    
    return duplicates


def calculate_visual_similarity_simple(img1_path, img2_path):
    """Calculate simple visual similarity using OpenCV"""
    try:
        # Read images
        img1 = cv2.imread(str(img1_path))
        img2 = cv2.imread(str(img2_path))
        
        if img1 is None or img2 is None:
            return 0.0
        
        # Resize to same size
        size = (64, 64)
        img1 = cv2.resize(img1, size)
        img2 = cv2.resize(img2, size)
        
        # Calculate color histogram similarity
        hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist1 = cv2.normalize(hist1, hist1).flatten()
        
        hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.normalize(hist2, hist2).flatten()
        
        # Calculate correlation
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        return float(similarity)
        
    except Exception as e:
        print(f"[VISUAL SIMILARITY] Error: {e}")
        return 0.0


def find_duplicates_by_size(person_dir):
    """Fallback: Find potential duplicates by file size"""
    size_map = defaultdict(list)
    
    image_files = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
    
    for img_file in image_files:
        size = img_file.stat().st_size
        size_map[size].append(img_file)
    
    # Return groups with same size
    duplicates = {}
    idx = 0
    for size, files in size_map.items():
        if len(files) > 1:
            duplicates[f"size_group_{idx}"] = files
            idx += 1
    
    return duplicates


def get_file_hash(filepath):
    """Calculate MD5 hash of a file"""
    import hashlib
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def update_metadata_after_removal(person_dir, removed_files):
    """Update metadata.json after removing duplicates"""
    metadata_path = person_dir / 'metadata.json'
    
    if not metadata_path.exists():
        return
    
    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # Get list of removed filenames
        removed_names = {f.name for f in removed_files}
        
        # Filter out removed images from metadata
        original_count = len(metadata.get('images', []))
        metadata['images'] = [
            img for img in metadata.get('images', [])
            if img['filename'] not in removed_names
        ]
        new_count = len(metadata['images'])
        
        if new_count < original_count:
            # Update counts
            metadata['total_images'] = new_count
            metadata['total_detections'] = new_count
            metadata['updated_at'] = datetime.now().isoformat()
            metadata['duplicates_removed'] = original_count - new_count
            
            # Save updated metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"  Updated metadata: {original_count} -> {new_count} images")
            
    except Exception as e:
        print(f"  Error updating metadata: {e}")


@persons_bp.route('/reset-all', methods=['POST'])
@login_required
def reset_all_persons():
    """Reset all person codes and start from PERSON-0001"""
    try:
        from flask import current_app
        db = current_app.db
        DetectedPerson = current_app.DetectedPerson
        Video = current_app.Video
        
        # 1. Clear all person folders
        persons_dir = Path('processing/outputs/persons')
        all_person_ids = []
        if persons_dir.exists():
            # Collect all person IDs before deletion
            person_folders = list(persons_dir.glob('PERSON-*'))
            all_person_ids = [folder.name for folder in person_folders]
            
            # Remove all PERSON-* directories
            for folder in person_folders:
                try:
                    shutil.rmtree(folder)
                except Exception as e:
                    print(f"[WARNING] Error removing {folder.name}: {e}")
            
            # Remove the counter file
            counter_file = persons_dir / 'person_id_counter.json'
            if counter_file.exists():
                counter_file.unlink()
        else:
            persons_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. Reset the person ID counter to 0
        counter_file = persons_dir / 'person_id_counter.json'
        counter_data = {
            'last_person_id': 0,
            'updated_at': None,
            'total_persons': 0
        }
        with open(counter_file, 'w') as f:
            json.dump(counter_data, f, indent=2)
        
        # 3. Clear database records
        person_count = DetectedPerson.query.count()
        
        if person_count > 0:
            # Clear all DetectedPerson records
            DetectedPerson.query.delete()
            
            # Update all videos to have 0 person count
            videos = Video.query.all()
            for video in videos:
                video.person_count = 0
            
            # Commit changes
            db.session.commit()
        
        # 4. Update all datasets to remove all persons
        if all_person_ids:
            datasets_updated = update_datasets_after_person_deletion(all_person_ids)
            if datasets_updated:
                print(f"[INFO] Updated {len(datasets_updated)} datasets after clearing all persons")
        
        return jsonify({
            'success': True,
            'message': 'All person codes have been reset. Next person will be PERSON-0001',
            'persons_cleared': person_count
        })
        
    except Exception as e:
        print(f"[ERROR] Reset all error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})