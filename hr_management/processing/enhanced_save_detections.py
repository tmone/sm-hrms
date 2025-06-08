"""
Enhanced Save Detections with OCR Data Integration
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def save_detections_with_ocr(video_id: int, detections: List[Dict], 
                           video_ocr_data: Optional[Dict], db, DetectedPerson):
    """
    Save detections to database with OCR data integration
    
    Args:
        video_id: ID of the video
        detections: List of detection dictionaries
        video_ocr_data: OCR data extracted from video (location, date, time)
        db: Database session
        DetectedPerson: DetectedPerson model class
    """
    logger.info(f"Saving {len(detections)} detections with OCR data for video {video_id}")
    
    # Extract OCR data if available
    ocr_location = video_ocr_data.get('location') if video_ocr_data else None
    ocr_video_date = video_ocr_data.get('video_date') if video_ocr_data else None
    ocr_video_time = video_ocr_data.get('video_time') if video_ocr_data else None
    
    if ocr_location:
        logger.info(f"OCR Location: {ocr_location}")
    if ocr_video_date:
        logger.info(f"OCR Date: {ocr_video_date}")
    if ocr_video_time:
        logger.info(f"OCR Time: {ocr_video_time}")
    
    saved_count = 0
    
    for detection in detections:
        try:
            # Create DetectedPerson record
            db_detection = DetectedPerson(
                video_id=video_id,
                frame_number=detection.get('frame_number', 0),
                timestamp=detection.get('timestamp', 0.0),
                bbox_x=detection.get('x', 0),
                bbox_y=detection.get('y', 0),
                bbox_width=detection.get('width', 0),
                bbox_height=detection.get('height', 0),
                confidence=detection.get('confidence', 0.0),
                person_id=detection.get('person_id'),
                track_id=detection.get('track_id')
            )
            
            # Add OCR/Attendance data
            if ocr_location:
                db_detection.attendance_location = ocr_location
            
            if ocr_video_date:
                db_detection.attendance_date = ocr_video_date
                
                # Calculate attendance time based on detection timestamp
                if detection.get('timestamp') is not None:
                    # Add the detection timestamp to the video's base time
                    detection_offset = timedelta(seconds=float(detection['timestamp']))
                    
                    if ocr_video_time:
                        # Combine date and time
                        base_datetime = datetime.combine(ocr_video_date, ocr_video_time)
                        attendance_datetime = base_datetime + detection_offset
                        db_detection.attendance_time = attendance_datetime.time()
                        db_detection.check_in_time = attendance_datetime
                    else:
                        # Use just the offset time
                        db_detection.attendance_time = (datetime.min + detection_offset).time()
            
            db.session.add(db_detection)
            saved_count += 1
            
        except Exception as e:
            logger.error(f"Error saving detection {detection.get('person_id', 'unknown')}: {e}")
            continue
    
    try:
        db.session.commit()
        logger.info(f"Successfully saved {saved_count} detections to database")
    except Exception as e:
        logger.error(f"Error committing detections: {e}")
        db.session.rollback()
        raise
    
    return saved_count


def update_detections_with_ocr(video_id: int, ocr_data: Dict, db, DetectedPerson):
    """
    Update existing detections with OCR data
    
    Args:
        video_id: ID of the video
        ocr_data: OCR data extracted from video
        db: Database session
        DetectedPerson: DetectedPerson model class
    """
    logger.info(f"Updating detections with OCR data for video {video_id}")
    
    # Get all detections for this video
    detections = DetectedPerson.query.filter_by(video_id=video_id).all()
    
    if not detections:
        logger.warning(f"No detections found for video {video_id}")
        return 0
    
    updated_count = 0
    
    for detection in detections:
        updated = False
        
        # Update location if not already set
        if not detection.attendance_location and ocr_data.get('location'):
            detection.attendance_location = ocr_data['location']
            updated = True
        
        # Update date if not already set
        if not detection.attendance_date and ocr_data.get('video_date'):
            detection.attendance_date = ocr_data['video_date']
            updated = True
            
            # Calculate attendance time
            if detection.timestamp is not None:
                detection_offset = timedelta(seconds=float(detection.timestamp))
                
                if ocr_data.get('video_time'):
                    # Combine date and time
                    base_datetime = datetime.combine(ocr_data['video_date'], ocr_data['video_time'])
                    attendance_datetime = base_datetime + detection_offset
                    detection.attendance_time = attendance_datetime.time()
                    detection.check_in_time = attendance_datetime
                else:
                    # Use just the offset time
                    detection.attendance_time = (datetime.min + detection_offset).time()
                updated = True
        
        if updated:
            updated_count += 1
    
    try:
        db.session.commit()
        logger.info(f"Successfully updated {updated_count} detections with OCR data")
    except Exception as e:
        logger.error(f"Error updating detections: {e}")
        db.session.rollback()
        raise
    
    return updated_count


def handle_person_split_ocr(original_person_id: str, new_person_ids: List[str], 
                          detection_mapping: Dict, db, DetectedPerson):
    """
    Handle OCR data when splitting a person into multiple persons
    
    Args:
        original_person_id: Original person ID (e.g., PERSON-0001)
        new_person_ids: List of new person IDs
        detection_mapping: Mapping of detection IDs to new person IDs
        db: Database session
        DetectedPerson: DetectedPerson model class
    """
    logger.info(f"Handling OCR data split for {original_person_id} -> {new_person_ids}")
    
    # Get all detections for the original person
    detections = DetectedPerson.query.filter_by(person_id=original_person_id).all()
    
    for detection in detections:
        # Check if this detection should be reassigned
        if detection.id in detection_mapping:
            new_person_id = detection_mapping[detection.id]
            detection.person_id = new_person_id
            logger.info(f"Reassigned detection {detection.id} to {new_person_id}")
            # OCR data remains with the detection
    
    db.session.commit()


def handle_person_merge_ocr(person_ids_to_merge: List[str], target_person_id: str,
                          db, DetectedPerson):
    """
    Handle OCR data when merging multiple persons into one
    
    Args:
        person_ids_to_merge: List of person IDs to merge
        target_person_id: Target person ID to merge into
        db: Database session
        DetectedPerson: DetectedPerson model class
    """
    logger.info(f"Handling OCR data merge: {person_ids_to_merge} -> {target_person_id}")
    
    # Update all detections to use the target person ID
    for person_id in person_ids_to_merge:
        if person_id != target_person_id:
            detections = DetectedPerson.query.filter_by(person_id=person_id).all()
            for detection in detections:
                detection.person_id = target_person_id
                # OCR data is preserved with each detection
            logger.info(f"Merged {len(detections)} detections from {person_id} to {target_person_id}")
    
    db.session.commit()