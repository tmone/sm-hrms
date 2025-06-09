"""
Save OCR data after video processing
This module ensures OCR data extracted during processing is saved to the database
"""

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def save_ocr_data_to_video(video_id, ocr_data, db, Video):
    """
    Save OCR data to video record after processing
    
    Args:
        video_id: ID of the video
        ocr_data: OCR data extracted during processing
        db: Database session
        Video: Video model class
    """
    if not ocr_data:
        logger.info(f"No OCR data to save for video {video_id}")
        return False
        
    try:
        video = Video.query.get(video_id)
        if not video:
            logger.error(f"Video {video_id} not found")
            return False
            
        # Update video with OCR data
        video.ocr_location = ocr_data.get('location')
        video.ocr_video_date = ocr_data.get('video_date')
        
        # Extract time from the timestamps if available
        if ocr_data.get('timestamps') and len(ocr_data['timestamps']) > 0:
            # Get the first valid timestamp
            first_timestamp = ocr_data['timestamps'][0]['timestamp']
            if isinstance(first_timestamp, datetime):
                video.ocr_video_time = first_timestamp.time()
            else:
                logger.warning(f"Invalid timestamp format for video {video_id}")
        
        # Mark OCR extraction as done
        video.ocr_extraction_done = True
        
        # Save confidence score if available
        extraction_summary = ocr_data.get('extraction_summary', {})
        video.ocr_extraction_confidence = extraction_summary.get('confidence', 0)
        
        db.session.commit()
        
        logger.info(f"OCR data saved for video {video_id}:")
        logger.info(f"  - Location: {video.ocr_location}")
        logger.info(f"  - Date: {video.ocr_video_date}")
        logger.info(f"  - Time: {video.ocr_video_time}")
        logger.info(f"  - Confidence: {video.ocr_extraction_confidence:.2%}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving OCR data for video {video_id}: {e}")
        db.session.rollback()
        return False


def update_detections_with_ocr_after_processing(video_id, ocr_data, db, Video, DetectedPerson):
    """
    Update detections with OCR data after processing
    
    Args:
        video_id: ID of the video  
        ocr_data: OCR data extracted during processing
        db: Database session
        Video: Video model class
        DetectedPerson: DetectedPerson model class
    """
    if not ocr_data:
        logger.info(f"No OCR data to update detections for video {video_id}")
        return 0
        
    try:
        video = Video.query.get(video_id)
        if not video:
            logger.error(f"Video {video_id} not found")
            return 0
            
        # Get all detections for this video
        detections = DetectedPerson.query.filter_by(video_id=video_id).all()
        
        if not detections:
            logger.warning(f"No detections found for video {video_id}")
            return 0
            
        updated_count = 0
        
        # Extract OCR values
        ocr_location = ocr_data.get('location')
        ocr_video_date = ocr_data.get('video_date')
        ocr_video_time = None
        
        if ocr_data.get('timestamps') and len(ocr_data['timestamps']) > 0:
            first_timestamp = ocr_data['timestamps'][0]['timestamp']
            if isinstance(first_timestamp, datetime):
                ocr_video_time = first_timestamp.time()
        
        # Update each detection
        for detection in detections:
            # Update attendance location
            if ocr_location and not detection.attendance_location:
                detection.attendance_location = ocr_location
                updated_count += 1
            
            # Update attendance date and time
            if ocr_video_date and not detection.attendance_date:
                detection.attendance_date = ocr_video_date
                
                # Calculate attendance time based on detection timestamp
                if detection.timestamp is not None and ocr_video_time:
                    from datetime import timedelta
                    # Combine date and time
                    base_datetime = datetime.combine(ocr_video_date, ocr_video_time)
                    detection_offset = timedelta(seconds=float(detection.timestamp))
                    attendance_datetime = base_datetime + detection_offset
                    
                    detection.attendance_time = attendance_datetime.time()
                    detection.check_in_time = attendance_datetime
                
                updated_count += 1
        
        db.session.commit()
        
        logger.info(f"Updated {updated_count} detections with OCR data for video {video_id}")
        return updated_count
        
    except Exception as e:
        logger.error(f"Error updating detections with OCR data for video {video_id}: {e}")
        db.session.rollback()
        return 0