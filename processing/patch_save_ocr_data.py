"""
Patch to ensure OCR data is saved after video processing
This should be integrated into the processing workflow
"""

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def patch_process_video_to_save_ocr():
    """
    Monkey patch the process_video function to save OCR data
    This ensures OCR data extracted during processing is saved to database
    """
    try:
        # Import the standalone tasks module
        from hr_management.processing import standalone_tasks
        
        # Save original function
        original_process_video_enhanced = standalone_tasks.process_video_enhanced
        
        def process_video_enhanced_with_ocr_save(video_id, video_path, app):
            """Enhanced process video that saves OCR data"""
            # Call original function
            result = original_process_video_enhanced(video_id, video_path, app)
            
            # If processing succeeded and we have OCR data, save it
            if result and not result.get('error') and result.get('ocr_data'):
                logger.info(f"Saving OCR data for video {video_id}")
                
                with app.app_context():
                    try:
                        db = app.db
                        Video = app.Video
                        DetectedPerson = app.DetectedPerson
                        
                        # Import the save functions
                        from processing.save_ocr_after_processing import (
                            save_ocr_data_to_video,
                            update_detections_with_ocr_after_processing
                        )
                        
                        # Save OCR data to video
                        ocr_saved = save_ocr_data_to_video(video_id, result['ocr_data'], db, Video)
                        
                        if ocr_saved:
                            # Update detections with OCR data
                            detections_updated = update_detections_with_ocr_after_processing(
                                video_id, result['ocr_data'], db, Video, DetectedPerson
                            )
                            logger.info(f"OCR data saved and {detections_updated} detections updated")
                        
                    except Exception as e:
                        logger.error(f"Error saving OCR data: {e}")
            
            return result
        
        # Replace the function
        standalone_tasks.process_video_enhanced = process_video_enhanced_with_ocr_save
        logger.info("Successfully patched process_video_enhanced to save OCR data")
        
    except Exception as e:
        logger.error(f"Failed to patch process_video_enhanced: {e}")


def ensure_ocr_saved_in_gpu_detection():
    """
    Ensure OCR data is saved when using GPU enhanced detection
    """
    try:
        # Import the module
        import processing.gpu_enhanced_detection as gpu_detection
        
        # Save original function  
        original_process = gpu_detection.process_video_enhanced_gpu
        
        def process_video_enhanced_gpu_with_ocr(video_id, video_path, app=None):
            """GPU processing that saves OCR data"""
            # Call original function
            result = original_process(video_id, video_path, app)
            
            # If we have OCR data in result, save it
            if result and not result.get('error') and result.get('ocr_data'):
                logger.info(f"GPU processing completed, saving OCR data for video {video_id}")
                
                # Import app if not provided
                if app is None:
                    from app import create_app
                    app = create_app()
                
                with app.app_context():
                    try:
                        db = app.db
                        Video = app.Video
                        DetectedPerson = app.DetectedPerson
                        
                        # Save OCR data
                        video = Video.query.get(video_id)
                        if video:
                            ocr_data = result['ocr_data']
                            
                            # Update video with OCR data
                            video.ocr_location = ocr_data.get('location')
                            video.ocr_video_date = ocr_data.get('video_date')
                            
                            # Extract time from timestamps
                            if ocr_data.get('timestamps') and len(ocr_data['timestamps']) > 0:
                                first_timestamp = ocr_data['timestamps'][0]['timestamp']
                                if isinstance(first_timestamp, datetime):
                                    video.ocr_video_time = first_timestamp.time()
                            
                            video.ocr_extraction_done = True
                            extraction_summary = ocr_data.get('extraction_summary', {})
                            video.ocr_extraction_confidence = extraction_summary.get('confidence', 0)
                            
                            db.session.commit()
                            logger.info(f"OCR data saved: location={video.ocr_location}, date={video.ocr_video_date}")
                            
                            # Update detections with OCR data
                            from hr_management.processing.enhanced_save_detections import update_detections_with_ocr
                            updated = update_detections_with_ocr(video_id, ocr_data, db, DetectedPerson)
                            logger.info(f"Updated {updated} detections with OCR data")
                            
                    except Exception as e:
                        logger.error(f"Error saving OCR data in GPU processing: {e}")
                        db.session.rollback()
            
            return result
        
        # Replace the function
        gpu_detection.process_video_enhanced_gpu = process_video_enhanced_gpu_with_ocr
        logger.info("Successfully patched GPU detection to save OCR data")
        
    except Exception as e:
        logger.error(f"Failed to patch GPU detection: {e}")


# Auto-apply patches when module is imported
patch_process_video_to_save_ocr()
ensure_ocr_saved_in_gpu_detection()