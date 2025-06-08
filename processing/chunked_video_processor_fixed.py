"""
Fixed version of chunked video processor with working recognition
This is a simplified version that focuses on making recognition work
"""
import os
import cv2
import numpy as np
import logging
from pathlib import Path
import json

# Import the simple recognition
from processing.simple_recognition_fix import recognize_person, get_recognition_model

logger = logging.getLogger(__name__)

def process_video_with_recognition(video_path, output_dir="static/uploads"):
    """Process video with working recognition"""
    
    logger.info(f"Processing video with recognition: {video_path}")
    
    # Load recognition model
    model = get_recognition_model()
    if not model.loaded:
        logger.error("Recognition model not loaded!")
        return {"error": "Recognition model not available"}
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Cannot open video"}
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Video: {total_frames} frames at {fps} FPS")
    
    # Process frames
    frame_num = 0
    recognized_persons = {}
    new_persons = {}
    next_person_id = _get_next_person_id()
    
    # Process every 10th frame for speed
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_num % 10 == 0:  # Process every 10th frame
            # Simple person detection (center crop for testing)
            h, w = frame.shape[:2]
            
            # Get center region as "person"
            x1 = w // 4
            y1 = h // 4
            x2 = 3 * w // 4
            y2 = 3 * h // 4
            
            person_img = frame[y1:y2, x1:x2]
            
            if person_img.size > 0:
                # Try recognition
                result = recognize_person(person_img, confidence_threshold=0.6)
                
                if result and result['person_id'] != 'unknown':
                    person_id = result['person_id']
                    confidence = result['confidence']
                    
                    if person_id not in recognized_persons:
                        recognized_persons[person_id] = {
                            'count': 0,
                            'confidences': []
                        }
                    
                    recognized_persons[person_id]['count'] += 1
                    recognized_persons[person_id]['confidences'].append(confidence)
                    
                    logger.info(f"Frame {frame_num}: Recognized {person_id} (confidence: {confidence:.2f})")
                else:
                    # New person
                    if frame_num not in new_persons:
                        person_id = f"PERSON-{next_person_id:04d}"
                        next_person_id += 1
                        new_persons[frame_num] = person_id
                        logger.info(f"Frame {frame_num}: New person {person_id}")
                        
        frame_num += 1
        
        if frame_num % 100 == 0:
            logger.info(f"Processed {frame_num}/{total_frames} frames")
    
    cap.release()
    
    # Summary
    logger.info("\n=== Recognition Summary ===")
    logger.info(f"Recognized persons: {len(recognized_persons)}")
    for person_id, data in recognized_persons.items():
        avg_conf = np.mean(data['confidences'])
        logger.info(f"  {person_id}: {data['count']} detections, avg confidence: {avg_conf:.2f}")
    
    logger.info(f"New persons: {len(new_persons)}")
    
    return {
        "success": True,
        "recognized_persons": recognized_persons,
        "new_persons": new_persons,
        "frames_processed": frame_num
    }

def _get_next_person_id():
    """Get next available person ID"""
    persons_dir = Path('processing/outputs/persons')
    if not persons_dir.exists():
        return 1
        
    existing = list(persons_dir.glob('PERSON-*'))
    max_id = 0
    
    for folder in existing:
        try:
            num = int(folder.name.replace('PERSON-', ''))
            max_id = max(max_id, num)
        except:
            pass
            
    return max_id + 1

if __name__ == "__main__":
    # Test with a video
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        videos = list(Path('static/uploads').glob('*.mp4'))
        if videos:
            video_path = str(videos[0])
        else:
            print("No videos found")
            sys.exit(1)
    
    result = process_video_with_recognition(video_path)
    print(json.dumps(result, indent=2))