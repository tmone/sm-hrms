"""
Fixed version of extract_persons_data_gpu that properly uses recognized IDs
"""
import cv2
import json
import tempfile
import os
import uuid
from pathlib import Path
from datetime import datetime

def extract_persons_data_gpu_fixed(video_path, person_tracks, persons_dir, ui_style_recognizer=None):
    """
    Extract person images with recognition BEFORE folder creation
    """
    print(f"[CAMERA] Extracting person data to {persons_dir}")
    
    # Validate and merge potential duplicate tracks
    from .gpu_enhanced_detection import validate_and_merge_tracks
    person_tracks = validate_and_merge_tracks(person_tracks)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video for person extraction: {video_path}")
        return 0
    
    extracted_count = 0
    
    for person_id, detections in person_tracks.items():
        # First, try to recognize this person from first few images
        recognized_person_id = None
        recognition_confidence = 0.0
        
        if ui_style_recognizer and len(detections) > 0:
            print(f"[SEARCH] Attempting recognition for person {person_id}...")
            
            # Try recognition on up to 3 frames for better accuracy
            for i, detection in enumerate(detections[:3]):
                frame_number = detection["frame_number"]
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                
                if ret:
                    x, y, w, h = detection["bbox"]
                    
                    # Skip small images
                    if w < 128:
                        continue
                    
                    # Extract person region
                    padding = 10
                    x1 = max(0, int(x - padding))
                    y1 = max(0, int(y - padding))
                    x2 = min(frame.shape[1], int(x + w + padding))
                    y2 = min(frame.shape[0], int(y + h + padding))
                    
                    person_img = frame[y1:y2, x1:x2]
                    
                    if person_img.size > 0:
                        try:
                            # Check if it's venv wrapper or direct recognizer
                            if hasattr(ui_style_recognizer, 'recognize_person'):
                                # VenvRecognitionWrapper
                                result = ui_style_recognizer.recognize_person(person_img, 0.8)
                                if result:
                                    recognized_person_id = result['person_id']
                                    recognition_confidence = result['confidence']
                            else:
                                # Direct recognizer
                                temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                                cv2.imwrite(temp_file.name, person_img)
                                temp_file.close()
                                
                                result = ui_style_recognizer.process_cropped_image(temp_file.name)
                                os.unlink(temp_file.name)
                                
                                if result and result.get('persons'):
                                    first_person = result['persons'][0]
                                    if first_person['person_id'] != 'unknown' and first_person['confidence'] >= 0.8:
                                        recognized_person_id = first_person['person_id']
                                        recognition_confidence = first_person['confidence']
                            
                            if recognized_person_id:
                                print(f"[OK] Recognized as {recognized_person_id} with {recognition_confidence:.2%} confidence")
                                break
                                
                        except Exception as e:
                            print(f"[WARNING] Recognition error: {e}")
        
        # Now determine the final person_id to use
        if recognized_person_id:
            # Use recognized ID
            final_person_id = recognized_person_id
            print(f"[TARGET] Using recognized ID: {final_person_id} (was {person_id})")
        else:
            # Use original ID
            if isinstance(person_id, int):
                final_person_id = f"PERSON-{person_id:04d}"
            else:
                final_person_id = str(person_id)
            print(f"[NEW] Using new ID: {final_person_id}")
        
        # Create folder with correct ID
        person_dir = persons_dir / final_person_id
        person_dir.mkdir(parents=True, exist_ok=True)
        
        # Rest of extraction logic...
        # Sample detections
        FRAME_SAMPLE_INTERVAL = 5
        if len(detections) <= 30:
            sample_detections = detections
        else:
            sample_detections = detections[::FRAME_SAMPLE_INTERVAL]
            if detections[0] not in sample_detections:
                sample_detections.insert(0, detections[0])
            if detections[-1] not in sample_detections:
                sample_detections.append(detections[-1])
        
        person_metadata = {
            "person_id": final_person_id,
            "original_tracking_id": person_id,
            "recognized": recognized_person_id is not None,
            "recognition_confidence": float(recognition_confidence) if recognized_person_id else 0,
            "total_detections": len(detections),
            "first_appearance": detections[0]["timestamp"],
            "last_appearance": detections[-1]["timestamp"],
            "avg_confidence": sum(d["confidence"] for d in detections) / len(detections),
            "images": [],
            "created_at": datetime.now().isoformat()
        }
        
        # Extract images
        for detection in sample_detections:
            frame_number = detection["frame_number"]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret:
                x, y, w, h = detection["bbox"]
                
                if w < 128:
                    continue
                
                # Extract person region
                padding = 10
                x1 = max(0, int(x - padding))
                y1 = max(0, int(y - padding))
                x2 = min(frame.shape[1], int(x + w + padding))
                y2 = min(frame.shape[0], int(y + h + padding))
                
                person_img = frame[y1:y2, x1:x2]
                
                if person_img.size > 0:
                    img_filename = f"{uuid.uuid4()}.jpg"
                    img_path = person_dir / img_filename
                    cv2.imwrite(str(img_path), person_img)
                    
                    person_metadata["images"].append({
                        "filename": img_filename,
                        "frame_number": frame_number,
                        "timestamp": detection["timestamp"],
                        "confidence": detection["confidence"],
                        "bbox": detection["bbox"]
                    })
        
        # Save metadata
        if len(person_metadata["images"]) > 0:
            metadata_path = person_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(person_metadata, f, indent=2)
            
            extracted_count += 1
            print(f"[OK] Created {final_person_id} folder with {len(person_metadata['images'])} images")
    
    cap.release()
    print(f"[CAMERA] Extracted {extracted_count} persons with valid images")
    return extracted_count