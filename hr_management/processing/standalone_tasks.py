"""
Standalone processing tasks that don't require Celery
These functions can be used for fallback processing when Celery is not available
"""
import os
import json
from datetime import datetime

# Optional CV2 import
try:
    import cv2
    CV2_AVAILABLE = True
    print("✅ OpenCV available for video processing")
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️ OpenCV not available. Video processing will use mock data.")

# Check for SAM/SAM2 availability
try:
    import torch
    TORCH_AVAILABLE = True
    print("✅ PyTorch available for AI models")
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available. AI model features disabled.")

# Display system capabilities
print(f"🖥️ System capabilities: CV2={CV2_AVAILABLE}, PyTorch={TORCH_AVAILABLE}")

def extract_video_metadata(filepath):
    """Extract metadata from video file using real backends"""
    print(f"📊 Extracting metadata from: {filepath}")
    
    # Use real detection module for metadata extraction
    try:
        from .real_detection import extract_video_metadata_real
        return extract_video_metadata_real(filepath)
    except ImportError:
        print("⚠️ Real detection module not available, using fallback")
    
    if not CV2_AVAILABLE:
        # Return mock metadata when OpenCV is not available
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
        else:
            file_size = 0
            
        print("⚠️ OpenCV not available, using mock metadata")
        return {
            'duration': 60.0,  # Mock 1 minute duration
            'fps': 25.0,
            'frame_count': 1500,
            'resolution': "1920x1080",
            'width': 1920,
            'height': 1080,
            'file_size': file_size
        }
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Video file not found: {filepath}")
    
    cap = cv2.VideoCapture(filepath)
    
    if not cap.isOpened():
        raise Exception(f"Could not open video file: {filepath}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    metadata = {
        'duration': duration,
        'fps': fps,
        'frame_count': frame_count,
        'resolution': f"{width}x{height}",
        'width': width,
        'height': height,
        'file_size': os.path.getsize(filepath)
    }
    
    print(f"📊 Extracted metadata: {metadata}")
    return metadata

def detect_persons_in_video(filepath):
    """Detect persons in video using real AI models or enhanced fallback"""
    print(f"👥 Starting person detection for: {filepath}")
    
    # Try to use state-of-the-art AI models first
    try:
        # First try transformer and SAM models
        from .transformer_detection import detect_persons_with_best_model, get_best_available_detector
        detector = get_best_available_detector()
        print(f"🤖 Attempting to use {detector.upper()} for person detection")
        
        print(f"🚀 Using REAL AI model: {detector}")
        return detect_persons_with_best_model(filepath)
        
    except ImportError as e:
        print(f"⚠️ Transformer/SAM models not available: {e}")
    except Exception as e:
        print(f"⚠️ Transformer/SAM detection failed: {e}")
    
    # Fallback to traditional models
    try:
        from .real_detection import detect_persons_real, get_best_available_detector as get_fallback_detector
        detector = get_fallback_detector()
        print(f"🔄 Falling back to {detector.upper()} detection")
        
        if detector != "mock":
            print(f"🚀 Using fallback AI model: {detector}")
            return detect_persons_real(filepath)
        else:
            print("⚠️ No AI models available")
    except ImportError as e:
        print(f"⚠️ Fallback detection not available: {e}")
    except Exception as e:
        print(f"⚠️ Fallback detection failed: {e}")
    
    # Only use enhanced mock if no real models available
    raise RuntimeError("❌ NO REAL AI MODELS AVAILABLE! Please install: pip install torch transformers segment-anything ultralytics")
    
    # Enhanced fallback detection
    if not CV2_AVAILABLE:
        print("⚠️ OpenCV not available, creating ENHANCED mock person detections...")
        # Return enhanced mock detections when OpenCV is not available
        detections = []
        
        # Scale number of detections based on file size (more realistic)
        file_size = os.path.getsize(filepath) if os.path.exists(filepath) else 1000000
        num_persons = min(15, max(3, int(file_size / 100000000)))  # 3-15 persons based on file size
        
        print(f"📊 File size: {file_size} bytes -> generating {num_persons} mock detections")
        
        for i in range(num_persons):
            person_code = f"PERSON-{i+1:04d}"
            start_time = i * 4.0 + (i % 3) * 2.0  # Varied timing
            
            detection = {
                'person_code': person_code,
                'start_frame': int(i * 100 + (i % 4) * 25),
                'end_frame': int(i * 100 + (i % 4) * 25) + 150,
                'start_time': start_time,
                'end_time': start_time + 5.0 + (i % 3),
                'confidence': 0.75 + (i % 4) * 0.05,  # Varied confidence 0.75-0.90
                'bbox_data': [{
                    'frame': int(i * 100 + (i % 4) * 25),
                    'x': 15 + (i % 5) * 15,  # Varied X positions 15-75%
                    'y': 25 + (i % 4) * 12,  # Varied Y positions 25-61%
                    'width': 12 + (i % 3) * 4,  # Varied widths 12-20%
                    'height': 35 + (i % 4) * 8   # Varied heights 35-59%
                }]
            }
            detections.append(detection)
            print(f"👤 Enhanced mock detected person {person_code} at {start_time:.1f}s (conf: {detection['confidence']:.2f})")
        
        print(f"🎯 Enhanced mock person detection completed: {len(detections)} persons found")
        return detections
    
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Video file not found: {filepath}")
    
    cap = cv2.VideoCapture(filepath)
    
    if not cap.isOpened():
        raise Exception(f"Could not open video file: {filepath}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Video properties: FPS={fps}, Total frames={total_frames}")
    
    detections = []
    person_counter = 1
    frame_number = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Log progress every 100 frames
        if frame_number % 100 == 0:
            progress = (frame_number / min(total_frames, 300)) * 100
            print(f"🔄 Processing frame {frame_number}/{min(total_frames, 300)} ({progress:.1f}%)")
        
        # Simulate person detection (replace with actual detection logic)
        # This would use YOLO, SAM, or other detection models
        
        # For demo purposes, create some fake detections
        if frame_number % 30 == 0:  # Every 30 frames
            person_code = f"PERSON-{person_counter:04d}"
            start_frame = frame_number
            start_time = frame_number / fps
            
            # Simulate person appearing for 5 seconds
            end_frame = start_frame + int(5 * fps)
            end_time = start_time + 5
            
            detection = {
                'person_code': person_code,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'start_time': start_time,
                'end_time': end_time,
                'confidence': 0.85,
                'bbox_data': [{
                    'frame': start_frame,
                    'x': 20,  # Percentage
                    'y': 30,
                    'width': 15,
                    'height': 40
                }]
            }
            
            detections.append(detection)
            print(f"👤 Detected person {person_code} at frame {start_frame} ({start_time:.1f}s)")
            
            person_counter += 1
        
        frame_number += 1
        
        # Limit processing for demo
        if frame_number > 300:  # Process only first 300 frames
            print(f"⏹️ Stopping processing at frame {frame_number} (demo limit)")
            break
    
    cap.release()
    print(f"🎯 Person detection completed: {len(detections)} persons found")
    return detections

def detect_persons_with_sam(filepath):
    """Detect persons using SAM (Segment Anything Model) - Future implementation"""
    print(f"🤖 SAM-based person detection for: {filepath}")
    
    if not TORCH_AVAILABLE:
        print("⚠️ PyTorch not available, falling back to basic detection")
        return detect_persons_in_video(filepath)
    
    # TODO: Implement actual SAM/SAM2 person detection
    # Example integration points:
    # 1. Load SAM model: model = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
    # 2. Process video frames with SAM
    # 3. Filter segments for person class
    # 4. Track persons across frames
    # 5. Generate detection data
    
    print("🚧 SAM integration not yet implemented, using fallback")
    return detect_persons_in_video(filepath)

def save_detections_to_db(video_id, detections, fps, db=None, DetectedPerson=None):
    """Save detection results to database"""
    print(f"💾 Saving {len(detections)} detections to database for video {video_id}")
    
    if not db or not DetectedPerson:
        print("⚠️ Database connection not available, skipping save")
        return
    
    try:
        # Convert our detection format to the actual model format
        for i, detection in enumerate(detections):
            # Create multiple detection records for each person (one per key frame)
            start_frame = detection['start_frame']
            end_frame = detection['end_frame']
            start_time = detection['start_time']
            end_time = detection['end_time']
            
            # Sample key frames (every 30 frames or 1 second)
            frame_interval = max(1, int(fps))  # Sample every second
            
            for frame_num in range(start_frame, min(end_frame, start_frame + 150), frame_interval):
                timestamp = frame_num / fps if fps > 0 else 0
                
                # Get bounding box for this frame (use first bbox if available)
                bbox = detection['bbox_data'][0] if detection['bbox_data'] else {}
                
                # Convert all values to standard Python types to avoid numpy/binary serialization issues
                bbox_x = float(bbox.get('x', 0)) if bbox.get('x') is not None else 0.0
                bbox_y = float(bbox.get('y', 0)) if bbox.get('y') is not None else 0.0
                bbox_width = float(bbox.get('width', 0)) if bbox.get('width') is not None else 0.0
                bbox_height = float(bbox.get('height', 0)) if bbox.get('height') is not None else 0.0
                confidence = float(detection['confidence']) if detection['confidence'] is not None else 0.0
                timestamp = float(timestamp) if timestamp is not None else 0.0
                
                print(f"   🔢 Converting bbox: x={bbox_x}, y={bbox_y}, w={bbox_width}, h={bbox_height}")
                
                db_detection = DetectedPerson(
                    video_id=video_id,
                    timestamp=timestamp,
                    frame_number=frame_num,
                    confidence=confidence,
                    bbox_x=int(round(bbox_x)),
                    bbox_y=int(round(bbox_y)), 
                    bbox_width=int(round(bbox_width)),
                    bbox_height=int(round(bbox_height)),
                    is_identified=False
                )
                
                db.session.add(db_detection)
                
            print(f"   💿 Queued detections for person {detection['person_code']} ({start_frame}-{end_frame})")
        
        db.session.commit()
        print(f"✅ Successfully saved detection frames to database")
        
    except Exception as e:
        print(f"❌ Error saving detections to database: {e}")
        if db:
            db.session.rollback()
        raise

def extract_faces_from_detection(video_filepath, detection, dataset_path):
    """Extract face images from a specific detection"""
    print(f"👤 Extracting faces for {detection.person_code} from {video_filepath}")
    
    if not CV2_AVAILABLE:
        print("⚠️ OpenCV not available, skipping face extraction")
        return 0
    
    # Create person directory
    person_dir = os.path.join(dataset_path, detection.person_code)
    os.makedirs(person_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_filepath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    faces_extracted = 0
    
    # Jump to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, detection.start_frame)
    
    for frame_num in range(detection.start_frame, detection.end_frame, int(fps)):  # Extract every second
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Extract face from frame (simplified)
        height, width = frame.shape[:2]
        
        # Use bbox data if available
        if detection.bbox_data:
            bbox = detection.bbox_data[0]  # Use first bbox
            x = int(bbox['x'] / 100 * width)
            y = int(bbox['y'] / 100 * height)
            w = int(bbox['width'] / 100 * width)
            h = int(bbox['height'] / 100 * height)
            
            face_crop = frame[y:y+h, x:x+w]
        else:
            # Fallback to center crop
            crop_size = min(width, height) // 4
            start_x = (width - crop_size) // 2
            start_y = (height - crop_size) // 2
            face_crop = frame[start_y:start_y+crop_size, start_x:start_x+crop_size]
        
        # Resize to standard face size
        face_resized = cv2.resize(face_crop, (128, 128))
        
        # Save face image
        face_filename = f"face_{detection.person_code}_frame_{frame_num:06d}.jpg"
        face_path = os.path.join(person_dir, face_filename)
        cv2.imwrite(face_path, face_resized)
        
        faces_extracted += 1
    
    cap.release()
    print(f"✅ Extracted {faces_extracted} face images for {detection.person_code}")
    return faces_extracted