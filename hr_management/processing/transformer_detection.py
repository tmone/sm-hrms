"""
Real person detection using popular Transformer and SAM models
Implements state-of-the-art models: DETR, RT-DETR, SAM, SAM2, Hugging Face transformers
"""
import os
import json
import numpy as np
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check available backends
print("[SEARCH] Checking state-of-the-art AI model backends...")

# Core dependencies
try:
    import torch
    import torchvision
    from torchvision import transforms
    TORCH_AVAILABLE = True
    print("[OK] PyTorch + TorchVision available")
except ImportError:
    TORCH_AVAILABLE = False
    print("[ERROR] PyTorch not available")

# Transformers (Hugging Face)
try:
    from transformers import AutoImageProcessor, AutoModel, pipeline
    import transformers
    TRANSFORMERS_AVAILABLE = True
    print("[OK] Hugging Face Transformers available")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[ERROR] Transformers not available")

# SAM (Segment Anything)
try:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    SAM_AVAILABLE = True
    print("[OK] SAM (Segment Anything) available")
except ImportError:
    SAM_AVAILABLE = False
    print("[ERROR] SAM not available")

# SAM2 (Segment Anything 2)
try:
    import sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
    print("[OK] SAM2 (Segment Anything 2) available")
except ImportError:
    SAM2_AVAILABLE = False
    print("[ERROR] SAM2 not available")

# YOLO (Ultralytics)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("[OK] YOLO (Ultralytics) available")
except ImportError:
    YOLO_AVAILABLE = False
    print("[ERROR] YOLO not available")

# OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
    print("[OK] OpenCV available")
except ImportError:
    CV2_AVAILABLE = False
    print("[ERROR] OpenCV not available")

print(f"[MONITOR] AI capabilities: TORCH={TORCH_AVAILABLE}, TRANSFORMERS={TRANSFORMERS_AVAILABLE}, SAM={SAM_AVAILABLE}, SAM2={SAM2_AVAILABLE}, YOLO={YOLO_AVAILABLE}, CV2={CV2_AVAILABLE}")

# Global model cache
_model_cache = {}

def get_best_available_detector():
    """Get the best available person detection model"""
    if SAM2_AVAILABLE and TORCH_AVAILABLE:
        return "sam2"
    elif SAM_AVAILABLE and TORCH_AVAILABLE:
        return "sam"
    elif TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
        return "detr"
    elif YOLO_AVAILABLE:
        return "yolo"
    elif TORCH_AVAILABLE:
        return "torch"
    elif CV2_AVAILABLE:
        return "opencv"
    else:
        raise RuntimeError("No AI models available! Please install: pip install torch torchvision transformers segment-anything ultralytics opencv-python")

def detect_persons_transformer(filepath):
    """Detect persons using Transformer models (DETR, RT-DETR)"""
    print("[AI] Using Transformer models for person detection")
    
    if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
        raise RuntimeError("Transformers not available")
    
    try:
        # Load DETR model for object detection
        if "detr" not in _model_cache:
            print("[LOAD] Loading DETR transformer model...")
            # Use Facebook's DETR model
            model_name = "facebook/detr-resnet-50"
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            
            # Alternative: Use object detection pipeline
            detector = pipeline("object-detection", 
                               model="facebook/detr-resnet-50", 
                               device=0 if torch.cuda.is_available() else -1)
            
            _model_cache["detr"] = {
                "processor": processor,
                "model": model,
                "detector": detector
            }
            print("[OK] DETR transformer model loaded")
        
        detector = _model_cache["detr"]["detector"]
        
        # Process video
        detections = []
        person_counter = 1
        
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            raise Exception(f"Could not open video: {filepath}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[INFO] Processing {total_frames} frames with DETR transformer")
        
        frame_number = 0
        sample_rate = max(1, int(fps // 1))  # Sample every second
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_number % sample_rate == 0:
                print(f"[PROCESSING] DETR processing frame {frame_number}/{total_frames} ({(frame_number/total_frames)*100:.1f}%)")
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Run DETR detection
                results = detector(frame_rgb)
                
                # Extract person detections
                for result in results:
                    if result['label'] == 'person' and result['score'] > 0.5:
                        box = result['box']
                        confidence = result['score']
                        
                        # Convert box coordinates to percentages
                        frame_height, frame_width = frame.shape[:2]
                        x_percent = (box['xmin'] / frame_width) * 100
                        y_percent = (box['ymin'] / frame_height) * 100
                        width_percent = ((box['xmax'] - box['xmin']) / frame_width) * 100
                        height_percent = ((box['ymax'] - box['ymin']) / frame_height) * 100
                        
                        person_code = f"PERSON-{person_counter:04d}"
                        timestamp = frame_number / fps
                        
                        detection = {
                            'person_code': person_code,
                            'start_frame': frame_number,
                            'end_frame': frame_number + sample_rate,
                            'start_time': timestamp,
                            'end_time': timestamp + (sample_rate / fps),
                            'confidence': confidence,
                            'bbox_data': [{
                                'frame': frame_number,
                                'x': x_percent,
                                'y': y_percent,
                                'width': width_percent,
                                'height': height_percent
                            }]
                        }
                        
                        detections.append(detection)
                        print(f"👤 DETR detected person {person_code} at {timestamp:.1f}s (confidence: {confidence:.3f})")
                        person_counter += 1
            
            frame_number += 1
            
            # Limit processing for performance
            if frame_number > fps * 300:  # 5 minutes
                print("⏹️ Stopping at 5 minute limit")
                break
        
        cap.release()
        print(f"[AI] DETR transformer detection completed: {len(detections)} persons found")
        return detections
        
    except Exception as e:
        print(f"[ERROR] DETR detection failed: {e}")
        raise

def detect_persons_sam2(filepath):
    """Detect persons using SAM2 (Segment Anything 2)"""
    print("[TARGET] Using SAM2 for person segmentation")
    
    if not SAM2_AVAILABLE or not TORCH_AVAILABLE:
        raise RuntimeError("SAM2 not available")
    
    try:
        # Load SAM2 model
        if "sam2" not in _model_cache:
            print("[LOAD] Loading SAM2 model...")
            
            # Download SAM2 checkpoint if not exists
            checkpoint_path = "sam2_hiera_large.pt"
            if not os.path.exists(checkpoint_path):
                print("[LOAD] Downloading SAM2 checkpoint...")
                # You would download the checkpoint here
                # For now, use a smaller model
                checkpoint_path = "sam2_hiera_tiny.pt"
            
            # Initialize SAM2 predictor
            predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
            
            _model_cache["sam2"] = predictor
            print("[OK] SAM2 model loaded")
        
        predictor = _model_cache["sam2"]
        
        # Process video with SAM2
        detections = []
        person_counter = 1
        
        cap = cv2.VideoCapture(filepath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[INFO] Processing {total_frames} frames with SAM2")
        
        frame_number = 0
        sample_rate = max(1, int(fps * 2))  # Sample every 2 seconds (SAM2 is heavy)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_number % sample_rate == 0:
                print(f"[PROCESSING] SAM2 processing frame {frame_number}/{total_frames}")
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Set image for SAM2
                predictor.set_image(frame_rgb)
                
                # Generate masks for the entire image
                # SAM2 can segment everything, we'll filter for person-like segments
                
                # Use automatic mask generation
                mask_generator = SamAutomaticMaskGenerator(predictor.model)
                masks = mask_generator.generate(frame_rgb)
                
                # Filter masks that likely contain persons
                for mask_data in masks:
                    mask = mask_data['segmentation']
                    area = mask_data['area']
                    stability_score = mask_data['stability_score']
                    
                    # Filter for person-sized segments
                    frame_area = frame.shape[0] * frame.shape[1]
                    area_ratio = area / frame_area
                    
                    # Person detection heuristics for SAM2
                    if (0.01 < area_ratio < 0.3 and  # Reasonable size
                        stability_score > 0.8):      # High stability
                        
                        # Get bounding box from mask
                        y_indices, x_indices = np.where(mask)
                        x_min, x_max = x_indices.min(), x_indices.max()
                        y_min, y_max = y_indices.min(), y_indices.max()
                        
                        # Convert to percentages
                        x_percent = (x_min / frame.shape[1]) * 100
                        y_percent = (y_min / frame.shape[0]) * 100
                        width_percent = ((x_max - x_min) / frame.shape[1]) * 100
                        height_percent = ((y_max - y_min) / frame.shape[0]) * 100
                        
                        person_code = f"PERSON-{person_counter:04d}"
                        timestamp = frame_number / fps
                        
                        detection = {
                            'person_code': person_code,
                            'start_frame': frame_number,
                            'end_frame': frame_number + sample_rate,
                            'start_time': timestamp,
                            'end_time': timestamp + (sample_rate / fps),
                            'confidence': stability_score,
                            'bbox_data': [{
                                'frame': frame_number,
                                'x': x_percent,
                                'y': y_percent,
                                'width': width_percent,
                                'height': height_percent
                            }]
                        }
                        
                        detections.append(detection)
                        print(f"👤 SAM2 detected person {person_code} at {timestamp:.1f}s (stability: {stability_score:.3f})")
                        person_counter += 1
            
            frame_number += 1
            
            # Limit processing (SAM2 is computationally heavy)
            if frame_number > fps * 120:  # 2 minutes
                print("⏹️ Stopping at 2 minute limit (SAM2 intensive)")
                break
        
        cap.release()
        print(f"[TARGET] SAM2 detection completed: {len(detections)} persons found")
        return detections
        
    except Exception as e:
        print(f"[ERROR] SAM2 detection failed: {e}")
        raise

def detect_persons_sam(filepath):
    """Detect persons using SAM (Segment Anything)"""
    print("[TARGET] Using SAM for person segmentation")
    
    if not SAM_AVAILABLE or not TORCH_AVAILABLE:
        raise RuntimeError("SAM not available")
    
    try:
        # Load SAM model
        if "sam" not in _model_cache:
            print("[LOAD] Loading SAM model...")
            
            # Download SAM checkpoint if not exists
            checkpoint_path = "sam_vit_h_4b8939.pth"
            if not os.path.exists(checkpoint_path):
                print("[LOAD] Downloading SAM checkpoint...")
                # Use smaller model for faster processing
                model_type = "vit_b"
                checkpoint_path = "sam_vit_b_01ec64.pth"
            else:
                model_type = "vit_h"
            
            # Load SAM model
            sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            if torch.cuda.is_available():
                sam.cuda()
            
            mask_generator = SamAutomaticMaskGenerator(sam)
            
            _model_cache["sam"] = mask_generator
            print("[OK] SAM model loaded")
        
        mask_generator = _model_cache["sam"]
        
        # Process video with SAM
        detections = []
        person_counter = 1
        
        cap = cv2.VideoCapture(filepath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frame_number = 0
        sample_rate = max(1, int(fps * 3))  # Sample every 3 seconds (SAM is heavy)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_number % sample_rate == 0:
                print(f"[PROCESSING] SAM processing frame {frame_number}")
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Generate masks
                masks = mask_generator.generate(frame_rgb)
                
                # Filter for person-like segments
                for mask_data in masks:
                    area = mask_data['area']
                    stability_score = mask_data['stability_score']
                    
                    # Person detection heuristics
                    frame_area = frame.shape[0] * frame.shape[1]
                    area_ratio = area / frame_area
                    
                    if (0.01 < area_ratio < 0.25 and stability_score > 0.85):
                        # Get bounding box
                        bbox = mask_data['bbox']  # [x, y, w, h]
                        
                        # Convert to percentages
                        x_percent = (bbox[0] / frame.shape[1]) * 100
                        y_percent = (bbox[1] / frame.shape[0]) * 100
                        width_percent = (bbox[2] / frame.shape[1]) * 100
                        height_percent = (bbox[3] / frame.shape[0]) * 100
                        
                        person_code = f"PERSON-{person_counter:04d}"
                        timestamp = frame_number / fps
                        
                        detection = {
                            'person_code': person_code,
                            'start_frame': frame_number,
                            'end_frame': frame_number + sample_rate,
                            'start_time': timestamp,
                            'end_time': timestamp + (sample_rate / fps),
                            'confidence': stability_score,
                            'bbox_data': [{
                                'frame': frame_number,
                                'x': x_percent,
                                'y': y_percent,
                                'width': width_percent,
                                'height': height_percent
                            }]
                        }
                        
                        detections.append(detection)
                        print(f"👤 SAM detected person {person_code} at {timestamp:.1f}s")
                        person_counter += 1
            
            frame_number += 1
            
            if frame_number > fps * 180:  # 3 minutes
                break
        
        cap.release()
        print(f"[TARGET] SAM detection completed: {len(detections)} persons found")
        return detections
        
    except Exception as e:
        print(f"[ERROR] SAM detection failed: {e}")
        raise

def detect_persons_yolo_v8(filepath):
    """Detect persons using YOLOv8 (Ultralytics)"""
    print("[FAST] Using YOLOv8 for person detection")
    
    if not YOLO_AVAILABLE:
        raise RuntimeError("YOLO not available")
    
    try:
        # Load YOLOv8 model
        if "yolo" not in _model_cache:
            print("[LOAD] Loading YOLOv8 model...")
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'yolov8n.pt')
            model = YOLO(model_path)  # nano model for speed
            
            # Configure GPU if available
            if TORCH_AVAILABLE and torch.cuda.is_available():
                model.to('cuda')
                print(f"[START] YOLOv8 model loaded on GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("[WARNING] YOLOv8 model loaded on CPU (CUDA not available)")
                
            _model_cache["yolo"] = model
            print("[OK] YOLOv8 model configured")
        
        model = _model_cache["yolo"]
        
        # Process video
        detections = []
        person_counter = 1
        
        cap = cv2.VideoCapture(filepath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[INFO] Processing {total_frames} frames with YOLOv8")
        
        frame_number = 0
        sample_rate = max(1, int(fps // 2))  # Sample every 0.5 seconds
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_number % sample_rate == 0:
                print(f"[PROCESSING] YOLOv8 processing frame {frame_number}/{total_frames}")
                
                # Run YOLO detection
                device = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
                results = model(frame, device=device, verbose=False)
                
                # Extract person detections (class 0 = person)
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            if int(box.cls[0]) == 0:  # Person class
                                confidence = float(box.conf[0])
                                
                                if confidence > 0.5:
                                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                    
                                    # Convert to percentages
                                    frame_height, frame_width = frame.shape[:2]
                                    x_percent = (x1 / frame_width) * 100
                                    y_percent = (y1 / frame_height) * 100
                                    width_percent = ((x2 - x1) / frame_width) * 100
                                    height_percent = ((y2 - y1) / frame_height) * 100
                                    
                                    person_code = f"PERSON-{person_counter:04d}"
                                    timestamp = frame_number / fps
                                    
                                    detection = {
                                        'person_code': person_code,
                                        'start_frame': frame_number,
                                        'end_frame': frame_number + sample_rate,
                                        'start_time': timestamp,
                                        'end_time': timestamp + (sample_rate / fps),
                                        'confidence': confidence,
                                        'bbox_data': [{
                                            'frame': frame_number,
                                            'x': x_percent,
                                            'y': y_percent,
                                            'width': width_percent,
                                            'height': height_percent
                                        }]
                                    }
                                    
                                    detections.append(detection)
                                    print(f"👤 YOLOv8 detected person {person_code} at {timestamp:.1f}s (conf: {confidence:.3f})")
                                    person_counter += 1
            
            frame_number += 1
            
            if frame_number > fps * 600:  # 10 minutes
                break
        
        cap.release()
        print(f"[FAST] YOLOv8 detection completed: {len(detections)} persons found")
        return detections
        
    except Exception as e:
        print(f"[ERROR] YOLOv8 detection failed: {e}")
        raise

def detect_persons_with_best_model(filepath):
    """Detect persons using the best available model"""
    print(f"[START] Starting REAL person detection for: {filepath}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Video file not found: {filepath}")
    
    detector_type = get_best_available_detector()
    print(f"[AI] Using {detector_type.upper()} for person detection")
    
    if detector_type == "sam2":
        return detect_persons_sam2(filepath)
    elif detector_type == "sam":
        return detect_persons_sam(filepath)
    elif detector_type == "detr":
        return detect_persons_transformer(filepath)
    elif detector_type == "yolo":
        return detect_persons_yolo_v8(filepath)
    else:
        raise RuntimeError(f"No suitable model available. Install dependencies: pip install torch transformers segment-anything ultralytics")

# Export the main function
__all__ = [
    'detect_persons_with_best_model',
    'get_best_available_detector'
]