"""
Enhanced Detection Class for chunked video processing
"""
import cv2
import numpy as np
import logging
from ultralytics import YOLO
from typing import List, Tuple

logger = logging.getLogger(__name__)


class EnhancedDetection:
    """Enhanced person detection using YOLO"""
    
    def __init__(self, use_gpu=True, model_size='n'):
        """
        Initialize enhanced detection
        
        Args:
            use_gpu: Whether to use GPU if available
            model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
        """
        self.use_gpu = use_gpu
        self.model_size = model_size
        
        # Load YOLO model
        model_name = f'yolov8{model_size}.pt'
        try:
            self.model = YOLO(model_name)
            
            # Move model to GPU if requested and available
            if self.use_gpu:
                import torch
                if torch.cuda.is_available():
                    self.model.to('cuda')
                    logger.info(f"[OK] YOLO model loaded on GPU")
                else:
                    logger.warning("GPU requested but CUDA not available, using CPU")
                    self.use_gpu = False
            else:
                logger.info(f"YOLO model loaded on CPU")
                
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
            
    def detect_persons(self, frame: np.ndarray) -> Tuple[List[List[int]], List[float]]:
        """
        Detect persons in a frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (boxes, confidences)
            - boxes: List of [x1, y1, x2, y2] coordinates
            - confidences: List of confidence scores
        """
        try:
            # Run YOLO detection
            results = self.model(frame, verbose=False)
            
            boxes = []
            confidences = []
            
            # Extract person detections (class 0 is 'person' in COCO)
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        if int(box.cls[0]) == 0:  # Person class
                            # Get box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf[0])
                            
                            # Convert to integers
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            boxes.append([x1, y1, x2, y2])
                            confidences.append(confidence)
                            
            return boxes, confidences
            
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return [], []
            
    def detect_persons_batch(self, frames: List[np.ndarray]) -> List[Tuple[List[List[int]], List[float]]]:
        """
        Detect persons in multiple frames (batch processing)
        
        Args:
            frames: List of frames
            
        Returns:
            List of (boxes, confidences) tuples for each frame
        """
        results = []
        
        try:
            # Process frames in batch if GPU is available
            if self.use_gpu and len(frames) > 1:
                # YOLO can process multiple images at once
                batch_results = self.model(frames, verbose=False)
                
                for r in batch_results:
                    boxes = []
                    confidences = []
                    
                    if r.boxes is not None:
                        for box in r.boxes:
                            if int(box.cls[0]) == 0:  # Person class
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                confidence = float(box.conf[0])
                                
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                
                                boxes.append([x1, y1, x2, y2])
                                confidences.append(confidence)
                                
                    results.append((boxes, confidences))
            else:
                # Process frames one by one
                for frame in frames:
                    result = self.detect_persons(frame)
                    results.append(result)
                    
        except Exception as e:
            logger.error(f"Error during batch detection: {e}")
            # Return empty results for all frames
            results = [([], [])] * len(frames)
            
        return results