"""Debug logger for person recognition during video processing"""

import logging
import json
from datetime import datetime
from pathlib import Path

class RecognitionDebugLogger:
    """Logger to track recognition decisions during video processing"""
    
    def __init__(self, video_name="unknown"):
        self.video_name = video_name
        self.log_dir = Path("processing/debug_logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Create log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"recognition_debug_{video_name}_{timestamp}.log"
        
        # Setup logger
        self.logger = logging.getLogger(f"recognition_debug_{video_name}")
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        fh = logging.FileHandler(self.log_file)
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        self.logger.info(f"=== Recognition Debug Log Started for {video_name} ===")
        
    def log_model_load(self, model_name, success, error=None):
        """Log model loading attempt"""
        if success:
            self.logger.info(f"[OK] Model loaded successfully: {model_name}")
        else:
            self.logger.error(f"[ERROR] Failed to load model: {model_name} - {error}")
    
    def log_recognition_attempt(self, frame_number, bbox, result=None, error=None):
        """Log recognition attempt"""
        if error:
            self.logger.error(f"Frame {frame_number}: Recognition failed - {error}")
        elif result:
            self.logger.info(f"Frame {frame_number}: Recognition result - {result}")
        else:
            self.logger.info(f"Frame {frame_number}: No recognition result")
    
    def log_person_id_decision(self, frame_number, recognized_id, assigned_id, reason):
        """Log person ID assignment decision"""
        self.logger.info(f"Frame {frame_number}: ID Decision - Recognized: {recognized_id}, Assigned: {assigned_id}, Reason: {reason}")
    
    def log_summary(self):
        """Log summary statistics"""
        self.logger.info("=== Recognition Summary ===")
        self.logger.info(f"Log saved to: {self.log_file}")
        
# Global debug logger instance
_debug_logger = None

def get_debug_logger(video_name=None):
    """Get or create debug logger"""
    global _debug_logger
    if _debug_logger is None and video_name:
        _debug_logger = RecognitionDebugLogger(video_name)
    return _debug_logger

def reset_debug_logger():
    """Reset debug logger for new video"""
    global _debug_logger
    if _debug_logger:
        _debug_logger.log_summary()
    _debug_logger = None