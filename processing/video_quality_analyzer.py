"""
Video Quality Analyzer - Analyzes frame quality for processing decisions
"""
import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class VideoQualityAnalyzer:
    """Analyzes video frame quality metrics"""
    
    def __init__(self):
        self.blur_threshold = 100.0  # Variance threshold for blur detection
        
    def check_frame_quality(self, frame: np.ndarray) -> Dict:
        """
        Check the quality of a frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Dictionary with quality metrics
        """
        quality_info = {
            'blur_score': 100.0,  # 0-100, higher is better (less blurry)
            'brightness': 128,     # Average brightness
            'contrast': 50,        # Contrast score
            'is_valid': True
        }
        
        if frame is None or frame.size == 0:
            quality_info['is_valid'] = False
            quality_info['blur_score'] = 0
            return quality_info
            
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Check blur using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            # Normalize to 0-100 scale (higher is less blurry)
            quality_info['blur_score'] = min(100, laplacian_var)
            
            # Check brightness
            quality_info['brightness'] = np.mean(gray)
            
            # Check contrast (standard deviation)
            quality_info['contrast'] = np.std(gray)
            
            # Determine if frame is valid for processing
            quality_info['is_valid'] = (
                quality_info['blur_score'] > 10 and
                quality_info['brightness'] > 20 and
                quality_info['brightness'] < 235
            )
            
        except Exception as e:
            logger.error(f"Error analyzing frame quality: {e}")
            quality_info['is_valid'] = False
            
        return quality_info
        
    def analyze_video_quality(self, video_path: str, sample_frames: int = 10) -> Dict:
        """
        Analyze overall video quality by sampling frames
        
        Args:
            video_path: Path to video file
            sample_frames: Number of frames to sample
            
        Returns:
            Dictionary with video quality metrics
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'error': 'Failed to open video'}
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Sample frames evenly throughout the video
        sample_interval = max(1, total_frames // sample_frames)
        
        quality_scores = []
        brightness_scores = []
        
        for i in range(0, total_frames, sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            
            if ret:
                quality = self.check_frame_quality(frame)
                quality_scores.append(quality['blur_score'])
                brightness_scores.append(quality['brightness'])
                
        cap.release()
        
        return {
            'avg_blur_score': np.mean(quality_scores) if quality_scores else 0,
            'min_blur_score': np.min(quality_scores) if quality_scores else 0,
            'avg_brightness': np.mean(brightness_scores) if brightness_scores else 128,
            'fps': fps,
            'total_frames': total_frames,
            'samples_analyzed': len(quality_scores)
        }


class FrameExtractor:
    """Extracts frames from video with various strategies"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def extract_key_frames(self, video_path: str, max_frames: int = 100) -> List[Tuple[int, np.ndarray]]:
        """
        Extract key frames from video
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of (frame_number, frame) tuples
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return []
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame interval
        interval = max(1, total_frames // max_frames)
        
        frames = []
        frame_num = 0
        
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_num % interval == 0:
                frames.append((frame_num, frame))
                
            frame_num += 1
            
        cap.release()
        return frames
        
    def extract_frames_by_quality(self, video_path: str, quality_threshold: float = 50.0) -> List[Tuple[int, np.ndarray]]:
        """
        Extract frames that meet quality threshold
        
        Args:
            video_path: Path to video file
            quality_threshold: Minimum blur score (0-100)
            
        Returns:
            List of (frame_number, frame) tuples
        """
        analyzer = VideoQualityAnalyzer()
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return []
            
        frames = []
        frame_num = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            quality = analyzer.check_frame_quality(frame)
            if quality['blur_score'] >= quality_threshold:
                frames.append((frame_num, frame))
                
            frame_num += 1
            
            # Log progress
            if frame_num % 100 == 0:
                logger.debug(f"Processed {frame_num} frames, extracted {len(frames)} quality frames")
                
        cap.release()
        return frames