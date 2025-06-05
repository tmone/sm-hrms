import os
import sys
import cv2
import numpy as np
import subprocess
import tempfile
import shutil
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from queue import Queue
from threading import Thread, Lock
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from collections import defaultdict
import json
import hashlib
import signal
import atexit

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from processing.enhanced_detection import EnhancedDetection
from processing.video_quality_analyzer import VideoQualityAnalyzer, FrameExtractor
from processing.enhanced_person_tracker import EnhancedPersonTracker
from hr_management.processing.person_recognition_inference_simple import SimplePersonRecognitionInference
from processing.gpu_resource_manager import get_gpu_manager, cleanup_gpu_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Register cleanup on exit
atexit.register(cleanup_gpu_manager)


class SharedStateManager:
    """Manages shared state across workers for person ID consistency"""
    
    def __init__(self):
        self.lock = Lock()
        self.person_counter = 0
        self.recognized_to_person_id = {}  # recognized_id -> assigned person_id
        self.track_to_person_id = {}  # track_id -> person_id
        self.person_id_appearances = defaultdict(list)  # person_id -> [(chunk_idx, frame_num)]
        
    def get_next_person_id(self) -> str:
        """Get next available person ID"""
        with self.lock:
            self.person_counter += 1
            return f"PERSON-{self.person_counter:04d}"
            
    def assign_person_id(self, recognized_id: Optional[str], track_id: int, 
                        chunk_idx: int, frame_num: int) -> str:
        """Assign person ID based on recognition or create new one"""
        with self.lock:
            # Check if this recognized person already has an ID
            if recognized_id and recognized_id in self.recognized_to_person_id:
                person_id = self.recognized_to_person_id[recognized_id]
            # Check if this track already has an ID
            elif track_id in self.track_to_person_id:
                person_id = self.track_to_person_id[track_id]
            else:
                # Create new person ID
                person_id = self.get_next_person_id()
                
                # Store mappings
                if recognized_id:
                    self.recognized_to_person_id[recognized_id] = person_id
                self.track_to_person_id[track_id] = person_id
            
            # Record appearance
            self.person_id_appearances[person_id].append((chunk_idx, frame_num))
            
            return person_id
            
    def get_state_dict(self) -> Dict:
        """Get current state as dictionary"""
        with self.lock:
            return {
                'person_counter': self.person_counter,
                'recognized_to_person_id': dict(self.recognized_to_person_id),
                'track_to_person_id': dict(self.track_to_person_id),
                'person_id_appearances': dict(self.person_id_appearances)
            }


class ChunkProcessor:
    """Processes a single video chunk"""
    
    def __init__(self, chunk_path: str, chunk_idx: int, output_dir: str,
                 shared_state: SharedStateManager, worker_id: str,
                 gpu_manager=None):
        self.chunk_path = chunk_path
        self.chunk_idx = chunk_idx
        self.output_dir = output_dir
        self.shared_state = shared_state
        self.worker_id = worker_id
        self.gpu_manager = gpu_manager or get_gpu_manager()
        
        # GPU allocation
        self.gpu_id = None
        self.use_gpu = False
        
        # Try to acquire GPU
        if self.gpu_manager.get_gpu_count() > 0:
            self.gpu_id = self.gpu_manager.acquire_gpu(self.worker_id, timeout=10.0)
            if self.gpu_id is not None:
                self.use_gpu = True
                logger.info(f"Worker {worker_id} (Chunk {chunk_idx}): Acquired GPU {self.gpu_id}")
                
                # Set CUDA device for this process
                os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
            else:
                logger.warning(f"Worker {worker_id} (Chunk {chunk_idx}): "
                             f"Could not acquire GPU, using CPU fallback")
        
        # Initialize components with GPU/CPU setting
        try:
            self.detector = EnhancedDetection(use_gpu=self.use_gpu)
        except Exception as e:
            logger.error(f"Failed to initialize detector with GPU={self.use_gpu}: {e}")
            # Fallback to CPU
            self.use_gpu = False
            self.detector = EnhancedDetection(use_gpu=False)
            
        self.quality_analyzer = VideoQualityAnalyzer()
        self.frame_extractor = FrameExtractor()
        self.tracker = EnhancedPersonTracker(use_recognition=True)
        
        # Initialize recognition model
        try:
            self.recognizer = SimplePersonRecognitionInference()
            self.use_recognition = True
            logger.info(f"Worker {worker_id} (Chunk {chunk_idx}): "
                       f"Recognition model loaded successfully")
        except Exception as e:
            logger.warning(f"Worker {worker_id} (Chunk {chunk_idx}): "
                          f"Failed to load recognition model: {e}")
            self.recognizer = None
            self.use_recognition = False
            
    def __del__(self):
        """Release GPU when processor is destroyed"""
        if self.gpu_id is not None:
            self.gpu_manager.release_gpu(self.worker_id)
            
    def process(self) -> Dict:
        """Process the chunk and return results"""
        logger.info(f"Worker {self.worker_id} processing chunk {self.chunk_idx}: {self.chunk_path}")
        logger.info(f"Using {'GPU ' + str(self.gpu_id) if self.use_gpu else 'CPU'}")
        
        # Get initial GPU status
        if self.use_gpu and self.gpu_id is not None:
            gpu_report = self.gpu_manager.get_monitoring_report()
            for gpu_stat in gpu_report.get('gpu_stats', []):
                if gpu_stat['gpu_id'] == self.gpu_id:
                    logger.info(f"GPU {self.gpu_id} status - "
                              f"Memory: {gpu_stat['memory_percent']:.1f}%, "
                              f"Temp: {gpu_stat['temperature']:.1f}°C, "
                              f"Util: {gpu_stat['utilization']:.1f}%")
        
        cap = cv2.VideoCapture(self.chunk_path)
        if not cap.isOpened():
            logger.error(f"Failed to open chunk {self.chunk_idx}")
            return {'chunk_idx': self.chunk_idx, 'status': 'failed', 'detections': []}
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create chunk output directory
        chunk_output = os.path.join(self.output_dir, f"chunk_{self.chunk_idx:03d}")
        os.makedirs(chunk_output, exist_ok=True)
        
        detections = []
        frame_num = 0
        processed_frames = 0
        
        # Calculate global frame offset (30 seconds per chunk)
        global_frame_offset = self.chunk_idx * 30 * fps
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            global_frame_num = global_frame_offset + frame_num
            
            # Check frame quality
            quality_info = self.quality_analyzer.check_frame_quality(frame)
            
            # Skip very blurry frames
            if quality_info['blur_score'] < 50:
                logger.debug(f"Chunk {self.chunk_idx}, Frame {frame_num}: Skipping due to blur")
                frame_num += 1
                continue
                
            # Run detection
            try:
                # Detect persons in frame
                boxes, confidences = self.detector.detect_persons(frame)
                
                if len(boxes) > 0:
                    # Update tracker
                    track_ids = self.tracker.update(boxes, frame)
                    
                    # Process each detection
                    for i, (box, conf, track_id) in enumerate(zip(boxes, confidences, track_ids)):
                        x1, y1, x2, y2 = box
                        
                        # Extract person crop
                        person_crop = frame[y1:y2, x1:x2]
                        
                        # Check crop quality
                        if person_crop.size == 0:
                            continue
                            
                        crop_quality = self.quality_analyzer.check_frame_quality(person_crop)
                        
                        # Skip poor quality crops
                        if crop_quality['blur_score'] < 40:
                            continue
                        
                        # Get recognition result if available
                        recognized_id = None
                        if self.use_recognition and self.recognizer:
                            try:
                                # Use enhanced tracker's recognition
                                recognized_id = self.tracker.get_recognized_person_id(track_id)
                                
                                if not recognized_id:
                                    # Try direct recognition
                                    recog_result = self.recognizer.predict_single(person_crop)
                                    if recog_result and recog_result.get('confidence', 0) > 0.85:
                                        recognized_id = recog_result.get('person_id')
                                        self.tracker.set_recognized_person_id(track_id, recognized_id)
                            except Exception as e:
                                logger.debug(f"Recognition failed: {e}")
                        
                        # Get person ID from shared state
                        person_id = self.shared_state.assign_person_id(
                            recognized_id, track_id, self.chunk_idx, global_frame_num
                        )
                        
                        # Save high-quality crop
                        if crop_quality['blur_score'] > 70:  # Only save good quality crops
                            crop_filename = f"{person_id}_chunk{self.chunk_idx:03d}_frame{frame_num:06d}.jpg"
                            crop_path = os.path.join(chunk_output, crop_filename)
                            cv2.imwrite(crop_path, person_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        
                        # Store detection info
                        detection = {
                            'chunk_idx': self.chunk_idx,
                            'frame_num': frame_num,
                            'global_frame_num': global_frame_num,
                            'timestamp': global_frame_num / fps,
                            'person_id': person_id,
                            'recognized_id': recognized_id,
                            'track_id': track_id,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(conf),
                            'quality_score': crop_quality['blur_score']
                        }
                        detections.append(detection)
                        
                processed_frames += 1
                
            except Exception as e:
                logger.error(f"Error processing frame {frame_num} in chunk {self.chunk_idx}: {e}")
                
            frame_num += 1
            
            # Log progress and check GPU health
            if frame_num % 100 == 0:
                logger.info(f"Chunk {self.chunk_idx}: Processed {frame_num}/{total_frames} frames")
                
                # Check GPU health
                if self.use_gpu and self.gpu_id is not None and frame_num % 300 == 0:
                    gpu_report = self.gpu_manager.get_monitoring_report()
                    for gpu_stat in gpu_report.get('gpu_stats', []):
                        if gpu_stat['gpu_id'] == self.gpu_id:
                            # Check if GPU is still healthy
                            if not gpu_stat['is_available']:
                                logger.warning(f"GPU {self.gpu_id} unhealthy! "
                                             f"Memory: {gpu_stat['memory_percent']:.1f}%, "
                                             f"Temp: {gpu_stat['temperature']:.1f}°C")
                                # Could implement fallback to CPU here if needed
                
        cap.release()
        
        logger.info(f"Chunk {self.chunk_idx} complete: {processed_frames} frames processed, "
                   f"{len(detections)} detections")
        
        return {
            'chunk_idx': self.chunk_idx,
            'status': 'success',
            'detections': detections,
            'total_frames': total_frames,
            'processed_frames': processed_frames
        }


class ChunkedVideoProcessor:
    """Main processor for handling large videos through chunking"""
    
    def __init__(self, max_workers: int = 4, chunk_duration: int = 30,
                 use_gpu_monitoring: bool = True):
        self.max_workers = max_workers
        self.chunk_duration = chunk_duration
        self.shared_state = SharedStateManager()
        self.use_gpu_monitoring = use_gpu_monitoring
        
        # Initialize GPU manager
        self.gpu_manager = get_gpu_manager()
        
        # Adjust workers based on available GPUs
        gpu_count = self.gpu_manager.get_gpu_count()
        if gpu_count > 0:
            # Limit workers to available GPUs or max_workers, whichever is smaller
            self.max_workers = min(self.max_workers, gpu_count)
            logger.info(f"Detected {gpu_count} GPUs, using {self.max_workers} workers")
        else:
            logger.info("No GPUs detected, using CPU processing")
        
    def split_video(self, video_path: str, output_dir: str) -> List[str]:
        """Split video into chunks using ffmpeg"""
        logger.info(f"Splitting video into {self.chunk_duration}s chunks")
        
        # Create chunks directory
        chunks_dir = os.path.join(output_dir, "chunks")
        os.makedirs(chunks_dir, exist_ok=True)
        
        # Get video duration
        cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', video_path
        ]
        try:
            duration = float(subprocess.check_output(cmd).decode().strip())
        except Exception as e:
            logger.error(f"Failed to get video duration: {e}")
            return []
            
        # Calculate number of chunks
        num_chunks = int(np.ceil(duration / self.chunk_duration))
        logger.info(f"Video duration: {duration:.1f}s, will create {num_chunks} chunks")
        
        chunk_paths = []
        
        # Split video into chunks
        for i in range(num_chunks):
            start_time = i * self.chunk_duration
            chunk_path = os.path.join(chunks_dir, f"chunk_{i:03d}.mp4")
            
            # FFmpeg command to extract chunk
            cmd = [
                'ffmpeg', '-i', video_path,
                '-ss', str(start_time),
                '-t', str(self.chunk_duration),
                '-c:v', 'libx264',  # Re-encode for compatibility
                '-preset', 'fast',
                '-crf', '22',  # Good quality
                '-c:a', 'copy',
                '-y',  # Overwrite
                chunk_path
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                chunk_paths.append(chunk_path)
                logger.info(f"Created chunk {i+1}/{num_chunks}: {chunk_path}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to create chunk {i}: {e.stderr.decode()}")
                
        return chunk_paths
        
    def process_video(self, video_path: str, output_dir: str) -> Dict:
        """Process large video by splitting into chunks and processing in parallel"""
        start_time = time.time()
        
        # Check file size
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        logger.info(f"Processing video: {video_path} ({file_size_mb:.1f} MB)")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Split video into chunks
        chunk_paths = self.split_video(video_path, output_dir)
        if not chunk_paths:
            logger.error("Failed to split video")
            return {'status': 'failed', 'error': 'Failed to split video'}
            
        # Show GPU status
        gpu_report = self.gpu_manager.get_monitoring_report()
        logger.info(f"GPU Status: {gpu_report['total_gpus']} total, "
                   f"{gpu_report['available_gpus']} available")
        
        # Process chunks in parallel
        all_detections = []
        chunk_results = []
        
        logger.info(f"Processing {len(chunk_paths)} chunks with {self.max_workers} workers")
        
        # Create thread pool for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all chunks for processing
            futures = []
            for idx, chunk_path in enumerate(chunk_paths):
                worker_id = f"chunk_{idx:03d}_{int(time.time()*1000)}"
                processor = ChunkProcessor(
                    chunk_path, idx, output_dir, 
                    self.shared_state, worker_id, self.gpu_manager
                )
                future = executor.submit(processor.process)
                futures.append((future, processor))
                
            # Collect results
            for future, processor in futures:
                try:
                    result = future.result(timeout=300)  # 5 minute timeout per chunk
                    chunk_results.append(result)
                    if result['status'] == 'success':
                        all_detections.extend(result['detections'])
                except Exception as e:
                    logger.error(f"Chunk processing failed: {e}")
                finally:
                    # Ensure GPU is released even if processing failed
                    if hasattr(processor, 'gpu_id') and processor.gpu_id is not None:
                        processor.gpu_manager.release_gpu(processor.worker_id)
                    
        # Sort detections by global frame number
        all_detections.sort(key=lambda x: x['global_frame_num'])
        
        # Generate annotated video
        annotated_path = self.create_annotated_video(
            video_path, all_detections, output_dir
        )
        
        # Clean up chunks
        logger.info("Cleaning up temporary chunks")
        chunks_dir = os.path.join(output_dir, "chunks")
        if os.path.exists(chunks_dir):
            shutil.rmtree(chunks_dir)
            
        # Get final GPU report
        final_gpu_report = self.gpu_manager.get_monitoring_report()
        
        # Save GPU monitoring history if enabled
        if self.use_gpu_monitoring:
            monitoring_file = os.path.join(output_dir, 'gpu_monitoring_history.json')
            self.gpu_manager.save_monitoring_history(monitoring_file)
        
        # Save processing metadata
        metadata = {
            'video_path': video_path,
            'file_size_mb': file_size_mb,
            'num_chunks': len(chunk_paths),
            'chunk_duration': self.chunk_duration,
            'num_workers': self.max_workers,
            'total_detections': len(all_detections),
            'unique_persons': len(self.shared_state.recognized_to_person_id),
            'processing_time': time.time() - start_time,
            'person_mappings': self.shared_state.get_state_dict(),
            'gpu_info': {
                'total_gpus': final_gpu_report['total_gpus'],
                'gpus_used': self.max_workers if final_gpu_report['total_gpus'] > 0 else 0,
                'monitoring_enabled': self.use_gpu_monitoring,
                'final_status': final_gpu_report['gpu_stats'] if final_gpu_report.get('gpu_stats') else []
            }
        }
        
        metadata_path = os.path.join(output_dir, 'processing_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Processing complete in {metadata['processing_time']:.1f}s")
        logger.info(f"Total detections: {metadata['total_detections']}")
        logger.info(f"Unique persons: {metadata['unique_persons']}")
        
        return {
            'status': 'success',
            'detections': all_detections,
            'metadata': metadata,
            'annotated_video': annotated_path
        }
        
    def create_annotated_video(self, original_video: str, detections: List[Dict], 
                              output_dir: str) -> str:
        """Create browser-compatible annotated video"""
        logger.info("Creating annotated video for browser preview")
        
        # Group detections by frame
        detections_by_frame = defaultdict(list)
        for det in detections:
            detections_by_frame[det['global_frame_num']].append(det)
            
        # Open original video
        cap = cv2.VideoCapture(original_video)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Output path for annotated video
        annotated_path = os.path.join(output_dir, 'annotated_preview.mp4')
        
        # FFmpeg command for H.264 encoding (browser compatible)
        cmd = [
            'ffmpeg',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'bgr24',
            '-r', str(fps),
            '-i', '-',  # Input from pipe
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',  # Browser compatible
            '-movflags', '+faststart',  # Enable streaming
            '-y',
            annotated_path
        ]
        
        # Start FFmpeg process
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        
        frame_num = 0
        annotated_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Get detections for this frame
            frame_detections = detections_by_frame.get(frame_num, [])
            
            # Draw annotations
            for det in frame_detections:
                x1, y1, x2, y2 = det['bbox']
                person_id = det['person_id']
                recognized_id = det.get('recognized_id', 'Unknown')
                confidence = det['confidence']
                
                # Draw bounding box
                color = (0, 255, 0) if recognized_id and recognized_id != 'Unknown' else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{person_id}"
                if recognized_id and recognized_id != 'Unknown':
                    label += f" ({recognized_id})"
                label += f" {confidence:.2f}"
                
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_y = y1 - 10 if y1 - 10 > 10 else y1 + label_size[1] + 10
                
                cv2.rectangle(frame, (x1, label_y - label_size[1] - 5),
                            (x1 + label_size[0] + 5, label_y + 5), color, -1)
                cv2.putText(frame, label, (x1 + 2, label_y - 2),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                          
                annotated_frames += 1
                
            # Write frame to FFmpeg
            process.stdin.write(frame.tobytes())
            
            frame_num += 1
            
            # Log progress
            if frame_num % 300 == 0:
                progress = (frame_num / total_frames) * 100
                logger.info(f"Annotating video: {progress:.1f}% ({frame_num}/{total_frames})")
                
        # Close everything
        cap.release()
        process.stdin.close()
        process.wait()
        
        logger.info(f"Annotated video created: {annotated_path}")
        logger.info(f"Annotated {annotated_frames} detections across {frame_num} frames")
        
        return annotated_path


def main():
    """Test the chunked video processor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process large videos using chunking')
    parser.add_argument('video_path', help='Path to input video')
    parser.add_argument('--output-dir', default='./chunked_output', 
                       help='Output directory (default: ./chunked_output)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    parser.add_argument('--chunk-duration', type=int, default=30,
                       help='Chunk duration in seconds (default: 30)')
    
    args = parser.parse_args()
    
    # Create processor
    processor = ChunkedVideoProcessor(
        max_workers=args.workers,
        chunk_duration=args.chunk_duration
    )
    
    # Process video
    result = processor.process_video(args.video_path, args.output_dir)
    
    if result['status'] == 'success':
        print(f"Processing successful!")
        print(f"Total detections: {len(result['detections'])}")
        print(f"Annotated video: {result['annotated_video']}")
        print(f"Metadata saved to: {args.output_dir}/processing_metadata.json")
    else:
        print(f"Processing failed: {result.get('error', 'Unknown error')}")
        

if __name__ == '__main__':
    main()