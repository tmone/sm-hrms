#!/usr/bin/env python3
"""
Conversion Manager - Handles video conversion with progress tracking
"""
import threading
import time
import uuid
import os
from datetime import datetime
from typing import Dict, Any, Optional, Callable

class ConversionTask:
    """Represents a single video conversion task"""
    
    def __init__(self, task_id: str, video_id: int, input_path: str, output_path: str):
        self.task_id = task_id
        self.video_id = video_id
        self.input_path = input_path
        self.output_path = output_path
        self.status = 'queued'  # queued, running, completed, failed
        self.progress = 0.0  # 0.0 to 100.0
        self.message = ''
        self.started_at = None
        self.completed_at = None
        self.error_message = None
        
    def to_dict(self):
        return {
            'task_id': self.task_id,
            'video_id': self.video_id,
            'status': self.status,
            'progress': self.progress,
            'message': self.message,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error_message': self.error_message
        }

class ConversionManager:
    """Manages video conversion tasks with progress tracking"""
    
    def __init__(self):
        self.tasks: Dict[str, ConversionTask] = {}
        self.active_threads: Dict[str, threading.Thread] = {}
        self.lock = threading.Lock()
        
    def create_task(self, video_id: int, input_path: str, output_path: str) -> str:
        """Create a new conversion task"""
        task_id = str(uuid.uuid4())
        task = ConversionTask(task_id, video_id, input_path, output_path)
        
        with self.lock:
            self.tasks[task_id] = task
            
        return task_id
    
    def get_task(self, task_id: str) -> Optional[ConversionTask]:
        """Get task by ID"""
        with self.lock:
            return self.tasks.get(task_id)
    
    def get_task_by_video_id(self, video_id: int) -> Optional[ConversionTask]:
        """Get task by video ID"""
        with self.lock:
            for task in self.tasks.values():
                if task.video_id == video_id and task.status in ['queued', 'running']:
                    return task
        return None
    
    def update_progress(self, task_id: str, progress: float, message: str = ''):
        """Update task progress and emit WebSocket event"""
        with self.lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.progress = progress
                task.message = message
                
                print(f"[PROCESSING] Task {task_id[:8]}: {progress:.1f}% - {message}")
                
                # Emit WebSocket event for real-time updates
                self._emit_progress_update(task)
    
    def _emit_progress_update(self, task):
        """Emit WebSocket event for progress update"""
        try:
            # Try to import SocketIO
            from flask import current_app
            from flask_socketio import emit
            
            if hasattr(current_app, '_get_current_object'):
                app = current_app._get_current_object()
                
                # Check if socketio is available
                if hasattr(app, 'extensions') and 'socketio' in app.extensions:
                    
                    # Prepare progress data
                    progress_data = {
                        'task_id': task.task_id,
                        'video_id': task.video_id,
                        'status': task.status,
                        'progress': task.progress,
                        'message': task.message,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    
                    # Emit to specific video room and general progress room
                    from flask_socketio import SocketIO
                    socketio = app.extensions['socketio']
                    
                    socketio.emit('conversion_progress', progress_data, room=f'video_{task.video_id}')
                    socketio.emit('conversion_progress', progress_data, room='admin')
                    
                    print(f"📡 WebSocket: Emitted progress for video {task.video_id}: {task.progress:.1f}%")
                    
        except Exception as e:
            print(f"[WARNING] WebSocket emit failed: {e}")
            # Continue without WebSocket - fallback to polling
    
    def start_conversion(self, task_id: str, app, processor_callback: Callable):
        """Start conversion in background thread"""
        task = self.get_task(task_id)
        if not task:
            return False
            
        def conversion_worker():
            with app.app_context():
                try:
                    # Update task status
                    with self.lock:
                        task.status = 'running'
                        task.started_at = datetime.utcnow()
                        task.progress = 0.0
                        task.message = 'Starting conversion...'
                    
                    # Create progress callback
                    def progress_callback(progress: float, message: str = ''):
                        self.update_progress(task_id, progress, message)
                    
                    # Run conversion with progress tracking
                    success = processor_callback(task, progress_callback, app)
                    
                    # Update final status
                    with self.lock:
                        if success:
                            task.status = 'completed'
                            task.progress = 100.0
                            task.message = 'Conversion completed successfully'
                        else:
                            task.status = 'failed'
                            task.error_message = task.message or 'Conversion failed'
                        
                        task.completed_at = datetime.utcnow()
                        
                        # Clean up thread reference
                        if task_id in self.active_threads:
                            del self.active_threads[task_id]
                    
                    print(f"[OK] Task {task_id[:8]} completed: success={success}")
                    
                except Exception as e:
                    with self.lock:
                        task.status = 'failed'
                        task.error_message = str(e)
                        task.completed_at = datetime.utcnow()
                        
                        if task_id in self.active_threads:
                            del self.active_threads[task_id]
                    
                    print(f"[ERROR] Task {task_id[:8]} failed: {e}")
        
        # Start background thread
        thread = threading.Thread(target=conversion_worker, daemon=True)
        thread.start()
        
        with self.lock:
            self.active_threads[task_id] = thread
            
        return True
    
    def get_all_tasks(self) -> Dict[str, Dict]:
        """Get all tasks as dictionary"""
        with self.lock:
            return {tid: task.to_dict() for tid, task in self.tasks.items()}
    
    def cleanup_completed_tasks(self, max_age_hours: int = 24):
        """Clean up old completed tasks"""
        cutoff = datetime.utcnow().timestamp() - (max_age_hours * 3600)
        
        with self.lock:
            to_remove = []
            for task_id, task in self.tasks.items():
                if (task.status in ['completed', 'failed'] and 
                    task.completed_at and 
                    task.completed_at.timestamp() < cutoff):
                    to_remove.append(task_id)
            
            for task_id in to_remove:
                del self.tasks[task_id]
                if task_id in self.active_threads:
                    del self.active_threads[task_id]
            
            if to_remove:
                print(f"[CLEANUP] Cleaned up {len(to_remove)} old conversion tasks")

# Global conversion manager instance
conversion_manager = ConversionManager()

def create_conversion_processor(video_processor):
    """Create a conversion processor with progress tracking"""
    
    def process_with_progress(task: ConversionTask, progress_callback: Callable, app):
        """Process video conversion with progress tracking"""
        try:
            import cv2
            import os
            
            progress_callback(5.0, "Opening input video...")
            
            # Open input video
            cap = cv2.VideoCapture(task.input_path)
            if not cap.isOpened():
                progress_callback(0.0, "Failed to open input video")
                return False
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 24
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            progress_callback(10.0, f"Video: {width}x{height}, {fps}fps, {total_frames} frames")
            
            # Try different codecs optimized for web streaming
            codecs = [
                ('avc1', 'H.264/AVC1 (Web optimized)'),
                ('h264', 'H.264 (Web compatible)'),
                ('mp4v', 'MP4V (Fallback)'),
                ('XVID', 'XVID (Legacy)'), 
                ('MJPG', 'MJPG (Last resort)')
            ]
            
            out = None
            for codec_name, codec_desc in codecs:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec_name)
                    
                    # Use .mp4 extension to ensure proper container format
                    output_path_mp4 = str(task.output_path)
                    if not output_path_mp4.endswith('.mp4'):
                        output_path_mp4 += '.mp4'
                    
                    out = cv2.VideoWriter(output_path_mp4, fourcc, fps, (width, height))
                    
                    if out and out.isOpened():
                        progress_callback(15.0, f"Using codec: {codec_desc}")
                        # Update task output path to reflect the actual file
                        task.output_path = output_path_mp4
                        break
                    else:
                        out = None
                except Exception as codec_error:
                    progress_callback(15.0, f"Codec {codec_desc} failed: {codec_error}")
                    continue
            
            if not out or not out.isOpened():
                cap.release()
                progress_callback(0.0, "Failed to initialize video writer")
                return False
            
            # Process frames with progress tracking
            frame_count = 0
            last_progress_update = 0
            
            progress_callback(20.0, "Starting frame processing...")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Write frame
                out.write(frame)
                frame_count += 1
                
                # Update progress every 100 frames or 5%
                if (frame_count - last_progress_update >= 100 or 
                    (total_frames > 0 and frame_count % max(1, total_frames // 20) == 0)):
                    
                    if total_frames > 0:
                        # Progress from 20% to 90% based on frame processing
                        frame_progress = (frame_count / total_frames) * 70  # 70% of total progress
                        total_progress = 20 + frame_progress
                        progress_callback(total_progress, f"Processed {frame_count}/{total_frames} frames")
                    else:
                        progress_callback(50.0, f"Processed {frame_count} frames")
                    
                    last_progress_update = frame_count
            
            # Clean up
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            
            progress_callback(95.0, "Finalizing video file...")
            
            # Verify output
            if os.path.exists(task.output_path) and os.path.getsize(task.output_path) > 0:
                file_size = os.path.getsize(task.output_path) / 1024 / 1024  # MB
                progress_callback(95.0, f"Initial conversion completed! Testing web compatibility...")
                
                # Test if the converted video is actually web-compatible
                test_compatible = test_video_compatibility(task.output_path)
                
                if not test_compatible:
                    progress_callback(96.0, "Video not web-compatible, trying FFmpeg conversion...")
                    
                    # Try FFmpeg conversion as fallback
                    ffmpeg_output = task.output_path.replace('.mp4', '_ffmpeg.mp4')
                    if try_ffmpeg_conversion(task.input_path, ffmpeg_output, progress_callback):
                        # Use FFmpeg output instead
                        os.rename(ffmpeg_output, task.output_path)
                        progress_callback(99.0, f"FFmpeg conversion successful! Output: {file_size:.1f}MB")
                    else:
                        progress_callback(97.0, f"FFmpeg failed, using OpenCV output: {file_size:.1f}MB")
                else:
                    progress_callback(99.0, f"Web-compatible conversion completed! Output: {file_size:.1f}MB")
                
                progress_callback(100.0, f"Conversion completed! Output: {file_size:.1f}MB")
                
                # Update database
                try:
                    Video = app.Video
                    video = Video.query.get(task.video_id)
                    if video:
                        video.status = 'completed'
                        video.processed_path = os.path.basename(task.output_path)
                        video.processing_completed_at = datetime.utcnow()
                        video.error_message = None
                        app.db.session.commit()
                        
                except Exception as db_error:
                    print(f"Database update error: {db_error}")
                
                return True
            else:
                progress_callback(0.0, "Output file was not created or is empty")
                return False
                
        except Exception as e:
            progress_callback(0.0, f"Conversion error: {str(e)}")
            
            # Update database with error
            try:
                Video = app.Video
                video = Video.query.get(task.video_id)
                if video:
                    video.status = 'failed'
                    video.error_message = str(e)
                    video.processing_completed_at = datetime.utcnow()
                    app.db.session.commit()
            except Exception as db_error:
                print(f"Database error update failed: {db_error}")
            
            return False
    
    return process_with_progress

def test_video_compatibility(video_path):
    """Test if a video file is web browser compatible"""
    try:
        import cv2
        
        # Try to open and read the first few frames
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        
        # Check if we can read frames
        for i in range(5):  # Test first 5 frames
            ret, frame = cap.read()
            if not ret:
                break
        
        cap.release()
        
        # Check file header for proper MP4 structure
        with open(video_path, 'rb') as f:
            header = f.read(32)
            
            # Look for proper MP4 markers
            if b'ftyp' in header and (b'isom' in header or b'mp4' in header):
                return True
            
        return False
        
    except Exception as e:
        print(f"Compatibility test error: {e}")
        return False

def try_ffmpeg_conversion(input_path, output_path, progress_callback):
    """Try FFmpeg conversion as a fallback"""
    try:
        import subprocess
        import shlex
        
        progress_callback(96.5, "Attempting FFmpeg conversion...")
        
        # FFmpeg command for web-compatible MP4
        cmd = [
            'ffmpeg', '-y',  # Overwrite output
            '-i', input_path,  # Input file
            '-c:v', 'libx264',  # H.264 video codec
            '-preset', 'medium',  # Encoding speed/quality tradeoff
            '-crf', '23',  # Quality (lower = better, 18-28 is good range)
            '-c:a', 'aac',  # AAC audio codec
            '-movflags', '+faststart',  # Web optimization - move metadata to beginning
            '-pix_fmt', 'yuv420p',  # Pixel format compatible with most browsers
            output_path
        ]
        
        # Run FFmpeg
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if process.returncode == 0:
            progress_callback(98.0, "FFmpeg conversion successful")
            return True
        else:
            print(f"FFmpeg error: {process.stderr}")
            progress_callback(97.0, f"FFmpeg failed: {process.stderr[:100]}")
            return False
            
    except subprocess.TimeoutExpired:
        progress_callback(97.0, "FFmpeg conversion timed out")
        return False
    except FileNotFoundError:
        progress_callback(97.0, "FFmpeg not found - install with: apt install ffmpeg")
        return False
    except Exception as e:
        progress_callback(97.0, f"FFmpeg error: {str(e)}")
        return False