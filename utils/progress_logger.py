"""
Progress logger utilities for cleaner console output
"""
import sys
from datetime import datetime
from config_logging import get_logger, ProgressBar

class VideoProcessingProgress:
    """Manages progress display for video processing"""
    
    def __init__(self, video_name, total_chunks=1):
        self.video_name = video_name
        self.total_chunks = total_chunks
        self.current_chunk = 0
        self.start_time = datetime.now()
        self.logger = get_logger('progress')
        
        # Create progress bar
        self.progress_bar = None
        if total_chunks > 1:
            self.progress_bar = ProgressBar(total_chunks, prefix=f"Processing {video_name}")
        
    def update_chunk(self, chunk_num, message=""):
        """Update chunk progress"""
        self.current_chunk = chunk_num
        if self.progress_bar:
            self.progress_bar.update(chunk_num)
        
        # Log to file
        file_logger = get_logger('video_processing')
        file_logger.info(f"[{self.video_name}] Chunk {chunk_num}/{self.total_chunks}: {message}")
        
    def log_error(self, error_message):
        """Log error (always shown on console)"""
        self.logger.error(f"[ERROR] [{self.video_name}] {error_message}")
        
    def log_success(self, message):
        """Log success message"""
        if self.progress_bar:
            self.progress_bar.finish()
            
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.logger.info(f"[OK] [{self.video_name}] {message} (took {elapsed:.1f}s)")
        
    def log_info(self, message, to_console=False):
        """Log info message (file only by default)"""
        file_logger = get_logger('video_processing')
        file_logger.info(f"[{self.video_name}] {message}")
        
        if to_console:
            self.logger.info(f"[{self.video_name}] {message}")


class GPUProcessingProgress:
    """Manages progress display for GPU operations"""
    
    def __init__(self, operation_name):
        self.operation_name = operation_name
        self.logger = get_logger('progress')
        self.file_logger = get_logger('gpu')
        self.start_time = datetime.now()
        
    def log_status(self, message, level='info'):
        """Log GPU status"""
        self.file_logger.info(f"[{self.operation_name}] {message}")
        
        # Only show important GPU messages on console
        if level == 'error':
            self.logger.error(f"[GPU] [{self.operation_name}] {message}")
        elif level == 'success':
            self.logger.info(f"[GPU] [{self.operation_name}] {message}")
            
    def log_memory(self, used_mb, total_mb):
        """Log GPU memory usage"""
        percent = (used_mb / total_mb * 100) if total_mb > 0 else 0
        self.file_logger.info(f"[{self.operation_name}] GPU Memory: {used_mb:.0f}/{total_mb:.0f} MB ({percent:.1f}%)")
        
        # Show on console if high usage
        if percent > 80:
            self.logger.warning(f"[WARNING] High GPU memory usage: {percent:.1f}%")


def simple_progress(message, current=None, total=None):
    """Simple progress message for console"""
    progress_logger = get_logger('progress')
    
    if current is not None and total is not None:
        percent = (current / total * 100) if total > 0 else 0
        progress_logger.info(f"{message} ({current}/{total} - {percent:.1f}%)")
    else:
        progress_logger.info(message)


def log_api_request(method, endpoint, status_code=None, duration_ms=None):
    """Log API request (file only)"""
    api_logger = get_logger('api')
    
    message = f"{method} {endpoint}"
    if status_code:
        message += f" -> {status_code}"
    if duration_ms:
        message += f" ({duration_ms:.0f}ms)"
        
    api_logger.info(message)


def log_background_task(task_name, status, details=None):
    """Log background task status"""
    bg_logger = get_logger('background')
    
    message = f"[{task_name}] {status}"
    if details:
        message += f" - {details}"
        
    bg_logger.info(message)
    
    # Show on console if completed or failed
    if status.lower() in ['completed', 'failed', 'error']:
        progress_logger = get_logger('progress')
        if status.lower() in ['failed', 'error']:
            progress_logger.error(f"[ERROR] {message}")
        else:
            progress_logger.info(f"[OK] {message}")