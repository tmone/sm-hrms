"""
Example of how to update your code to use the new logging system
"""

# OLD WAY (too much console output):
"""
print("🚀 Starting GPU-accelerated person extraction for video")
print(f"✅ Video file exists: {video_path} ({os.path.getsize(video_path)} bytes)")
print("🎮 Using GPU-accelerated detection")
print(f"🎯 GPU detection completed - found {len(detections)} tracked detections")
"""

# NEW WAY (cleaner console, detailed file logs):
"""
from config_logging import get_logger
from utils.progress_logger import VideoProcessingProgress, simple_progress

# Get logger for your module
logger = get_logger(__name__)

# For video processing with progress bar:
progress = VideoProcessingProgress(video_filename, total_chunks=40)

# Update progress
progress.update_chunk(1, "Processing chunk")
progress.update_chunk(2, "Processing chunk")

# Log errors (always shown on console)
progress.log_error("Failed to process chunk 3")

# Log success
progress.log_success("Video processing completed")

# For simple messages:
simple_progress("Uploading video...")
simple_progress("Processing", current=10, total=100)

# For detailed logging (goes to file only):
logger.debug("Detailed debug information")
logger.info("Regular information")

# For important messages (shown on console):
logger.warning("This is important")
logger.error("This is an error")

# For GPU operations:
from utils.progress_logger import GPUProcessingProgress

gpu_progress = GPUProcessingProgress("YOLO Detection")
gpu_progress.log_status("Initializing CUDA")
gpu_progress.log_memory(used_mb=2048, total_mb=8192)
gpu_progress.log_status("Detection completed", level='success')
"""

# BENEFITS:
# 1. Console shows only important messages
# 2. Detailed logs go to files (logs/video_processing_20250107.log, etc.)
# 3. Progress bars for long operations
# 4. Automatic log rotation (max 10MB per file, keeps 5 backups)
# 5. Separate log files for different components

# LOG FILE LOCATIONS:
# - logs/api_YYYYMMDD.log - API requests and responses
# - logs/background_YYYYMMDD.log - Background tasks
# - logs/video_processing_YYYYMMDD.log - Video processing details
# - logs/gpu_YYYYMMDD.log - GPU operations
# - logs/database_YYYYMMDD.log - Database queries (warnings only)
# - logs/app_YYYYMMDD.log - General application logs

print("""
To view logs in real-time:
- Windows: type logs\\video_processing_20250107.log
- Linux: tail -f logs/video_processing_20250107.log

To search logs:
- Windows: findstr "error" logs\\*.log
- Linux: grep "error" logs/*.log
""")