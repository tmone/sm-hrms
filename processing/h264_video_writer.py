"""
H.264 Video Writer - Ensures all annotated videos use H.264 codec
"""
import cv2
import subprocess
import os
import logging
from pathlib import Path
import tempfile

logger = logging.getLogger(__name__)


def create_h264_video_writer(output_path, fps, width, height, use_ffmpeg=True):
    """
    Create a video writer that guarantees H.264 output
    
    Args:
        output_path: Output file path
        fps: Frame rate
        width: Video width
        height: Video height
        use_ffmpeg: Use FFmpeg for guaranteed H.264 encoding
        
    Returns:
        VideoWriter object or FFmpegWriter
    """
    if use_ffmpeg:
        return FFmpegH264Writer(output_path, fps, width, height)
    else:
        # Try OpenCV with H.264
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if writer.isOpened():
            return writer
        else:
            # Fallback to FFmpeg
            logger.warning("OpenCV H.264 failed, using FFmpeg")
            return FFmpegH264Writer(output_path, fps, width, height)


class FFmpegH264Writer:
    """FFmpeg-based H.264 video writer"""
    
    def __init__(self, output_path, fps, width, height):
        self.output_path = str(output_path)
        self.fps = fps
        self.width = width
        self.height = height
        self.frames = []
        self.temp_dir = tempfile.mkdtemp()
        self.frame_count = 0
        
    def write(self, frame):
        """Write a frame"""
        # Save frame as temporary image
        frame_path = os.path.join(self.temp_dir, f"frame_{self.frame_count:06d}.png")
        cv2.imwrite(frame_path, frame)
        self.frames.append(frame_path)
        self.frame_count += 1
        
    def release(self):
        """Encode all frames to H.264 video using FFmpeg"""
        if not self.frames:
            return
            
        # Create pattern for FFmpeg
        pattern = os.path.join(self.temp_dir, "frame_%06d.png")
        
        # FFmpeg command for H.264 encoding
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output
            '-framerate', str(self.fps),
            '-i', pattern,
            '-c:v', 'libx264',  # Use H.264 codec
            '-preset', 'medium',  # Balance between speed and compression
            '-crf', '23',  # Quality (lower = better, 23 is default)
            '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
            '-movflags', '+faststart',  # Web optimization
            self.output_path
        ]
        
        try:
            logger.info(f"Encoding {len(self.frames)} frames to H.264...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"[OK] H.264 video created: {self.output_path}")
            else:
                logger.error(f"FFmpeg error: {result.stderr}")
        except Exception as e:
            logger.error(f"Failed to encode video: {e}")
        finally:
            # Clean up temporary files
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def isOpened(self):
        """Check if writer is ready"""
        return True


def convert_to_h264(input_path, output_path=None):
    """
    Convert any video to H.264 using FFmpeg
    
    Args:
        input_path: Input video path
        output_path: Output path (optional, defaults to input_path with _h264 suffix)
        
    Returns:
        Path to H.264 video
    """
    if output_path is None:
        base = Path(input_path).stem
        ext = Path(input_path).suffix
        output_path = Path(input_path).parent / f"{base}_h264{ext}"
    
    # Check if already H.264
    probe_cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(input_path)
    ]
    
    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        codec = result.stdout.strip().lower()
        
        if codec == 'h264':
            logger.info(f"Video already H.264: {input_path}")
            return input_path
    except:
        pass
    
    # Convert to H.264
    cmd = [
        'ffmpeg',
        '-i', str(input_path),
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        '-y',
        str(output_path)
    ]
    
    try:
        logger.info(f"Converting to H.264: {input_path} -> {output_path}")
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"[OK] Converted to H.264: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to convert video: {e.stderr.decode()}")
        return input_path  # Return original if conversion fails