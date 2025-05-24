"""
Alternative video writer using FFmpeg for better compression
"""
import cv2
import numpy as np
import subprocess
import os
from pathlib import Path

class FFmpegVideoWriter:
    """Video writer that uses FFmpeg directly for better compression"""
    
    def __init__(self, output_path, fps, width, height, codec='libx264', crf=23, preset='medium'):
        """
        Initialize FFmpeg video writer
        
        Args:
            output_path: Output video file path
            fps: Frame rate
            width: Video width
            height: Video height  
            codec: Video codec (libx264, libx265, etc.)
            crf: Constant Rate Factor (0-51, lower = better quality, 23 = default)
            preset: Encoding preset (ultrafast, fast, medium, slow, veryslow)
        """
        self.output_path = str(output_path)
        self.fps = fps
        self.width = width
        self.height = height
        self.codec = codec
        self.crf = crf
        self.preset = preset
        self.process = None
        self.frame_count = 0
        
    def open(self):
        """Open FFmpeg process for writing"""
        # FFmpeg command for H.264 encoding with good compression
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-f', 'rawvideo',  # Input format
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',  # OpenCV uses BGR
            '-s', f'{self.width}x{self.height}',  # Size
            '-r', str(self.fps),  # Input framerate
            '-i', '-',  # Input from pipe
            '-c:v', self.codec,  # Video codec
            '-crf', str(self.crf),  # Quality setting
            '-preset', self.preset,  # Encoding speed/quality tradeoff
            '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
            '-movflags', '+faststart',  # Enable streaming
            self.output_path
        ]
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            return True
        except Exception as e:
            print(f"Failed to start FFmpeg: {e}")
            return False
    
    def write(self, frame):
        """Write a frame to the video"""
        if self.process is None:
            if not self.open():
                return False
        
        try:
            # Write frame to FFmpeg stdin
            self.process.stdin.write(frame.tobytes())
            self.frame_count += 1
            return True
        except Exception as e:
            print(f"Error writing frame: {e}")
            return False
    
    def release(self):
        """Close the video writer"""
        if self.process:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=30)
                
                # Check if FFmpeg succeeded
                if self.process.returncode != 0:
                    stderr = self.process.stderr.read().decode()
                    print(f"FFmpeg error: {stderr}")
                else:
                    # Get file size
                    if os.path.exists(self.output_path):
                        size_mb = os.path.getsize(self.output_path) / (1024 * 1024)
                        print(f"✅ Video saved: {self.output_path} ({size_mb:.1f} MB)")
                        print(f"   Frames: {self.frame_count}, Codec: {self.codec}, CRF: {self.crf}")
                
            except Exception as e:
                print(f"Error closing FFmpeg: {e}")
            finally:
                self.process = None
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def create_video_writer(output_path, fps, width, height, use_ffmpeg=True):
    """
    Create a video writer with optimal settings for file size
    
    Args:
        output_path: Output video path
        fps: Frame rate
        width: Video width
        height: Video height
        use_ffmpeg: Use FFmpeg writer if available
    
    Returns:
        Video writer object
    """
    # Check if FFmpeg is available
    if use_ffmpeg:
        try:
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True)
            if result.returncode == 0:
                print("✅ Using FFmpeg for video encoding (better compression)")
                return FFmpegVideoWriter(output_path, fps, width, height, crf=23)
        except:
            print("⚠️  FFmpeg not found, falling back to OpenCV")
    
    # Fallback to OpenCV
    print("📹 Using OpenCV VideoWriter")
    
    # Try H.264 first
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height), True)
    
    if writer.isOpened():
        # Set compression parameters
        writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 85)
        return writer
    
    # Try alternative codecs
    for codec in ['h264', 'avc1', 'mp4v']:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height), True)
        if writer.isOpened():
            writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 85)
            print(f"✅ Using {codec} codec")
            return writer
    
    # Last resort - XVID in AVI
    output_avi = str(output_path).replace('.mp4', '.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(output_avi, fourcc, fps, (width, height), True)
    
    if writer.isOpened():
        print("⚠️  Using XVID codec in AVI container")
        return writer
    
    raise Exception("Could not create video writer with any codec")