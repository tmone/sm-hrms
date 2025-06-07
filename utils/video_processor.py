#!/usr/bin/env python3
"""
Server-side video processing and conversion for HRM system
Handles IMKH and other non-standard formats
"""
import os
import tempfile
from pathlib import Path
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoProcessor:
    """Video processing class with multiple conversion backends"""
    
    def __init__(self):
        self.moviepy_available = False
        self.opencv_available = False
        self.ffmpeg_available = False
        
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check which video processing libraries are available"""
        try:
            import moviepy.editor as mp
            self.moviepy_available = True
            logger.info("[OK] MoviePy available")
        except ImportError:
            logger.warning("[WARNING] MoviePy not available")
        
        try:
            import cv2
            self.opencv_available = True
            logger.info("[OK] OpenCV available")
        except ImportError:
            logger.warning("[WARNING] OpenCV not available")
        
        try:
            import subprocess
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, timeout=5)
            if result.returncode == 0:
                self.ffmpeg_available = True
                logger.info("[OK] FFmpeg available")
        except:
            logger.warning("[WARNING] FFmpeg not available")
    
    def get_available_methods(self):
        """Get list of available conversion methods"""
        methods = []
        if self.moviepy_available:
            methods.append("moviepy")
        if self.opencv_available:
            methods.append("opencv")
        if self.ffmpeg_available:
            methods.append("ffmpeg")
        return methods
    
    def convert_with_moviepy(self, input_path, output_path, **kwargs):
        """Convert video using MoviePy"""
        if not self.moviepy_available:
            raise ImportError("MoviePy not available")
        
        try:
            import moviepy.editor as mp
            
            logger.info(f"[ACTION] Converting with MoviePy: {input_path}")
            
            # Load video with error handling for problematic formats
            try:
                video = mp.VideoFileClip(input_path)
            except Exception as load_error:
                logger.error(f"[ERROR] MoviePy could not load video: {load_error}")
                # Try with different backend
                try:
                    video = mp.VideoFileClip(input_path, audio=False)  # Try without audio
                    logger.info("[VIDEO] Loaded video without audio track")
                except Exception as load_error2:
                    logger.error(f"[ERROR] MoviePy failed completely: {load_error2}")
                    return False
            
            # Get conversion settings
            fps = kwargs.get('fps', min(24, video.fps or 24))
            quality = kwargs.get('quality', 'medium')
            
            # Quality settings
            if quality == 'high':
                bitrate = "2000k"
                audio_bitrate = "192k"
            elif quality == 'low':
                bitrate = "500k" 
                audio_bitrate = "128k"
            else:  # medium
                bitrate = "1000k"
                audio_bitrate = "128k"
            
            # Convert and save
            video.write_videofile(
                str(output_path),
                fps=fps,
                bitrate=bitrate,
                audio_bitrate=audio_bitrate if video.audio else None,
                codec='libx264',
                audio_codec='aac' if video.audio else None,
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                verbose=False,
                logger=None
            )
            
            video.close()
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] MoviePy conversion failed: {e}")
            return False
    
    def convert_with_opencv(self, input_path, output_path, **kwargs):
        """Convert video using OpenCV"""
        if not self.opencv_available:
            raise ImportError("OpenCV not available")
        
        try:
            import cv2
            import os
            
            logger.info(f"[MOVIE] Converting with OpenCV: {input_path}")
            
            # Open input video
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                logger.error("[ERROR] Could not open input video with OpenCV")
                return False
            
            # Get video properties
            fps = kwargs.get('fps', int(cap.get(cv2.CAP_PROP_FPS)) or 24)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"[INFO] Video properties: {width}x{height}, {fps}fps, {total_frames} frames")
            
            # Try different codecs for better compatibility
            codecs = [
                ('mp4v', 'MP4V'),
                ('XVID', 'XVID'), 
                ('MJPG', 'MJPG'),
                ('X264', 'X264')
            ]
            
            out = None
            for codec_name, codec_desc in codecs:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec_name)
                    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                    
                    # Test if writer is properly initialized
                    if out and out.isOpened():
                        logger.info(f"[OK] Using codec: {codec_desc}")
                        break
                    else:
                        out = None
                except Exception as codec_error:
                    logger.warning(f"[WARNING] Codec {codec_desc} failed: {codec_error}")
                    continue
            
            if not out or not out.isOpened():
                logger.error("[ERROR] Could not initialize video writer with any codec")
                cap.release()
                return False
            
            frame_count = 0
            last_log_frame = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Write frame
                out.write(frame)
                frame_count += 1
                
                # Log progress every 10% or 1000 frames
                if frame_count - last_log_frame >= 1000 or (total_frames > 0 and frame_count % max(1, total_frames // 10) == 0):
                    progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                    logger.info(f"[PROCESSING] Processed {frame_count} frames ({progress:.1f}%)")
                    last_log_frame = frame_count
            
            # Release everything
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            
            # Verify output file was created and has content
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"[OK] OpenCV conversion complete: {frame_count} frames written")
                return True
            else:
                logger.error("[ERROR] Output file was not created or is empty")
                return False
            
        except Exception as e:
            logger.error(f"[ERROR] OpenCV conversion failed: {e}")
            try:
                if 'cap' in locals():
                    cap.release()
                if 'out' in locals():
                    out.release()
                cv2.destroyAllWindows()
            except:
                pass
            return False
    
    def convert_with_ffmpeg(self, input_path, output_path, **kwargs):
        """Convert video using FFmpeg subprocess"""
        if not self.ffmpeg_available:
            raise ImportError("FFmpeg not available")
        
        try:
            import subprocess
            
            logger.info(f"[SETTINGS] Converting with FFmpeg: {input_path}")
            
            quality = kwargs.get('quality', 'medium')
            fps = kwargs.get('fps', 24)
            
            # Quality presets
            if quality == 'high':
                crf = "18"
                preset = "slow"
            elif quality == 'low':
                crf = "28"
                preset = "fast"
            else:  # medium
                crf = "23"
                preset = "medium"
            
            cmd = [
                'ffmpeg', '-y',
                '-i', input_path,
                '-c:v', 'libx264',
                '-preset', preset,
                '-crf', crf,
                '-c:a', 'aac',
                '-b:a', '128k',
                '-movflags', '+faststart',
                '-pix_fmt', 'yuv420p',
                '-r', str(fps),
                '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("[OK] FFmpeg conversion successful")
                return True
            else:
                logger.error(f"[ERROR] FFmpeg error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("[ERROR] FFmpeg conversion timed out")
            return False
        except Exception as e:
            logger.error(f"[ERROR] FFmpeg conversion failed: {e}")
            return False
    
    def convert_video(self, input_path, output_path=None, method='auto', **kwargs):
        """
        Convert video to web-compatible MP4 format
        
        Args:
            input_path: Path to input video file
            output_path: Path for output file (optional)
            method: Conversion method ('auto', 'moviepy', 'opencv', 'ffmpeg')
            **kwargs: Additional options (quality, fps, etc.)
        
        Returns:
            tuple: (success, output_path, message)
        """
        
        if not os.path.exists(input_path):
            return False, None, f"Input file not found: {input_path}"
        
        # Generate output path if not provided
        if output_path is None:
            input_file = Path(input_path)
            output_path = input_file.parent / f"{input_file.stem}_converted.mp4"
        
        output_path = Path(output_path)
        
        # Determine conversion method
        if method == 'auto':
            available = self.get_available_methods()
            if not available:
                return False, None, "No video conversion libraries available. Install: pip install moviepy opencv-python"
            
            # Prefer MoviePy > OpenCV > FFmpeg (easier installation)
            if 'moviepy' in available:
                method = 'moviepy'
            elif 'opencv' in available:
                method = 'opencv'
            elif 'ffmpeg' in available:
                method = 'ffmpeg'
            else:
                method = available[0]
        
        # Validate method availability
        if method == 'moviepy' and not self.moviepy_available:
            return False, None, "MoviePy not available"
        elif method == 'opencv' and not self.opencv_available:
            return False, None, "OpenCV not available"
        elif method == 'ffmpeg' and not self.ffmpeg_available:
            return False, None, "FFmpeg not available"
        
        logger.info(f"[PROCESSING] Starting conversion using {method}")
        start_time = time.time()
        
        try:
            # Perform conversion
            if method == 'moviepy':
                success = self.convert_with_moviepy(input_path, output_path, **kwargs)
            elif method == 'opencv':
                success = self.convert_with_opencv(input_path, output_path, **kwargs)
            elif method == 'ffmpeg':
                success = self.convert_with_ffmpeg(input_path, output_path, **kwargs)
            else:
                return False, None, f"Unknown conversion method: {method}"
            
            elapsed = time.time() - start_time
            
            if success and output_path.exists():
                file_size = output_path.stat().st_size
                logger.info(f"[OK] Conversion completed in {elapsed:.1f}s")
                logger.info(f"[INFO] Output size: {file_size / 1024 / 1024:.1f} MB")
                return True, str(output_path), f"Conversion successful ({elapsed:.1f}s)"
            else:
                return False, None, "Conversion failed - output file not created"
                
        except Exception as e:
            logger.error(f"[ERROR] Conversion error: {e}")
            return False, None, f"Conversion error: {str(e)}"
    
    def get_video_info(self, video_path):
        """Get video file information"""
        info = {
            'file_size': 0,
            'duration': 0,
            'width': 0,
            'height': 0,
            'fps': 0,
            'format': 'unknown'
        }
        
        if not os.path.exists(video_path):
            return info
        
        info['file_size'] = os.path.getsize(video_path)
        
        # Try with MoviePy first
        if self.moviepy_available:
            try:
                import moviepy.editor as mp
                with mp.VideoFileClip(video_path) as video:
                    info['duration'] = video.duration or 0
                    info['width'] = video.w or 0
                    info['height'] = video.h or 0
                    info['fps'] = video.fps or 0
                    info['format'] = 'readable'
                return info
            except:
                pass
        
        # Try with OpenCV
        if self.opencv_available:
            try:
                import cv2
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    info['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    info['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    info['fps'] = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    if info['fps'] > 0:
                        info['duration'] = frame_count / info['fps']
                    info['format'] = 'readable'
                cap.release()
                return info
            except:
                pass
        
        return info

def install_dependencies():
    """Install video processing dependencies"""
    import subprocess
    import sys
    
    packages = [
        'moviepy>=1.0.3',
        'opencv-python>=4.8.0',
        'imageio>=2.31.1',
        'imageio-ffmpeg>=0.4.7',
        'pillow>=9.0.0',
        'numpy>=1.21.0'
    ]
    
    print("[PACKAGE] Installing video processing dependencies...")
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"[OK] {package} installed")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to install {package}: {e}")
    
    print("🎉 Installation complete!")

if __name__ == "__main__":
    # Test the video processor
    processor = VideoProcessor()
    
    print("[MOVIE] Video Processor Test")
    print("=" * 30)
    print(f"Available methods: {processor.get_available_methods()}")
    
    # Test with your video file
    test_file = "../static/uploads/4a5b80c6-1959-4032-8ad1-f375408b1f43_TANG_TRET_84A_Tret_84A_Tret_20250512085459_20250512091458_472401.mp4"
    
    if os.path.exists(test_file):
        print(f"\n[SEARCH] Testing with: {test_file}")
        
        # Get video info
        info = processor.get_video_info(test_file)
        print(f"[INFO] Video info: {info}")
        
        # Test conversion (if libraries available)
        if processor.get_available_methods():
            print("\n[PROCESSING] Testing conversion...")
            success, output_path, message = processor.convert_video(
                test_file, 
                quality='medium'
            )
            print(f"Result: {success}")
            print(f"Output: {output_path}")
            print(f"Message: {message}")
        else:
            print("\n[WARNING] No conversion libraries available")
            print("Run: pip install -r requirements.txt")
    else:
        print(f"\n[WARNING] Test file not found: {test_file}")