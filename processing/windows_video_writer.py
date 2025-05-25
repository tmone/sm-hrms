"""
Windows-compatible video writer that avoids OpenH264 issues
"""
import cv2
import platform
from pathlib import Path

def get_working_codec(output_path, fps, width, height):
    """
    Get a video codec that works on Windows without OpenH264
    
    Returns:
        tuple: (VideoWriter object, actual_output_path)
    """
    system = platform.system()
    
    if system == "Windows":
        # On Windows, try web-compatible codecs
        codecs_to_try = [
            # (fourcc, extension, description)
            ('H264', '.mp4', 'H.264'),     # Try H.264 first
            ('avc1', '.mp4', 'H.264/AVC'), # Apple's H.264
            ('XVID', '.avi', 'Xvid'),      # Good compression, widely supported
            ('mp4v', '.mp4', 'MPEG-4'),    # Fallback
        ]
    else:
        # On Linux/Mac, H264 usually works
        codecs_to_try = [
            ('avc1', '.mp4', 'H.264/AVC'),
            ('H264', '.mp4', 'H.264'),
            ('mp4v', '.mp4', 'MPEG-4'),
            ('XVID', '.avi', 'Xvid'),
        ]
    
    base_path = str(output_path).rsplit('.', 1)[0]
    
    for fourcc_str, ext, description in codecs_to_try:
        try:
            output_file = base_path + ext
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height), True)
            
            if writer.isOpened():
                print(f"✅ Using {description} codec ({fourcc_str})")
                print(f"📁 Output file: {output_file}")
                
                # Set quality if possible (lower value = smaller file size)
                try:
                    writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 70)  # Reduced from 90 for smaller files
                except:
                    pass
                
                return writer, Path(output_file)
            else:
                writer.release()
        except Exception as e:
            print(f"⚠️  Failed to initialize {description}: {e}")
            continue
    
    # If all fail, raise exception
    raise Exception("Could not initialize any video codec")

def create_windows_compatible_writer(output_path, fps, width, height):
    """
    Create a video writer that works on Windows
    """
    # Always use .mp4 extension for the path
    output_path = Path(str(output_path).replace('.avi', '.mp4'))
    
    print(f"🎥 Creating video writer for {width}x{height} @ {fps}fps")
    print(f"🖥️  Platform: {platform.system()}")
    
    try:
        writer, actual_path = get_working_codec(output_path, fps, width, height)
        return writer, actual_path
    except Exception as e:
        print(f"❌ Failed to create video writer: {e}")
        raise