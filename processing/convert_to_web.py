"""
Convert video to web-compatible format using FFmpeg
"""
import subprocess
import os
from pathlib import Path

def convert_video_to_web_format(input_path, output_path=None):
    """
    Convert video to H.264 format that works in all browsers
    
    Args:
        input_path: Path to input video
        output_path: Path to output video (optional)
    
    Returns:
        Path to converted video
    """
    input_path = Path(input_path)
    
    if output_path is None:
        # Create output path with _web suffix
        output_path = input_path.parent / f"{input_path.stem}_web.mp4"
    else:
        output_path = Path(output_path)
    
    # Check if FFmpeg is available
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True)
        if result.returncode != 0:
            print("[ERROR] FFmpeg not found. Please install FFmpeg for web-compatible video conversion.")
            return None
    except:
        print("[ERROR] FFmpeg not found. Please install FFmpeg.")
        print("   Windows: Download from https://ffmpeg.org/download.html")
        print("   Or use: winget install ffmpeg")
        return None
    
    # Get input file size to calculate target bitrate
    input_size_mb = input_path.stat().st_size / (1024 * 1024)
    
    # FFmpeg command for web-compatible video with size optimization
    # Using libx264 with specific settings for browser compatibility and smaller size
    cmd = [
        'ffmpeg',
        '-i', str(input_path),      # Input file
        '-c:v', 'libx264',          # Use H.264 codec
        '-preset', 'medium',        # Encoding speed/quality tradeoff
        '-crf', '25',              # Quality (increased to 25 for smaller size)
        '-maxrate', '1500k',       # Maximum bitrate constraint
        '-bufsize', '3000k',       # Buffer size for rate control
        '-pix_fmt', 'yuv420p',     # Pixel format for compatibility
        '-movflags', '+faststart',  # Enable streaming
        '-c:a', 'aac',            # Audio codec
        '-b:a', '96k',            # Audio bitrate (reduced for smaller size)
        '-y',                      # Overwrite output
        str(output_path)
    ]
    
    print(f"[ACTION] Converting video to web format...")
    print(f"[FILE] Input: {input_path}")
    print(f"[FILE] Output: {output_path}")
    
    try:
        # Run FFmpeg
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Monitor progress
        for line in process.stderr:
            if 'frame=' in line:
                # Extract progress info
                print(f"\r{line.strip()}", end='')
        
        process.wait()
        
        if process.returncode == 0:
            # Get file sizes
            input_size = input_path.stat().st_size / (1024 * 1024)
            output_size = output_path.stat().st_size / (1024 * 1024)
            
            print(f"\n[OK] Conversion successful!")
            print(f"[INFO] Input size: {input_size:.1f} MB")
            print(f"[INFO] Output size: {output_size:.1f} MB")
            print(f"[INFO] Compression ratio: {input_size/output_size:.1f}x")
            
            return output_path
        else:
            print(f"\n[ERROR] Conversion failed with code {process.returncode}")
            return None
            
    except Exception as e:
        print(f"\n[ERROR] Error during conversion: {e}")
        return None

def batch_convert_to_web(directory, pattern="*_annotated_*.mp4"):
    """
    Convert all annotated videos in a directory to web format
    """
    directory = Path(directory)
    videos = list(directory.glob(pattern))
    
    if not videos:
        print(f"No videos found matching pattern: {pattern}")
        return
    
    print(f"Found {len(videos)} videos to convert")
    
    converted = []
    for video in videos:
        print(f"\n{'='*60}")
        result = convert_video_to_web_format(video)
        if result:
            converted.append(result)
    
    print(f"\n{'='*60}")
    print(f"[OK] Converted {len(converted)}/{len(videos)} videos successfully")
    
    return converted

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Convert single video
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        
        result = convert_video_to_web_format(input_path, output_path)
        if result:
            print(f"\n[OK] Web-compatible video: {result}")
    else:
        # Convert all videos in outputs directory
        print("Converting all annotated videos to web format...")
        batch_convert_to_web("processing/outputs")