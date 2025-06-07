#!/usr/bin/env python3
"""
Video Converter for IMKH and other non-standard formats
Converts videos to web-compatible MP4 format
"""
import os
import subprocess
import sys
from pathlib import Path

def check_ffmpeg():
    """Check if FFmpeg is available"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("[OK] FFmpeg is available")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    print("[ERROR] FFmpeg not found")
    print("[TIP] Install FFmpeg:")
    print("   - Windows: Download from https://ffmpeg.org/download.html")
    print("   - Ubuntu/Debian: sudo apt install ffmpeg")
    print("   - macOS: brew install ffmpeg")
    return False

def convert_video(input_path, output_path=None, force_overwrite=False):
    """Convert video to web-compatible MP4 format"""
    
    if not os.path.exists(input_path):
        print(f"[ERROR] Input file not found: {input_path}")
        return False
    
    if output_path is None:
        # Generate output filename
        input_file = Path(input_path)
        output_path = input_file.parent / f"{input_file.stem}_converted.mp4"
    
    output_path = Path(output_path)
    
    if output_path.exists() and not force_overwrite:
        print(f"[WARNING] Output file already exists: {output_path}")
        print("   Use --force to overwrite")
        return False
    
    print(f"[PROCESSING] Converting: {input_path}")
    print(f"[FILE] Output: {output_path}")
    
    # FFmpeg command for web-compatible MP4
    cmd = [
        'ffmpeg',
        '-i', str(input_path),           # Input file
        '-c:v', 'libx264',               # Video codec: H.264
        '-preset', 'medium',             # Encoding speed vs compression
        '-crf', '23',                    # Quality (lower = better, 18-28 range)
        '-c:a', 'aac',                   # Audio codec: AAC
        '-b:a', '128k',                  # Audio bitrate
        '-movflags', '+faststart',       # Web optimization
        '-pix_fmt', 'yuv420p',          # Pixel format for compatibility
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',  # Ensure even dimensions
    ]
    
    if force_overwrite:
        cmd.append('-y')
    
    cmd.append(str(output_path))
    
    try:
        print("[WAIT] Starting conversion...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            if output_path.exists():
                file_size = output_path.stat().st_size
                print(f"[OK] Conversion successful!")
                print(f"[INFO] Output size: {file_size / 1024 / 1024:.1f} MB")
                return True
            else:
                print("[ERROR] Conversion failed: Output file not created")
                return False
        else:
            print(f"[ERROR] FFmpeg error (code {result.returncode}):")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("[ERROR] Conversion timed out")
        return False
    except Exception as e:
        print(f"[ERROR] Conversion error: {e}")
        return False

def analyze_video(video_path):
    """Analyze video file properties"""
    print(f"[SEARCH] Analyzing: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"[ERROR] File not found: {video_path}")
        return
    
    # File size
    file_size = os.path.getsize(video_path)
    print(f"[FILE] File size: {file_size:,} bytes ({file_size / 1024 / 1024:.1f} MB)")
    
    # File header
    try:
        with open(video_path, 'rb') as f:
            header = f.read(32)
            print(f"📄 File header: {header.hex()}")
            print(f"📄 Header (ASCII): {header.decode('ascii', errors='ignore')}")
            
            # Check for known formats
            if header.startswith(b'IMKH'):
                print("[SEARCH] Format detected: IMKH (proprietary format)")
                print("[TIP] This format may not be supported by web browsers")
            elif header[4:8] == b'ftyp':
                print("[SEARCH] Format detected: MP4/MOV")
            elif header.startswith(b'RIFF'):
                print("[SEARCH] Format detected: AVI")
            elif header.startswith(b'\x1a\x45\xdf\xa3'):
                print("[SEARCH] Format detected: MKV/WebM")
            else:
                print("[SEARCH] Format: Unknown/Unrecognized")
    except Exception as e:
        print(f"[ERROR] Error reading file header: {e}")
    
    # Try to get video info with FFmpeg
    if check_ffmpeg():
        try:
            cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', video_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                
                if 'format' in data:
                    format_info = data['format']
                    print(f"[ACTION] Duration: {float(format_info.get('duration', 0)):.1f} seconds")
                    print(f"[ACTION] Bitrate: {int(format_info.get('bit_rate', 0)) // 1000} kbps")
                
                video_streams = [s for s in data.get('streams', []) if s.get('codec_type') == 'video']
                if video_streams:
                    video = video_streams[0]
                    print(f"📺 Video codec: {video.get('codec_name', 'Unknown')}")
                    print(f"📺 Resolution: {video.get('width', '?')}x{video.get('height', '?')}")
                    print(f"📺 FPS: {eval(video.get('r_frame_rate', '0/1')):.2f}")
                
                audio_streams = [s for s in data.get('streams', []) if s.get('codec_type') == 'audio']
                if audio_streams:
                    audio = audio_streams[0]
                    print(f"🔊 Audio codec: {audio.get('codec_name', 'Unknown')}")
                    print(f"🔊 Sample rate: {audio.get('sample_rate', '?')} Hz")
            else:
                print("[WARNING] FFprobe could not analyze the file (unsupported format)")
                
        except Exception as e:
            print(f"[WARNING] Error analyzing with FFprobe: {e}")

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("[MOVIE] Video Converter for Web Compatibility")
        print("=" * 50)
        print("Usage:")
        print("  python video_converter.py <input_file> [output_file] [--force]")
        print("  python video_converter.py --analyze <input_file>")
        print()
        print("Examples:")
        print("  python video_converter.py video.mp4")
        print("  python video_converter.py video.mp4 converted.mp4 --force")
        print("  python video_converter.py --analyze video.mp4")
        return
    
    if sys.argv[1] == '--analyze':
        if len(sys.argv) >= 3:
            analyze_video(sys.argv[2])
        else:
            print("[ERROR] Please specify a file to analyze")
        return
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) >= 3 and not sys.argv[2].startswith('--') else None
    force = '--force' in sys.argv
    
    if not check_ffmpeg():
        return
    
    print("[MOVIE] Video Converter")
    print("=" * 30)
    
    # Analyze input first
    analyze_video(input_file)
    print()
    
    # Convert
    success = convert_video(input_file, output_file, force)
    
    if success:
        print()
        print("🎉 Conversion completed successfully!")
        print("[TIP] You can now use the converted file in your web application")
    else:
        print()
        print("[ERROR] Conversion failed")

if __name__ == "__main__":
    main()