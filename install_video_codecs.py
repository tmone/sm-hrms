#!/usr/bin/env python3
"""
Install video codecs for better H.264 support
"""
import subprocess
import sys
import platform

def install_codecs():
    """Install necessary packages for H.264 codec support"""
    
    print("🎬 Installing video codec support for H.264...")
    
    packages = [
        'opencv-python-headless',  # OpenCV without GUI dependencies
        'opencv-contrib-python',   # Additional OpenCV modules
        'imageio-ffmpeg',         # FFmpeg bindings
        'ffmpeg-python'           # Python FFmpeg wrapper
    ]
    
    for package in packages:
        print(f"\n📦 Installing {package}...")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '--upgrade', package
            ])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
    
    # Platform-specific instructions
    system = platform.system()
    print(f"\n💻 Platform: {system}")
    
    if system == "Windows":
        print("""
📌 For Windows, you may also need to:
1. Download FFmpeg from https://ffmpeg.org/download.html
2. Extract it to C:\\ffmpeg
3. Add C:\\ffmpeg\\bin to your PATH environment variable
4. Restart your Python environment

Alternatively, install using chocolatey:
  choco install ffmpeg
        """)
    elif system == "Linux":
        print("""
📌 For Linux, install system packages:
  sudo apt-get update
  sudo apt-get install ffmpeg libavcodec-dev libavformat-dev
        """)
    elif system == "Darwin":  # macOS
        print("""
📌 For macOS, install using Homebrew:
  brew install ffmpeg
        """)
    
    print("\n✅ Codec installation guide complete!")
    print("💡 After installing FFmpeg, restart your application for H.264 support")

if __name__ == '__main__':
    install_codecs()