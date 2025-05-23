#!/usr/bin/env python3
"""
Quick installer for video processing dependencies
"""
import subprocess
import sys
import importlib

def check_package(package_name):
    """Check if a package is installed"""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("🎥 Video Processing Dependencies Installer")
    print("=" * 50)
    
    # Required packages for video conversion
    packages = [
        ('moviepy', 'moviepy>=1.0.3'),
        ('cv2', 'opencv-python>=4.8.0'),
        ('imageio', 'imageio>=2.31.1'),
        ('imageio_ffmpeg', 'imageio-ffmpeg>=0.4.7'),
        ('PIL', 'pillow>=9.0.0'),
        ('numpy', 'numpy>=1.21.0')
    ]
    
    installed = []
    to_install = []
    
    # Check what's already installed
    print("🔍 Checking current installations...")
    for import_name, pip_name in packages:
        if check_package(import_name):
            print(f"✅ {pip_name} - Already installed")
            installed.append(pip_name)
        else:
            print(f"❌ {pip_name} - Not installed")
            to_install.append(pip_name)
    
    if not to_install:
        print("\n🎉 All video processing dependencies are already installed!")
        return
    
    print(f"\n📦 Installing {len(to_install)} missing packages...")
    
    failed = []
    for package in to_install:
        print(f"\n📥 Installing {package}...")
        if install_package(package):
            print(f"✅ {package} installed successfully")
        else:
            print(f"❌ Failed to install {package}")
            failed.append(package)
    
    print("\n" + "=" * 50)
    print("📊 Installation Summary:")
    print(f"✅ Already installed: {len(installed)}")
    print(f"📥 Successfully installed: {len(to_install) - len(failed)}")
    print(f"❌ Failed to install: {len(failed)}")
    
    if failed:
        print(f"\n⚠️ Failed packages: {', '.join(failed)}")
        print("💡 Try installing manually:")
        for package in failed:
            print(f"   pip install {package}")
    else:
        print("\n🎉 All video processing dependencies installed successfully!")
        print("🚀 You can now convert IMKH and other video formats!")

if __name__ == "__main__":
    main()