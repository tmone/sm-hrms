#!/usr/bin/env python3
"""
Setup GPU support for video processing
"""
import subprocess
import sys
import platform
import os

def check_nvidia_gpu():
    """Check if NVIDIA GPU is available"""
    print("🔍 Checking for NVIDIA GPU...")
    
    try:
        # Try nvidia-smi command
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected!")
            print(result.stdout.split('\n')[0:10])  # Show first 10 lines
            return True
    except FileNotFoundError:
        print("❌ nvidia-smi not found. NVIDIA drivers may not be installed.")
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
    
    return False

def check_cuda_version():
    """Check CUDA version"""
    print("\n🔍 Checking CUDA version...")
    
    try:
        # Check nvcc version
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ CUDA compiler found:")
            print(result.stdout)
            
            # Extract CUDA version
            for line in result.stdout.split('\n'):
                if 'release' in line:
                    cuda_version = line.split('release')[1].split(',')[0].strip()
                    return cuda_version
    except FileNotFoundError:
        print("❌ nvcc not found. CUDA toolkit may not be installed.")
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")
    
    return None

def install_pytorch_cuda():
    """Install PyTorch with CUDA support"""
    print("\n📦 Installing PyTorch with CUDA support...")
    
    system = platform.system()
    
    # Determine CUDA version and PyTorch installation command
    cuda_versions = {
        '11.8': 'cu118',
        '11.7': 'cu117',
        '11.6': 'cu116',
        '12.1': 'cu121',
        '12.4': 'cu124'
    }
    
    # Check current CUDA version
    cuda_version = check_cuda_version()
    
    if cuda_version:
        # Extract major.minor version
        cuda_major_minor = '.'.join(cuda_version.split('.')[:2])
        cuda_suffix = cuda_versions.get(cuda_major_minor, 'cu118')  # Default to 11.8
    else:
        print("⚠️ CUDA not detected. Installing CPU-only PyTorch.")
        cuda_suffix = 'cpu'
    
    # PyTorch installation commands
    if cuda_suffix == 'cpu':
        packages = ['torch', 'torchvision', 'torchaudio']
    else:
        # Install with specific CUDA version
        if system == "Windows":
            packages = [
                f'torch==2.1.0+{cuda_suffix}',
                f'torchvision==0.16.0+{cuda_suffix}',
                f'torchaudio==2.1.0+{cuda_suffix}'
            ]
            index_url = f'https://download.pytorch.org/whl/{cuda_suffix}'
        else:
            packages = ['torch', 'torchvision', 'torchaudio']
            index_url = f'https://download.pytorch.org/whl/{cuda_suffix}'
    
    # Install PyTorch
    try:
        if cuda_suffix != 'cpu' and 'index_url' in locals():
            cmd = [sys.executable, '-m', 'pip', 'install'] + packages + ['--index-url', index_url]
        else:
            cmd = [sys.executable, '-m', 'pip', 'install'] + packages
        
        print(f"Running: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        print("✅ PyTorch installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install PyTorch: {e}")
        return False

def install_gpu_packages():
    """Install all necessary packages for GPU video processing"""
    print("\n📦 Installing GPU-accelerated packages...")
    
    packages = [
        # Core GPU packages
        'opencv-python-headless',    # OpenCV without GUI
        'opencv-contrib-python',     # Additional OpenCV modules
        
        # Deep learning models
        'ultralytics',              # YOLOv8 for object detection
        'supervision',              # Video processing utilities
        
        # GPU acceleration
        'cupy-cuda11x',            # GPU arrays (will auto-select CUDA version)
        'pycuda',                  # CUDA Python bindings
        
        # Video processing
        'imageio-ffmpeg',          # FFmpeg bindings
        'ffmpeg-python',           # FFmpeg wrapper
        'decord',                  # GPU video reader
        
        # Additional utilities
        'nvidia-ml-py3',           # NVIDIA Management Library
        'gpustat',                 # GPU monitoring
        'py3nvml'                  # NVIDIA ML Python bindings
    ]
    
    failed_packages = []
    
    for package in packages:
        print(f"\n📦 Installing {package}...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"⚠️ Failed to install {package}")
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n⚠️ Failed packages: {', '.join(failed_packages)}")
        print("These packages may require additional system dependencies.")
    
    return len(failed_packages) == 0

def setup_opencv_cuda():
    """Instructions for OpenCV with CUDA support"""
    print("\n🔧 OpenCV CUDA Setup Instructions:")
    
    system = platform.system()
    
    if system == "Windows":
        print("""
For Windows - OpenCV with CUDA:

Option 1: Pre-built OpenCV with CUDA (Recommended)
1. Download from: https://github.com/opencv/opencv/releases
2. Choose opencv-*-cuda*.exe version
3. Extract and add to Python:
   pip uninstall opencv-python opencv-contrib-python
   cd <opencv_extract_path>/python
   pip install opencv_python-*.whl

Option 2: Build from source (Advanced)
1. Install Visual Studio 2019/2022
2. Install CUDA Toolkit from NVIDIA
3. Install cuDNN
4. Build OpenCV with CUDA support
   - Guide: https://docs.opencv.org/master/d5/de5/tutorial_py_setup_in_windows.html
""")
    elif system == "Linux":
        print("""
For Linux - OpenCV with CUDA:

# Install dependencies
sudo apt-get update
sudo apt-get install -y build-essential cmake git pkg-config
sudo apt-get install -y libjpeg-dev libtiff-dev libpng-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install -y libgtk2.0-dev libcanberra-gtk-module
sudo apt-get install -y python3-dev python3-numpy

# Build OpenCV with CUDA
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd opencv
mkdir build && cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D WITH_CUDA=ON \
      -D WITH_CUDNN=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D WITH_CUBLAS=1 \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      ..

make -j$(nproc)
sudo make install
""")

def test_gpu_setup():
    """Test if GPU setup is working"""
    print("\n🧪 Testing GPU Setup...")
    
    # Test PyTorch CUDA
    try:
        import torch
        print(f"\n✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ GPU count: {torch.cuda.device_count()}")
            print(f"✅ GPU name: {torch.cuda.get_device_name(0)}")
            print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    except ImportError:
        print("❌ PyTorch not installed")
    except Exception as e:
        print(f"❌ PyTorch error: {e}")
    
    # Test OpenCV
    try:
        import cv2
        print(f"\n✅ OpenCV version: {cv2.__version__}")
        print(f"✅ CUDA support in OpenCV: {cv2.cuda.getCudaEnabledDeviceCount() > 0}")
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            print(f"✅ CUDA devices in OpenCV: {cv2.cuda.getCudaEnabledDeviceCount()}")
    except AttributeError:
        print("⚠️ OpenCV installed but without CUDA support")
    except ImportError:
        print("❌ OpenCV not installed")
    except Exception as e:
        print(f"❌ OpenCV error: {e}")
    
    # Test YOLO
    try:
        from ultralytics import YOLO
        print("\n✅ YOLOv8 (ultralytics) installed")
        # Try to load a model
        model = YOLO('yolov8n.pt')
        print("✅ YOLOv8 nano model loaded successfully")
    except ImportError:
        print("❌ Ultralytics (YOLOv8) not installed")
    except Exception as e:
        print(f"⚠️ YOLOv8 warning: {e}")

def main():
    """Main setup function"""
    print("🚀 GPU Video Processing Setup")
    print("=" * 50)
    
    system = platform.system()
    print(f"💻 Operating System: {system}")
    print(f"🐍 Python Version: {sys.version}")
    
    # Check for NVIDIA GPU
    has_gpu = check_nvidia_gpu()
    
    if not has_gpu:
        print("\n⚠️ No NVIDIA GPU detected. You can still use CPU processing.")
        response = input("\nContinue with CPU-only setup? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Check CUDA
    cuda_version = check_cuda_version()
    
    if has_gpu and not cuda_version:
        print("\n⚠️ CUDA not found. Please install CUDA Toolkit from:")
        print("   https://developer.nvidia.com/cuda-downloads")
        print("\n   Recommended: CUDA 11.8 or 12.1")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Install PyTorch
    install_pytorch_cuda()
    
    # Install other packages
    install_gpu_packages()
    
    # OpenCV CUDA instructions
    setup_opencv_cuda()
    
    # Test setup
    test_gpu_setup()
    
    print("\n✅ Setup complete!")
    print("\n📌 Next steps:")
    print("1. Restart your Python environment")
    print("2. Run: python test_gpu_performance.py")
    print("3. Process a video to test GPU acceleration")

if __name__ == '__main__':
    main()