#!/usr/bin/env python3
"""
Installation script for AI models and dependencies
Run this to install real person detection models
"""
import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"📋 Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stderr:
            print(f"📋 Error: {e.stderr.strip()}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required for AI models")
        return False
    
    print("✅ Python version is compatible")
    return True

def install_basic_dependencies():
    """Install basic computer vision dependencies"""
    print("📦 Installing basic computer vision dependencies...")
    
    commands = [
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install opencv-python>=4.8.0", "Installing OpenCV"),
        ("pip install numpy>=1.21.0", "Installing NumPy"),
        ("pip install pillow>=9.0.0", "Installing Pillow"),
        ("pip install imageio>=2.31.1", "Installing ImageIO"),
    ]
    
    success = True
    for command, description in commands:
        if not run_command(command, description):
            success = False
    
    return success

def install_pytorch():
    """Install PyTorch for AI models"""
    print("🔥 Installing PyTorch for AI models...")
    
    # Try to detect if CUDA is available
    try:
        import torch
        if torch.cuda.is_available():
            print("🎮 CUDA detected, installing PyTorch with CUDA support")
            command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        else:
            print("💻 No CUDA detected, installing CPU-only PyTorch")
            command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    except ImportError:
        print("💻 Installing CPU-only PyTorch")
        command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    
    return run_command(command, "Installing PyTorch")

def install_transformers():
    """Install Hugging Face Transformers for DETR and other models"""
    print("🤖 Installing Hugging Face Transformers...")
    commands = [
        ("pip install transformers>=4.30.0", "Installing Transformers"),
        ("pip install accelerate>=0.20.0", "Installing Accelerate"),
        ("pip install timm>=0.9.0", "Installing TIMM"),
    ]
    
    success = True
    for command, description in commands:
        if not run_command(command, description):
            success = False
    return success

def install_sam():
    """Install SAM (Segment Anything Model)"""
    print("🎯 Installing SAM (Segment Anything)...")
    commands = [
        ("pip install segment-anything", "Installing SAM"),
        ("pip install git+https://github.com/facebookresearch/sam2.git", "Installing SAM2"),
    ]
    
    success = True
    for command, description in commands:
        if not run_command(command, description):
            print(f"⚠️ {description} failed, continuing...")
    return True  # Don't fail if SAM2 fails

def install_yolo():
    """Install YOLO (Ultralytics) for object detection"""
    print("⚡ Installing YOLO (Ultralytics) for object detection...")
    return run_command("pip install ultralytics>=8.0.0", "Installing YOLO")

def install_onnx():
    """Install ONNX Runtime for model inference"""
    print("⚙️ Installing ONNX Runtime...")
    return run_command("pip install onnxruntime>=1.15.0", "Installing ONNX Runtime")

def install_mediapipe():
    """Install MediaPipe for lightweight detection"""
    print("🎨 Installing MediaPipe for lightweight detection...")
    return run_command("pip install mediapipe>=0.10.0", "Installing MediaPipe")

def download_models():
    """Download models to the correct models/ directory"""
    print("📥 Downloading AI models to models/ directory...")
    
    # Ensure models directory exists
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    try:
        # Download YOLO model to correct location
        yolo_path = os.path.join(models_dir, 'yolov8n.pt')
        if not os.path.exists(yolo_path):
            print("📥 Downloading YOLOv8n model...")
            from ultralytics import YOLO
            model = YOLO('yolov8n.pt')  # This downloads it
            # Move from default location to our models dir
            import shutil
            default_path = 'yolov8n.pt'
            if os.path.exists(default_path):
                shutil.move(default_path, yolo_path)
                print(f"✅ YOLOv8n model saved to {yolo_path}")
        else:
            print(f"✅ YOLOv8n model already exists at {yolo_path}")
            
        return True
    except Exception as e:
        print(f"⚠️ Model download failed: {e}")
        return False

def test_installations():
    """Test if the installations work"""
    print("🧪 Testing AI model installations...")
    
    tests = [
        ("import cv2; print(f'OpenCV: {cv2.__version__}')", "OpenCV"),
        ("import torch; print(f'PyTorch: {torch.__version__}')", "PyTorch"),
        ("import torchvision; print(f'TorchVision: {torchvision.__version__}')", "TorchVision"),
        ("from ultralytics import YOLO; print('YOLO: Available')", "YOLO"),
        ("import onnxruntime; print(f'ONNX Runtime: {onnxruntime.__version__}')", "ONNX"),
        ("import mediapipe; print(f'MediaPipe: {mediapipe.__version__}')", "MediaPipe"),
    ]
    
    results = {}
    for test_code, name in tests:
        try:
            result = subprocess.run([sys.executable, "-c", test_code], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print(f"✅ {name}: {result.stdout.strip()}")
                results[name] = True
            else:
                print(f"❌ {name}: Failed to import")
                results[name] = False
        except Exception as e:
            print(f"❌ {name}: Error - {e}")
            results[name] = False
    
    return results

def download_sample_models():
    """Download sample models for testing"""
    print("📥 Downloading sample models...")
    
    try:
        # Test YOLO model download
        test_code = """
from ultralytics import YOLO
import os
print("📥 Downloading YOLOv8 nano model...")
model = YOLO('yolov8n.pt')
print("✅ YOLOv8 model downloaded successfully")
print(f"📁 Model saved to: {model.model_path if hasattr(model, 'model_path') else 'default location'}")
"""
        result = subprocess.run([sys.executable, "-c", test_code], 
                              capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✅ Sample YOLO model downloaded")
            print(result.stdout)
        else:
            print("⚠️ Failed to download YOLO model")
            print(result.stderr)
    
    except Exception as e:
        print(f"⚠️ Error downloading models: {e}")

def main():
    """Main installation process"""
    print("🚀 AI Models Installation Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies step by step
    print("\n📦 Step 1: Installing basic dependencies...")
    if not install_basic_dependencies():
        print("❌ Basic dependencies installation failed")
        sys.exit(1)
    
    print("\n🔥 Step 2: Installing PyTorch...")
    pytorch_success = install_pytorch()
    
    print("\n🤖 Step 3: Installing Transformers (DETR)...")
    transformers_success = install_transformers()
    
    print("\n🎯 Step 4: Installing SAM/SAM2...")
    sam_success = install_sam()
    
    print("\n⚡ Step 5: Installing YOLO...")
    yolo_success = install_yolo()
    
    print("\n⚙️ Step 6: Installing ONNX Runtime...")
    onnx_success = install_onnx()
    
    print("\n🎨 Step 7: Installing MediaPipe...")
    mediapipe_success = install_mediapipe()
    
    print("\n🧪 Step 6: Testing installations...")
    test_results = test_installations()
    
    print("\n📥 Step 7: Downloading models to correct directory...")
    if yolo_success and test_results.get('YOLO', False):
        download_models()
        download_sample_models()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Installation Summary:")
    
    available_models = []
    if test_results.get('OpenCV', False):
        available_models.append("OpenCV (HOG)")
    if test_results.get('PyTorch', False) and test_results.get('TorchVision', False):
        available_models.append("PyTorch (Faster R-CNN)")
    if test_results.get('YOLO', False):
        available_models.append("YOLO (YOLOv8)")
    if test_results.get('ONNX', False):
        available_models.append("ONNX Runtime")
    if test_results.get('MediaPipe', False):
        available_models.append("MediaPipe (Pose)")
    
    if available_models:
        print("✅ Available AI models:")
        for model in available_models:
            print(f"   • {model}")
        print(f"\n🎉 {len(available_models)} AI model(s) ready for use!")
        print("\n🚀 You can now run real person detection instead of mock data!")
        print("\nNext steps:")
        print("1. Restart your Flask application")
        print("2. Try processing a video to see real AI detection in action")
        print("3. Check the logs for 'Using REAL AI model' messages")
    else:
        print("❌ No AI models successfully installed")
        print("The system will continue to use mock detection")
    
    print("\n🔧 To check which models are available, look for these log messages:")
    print("   ✅ OpenCV available")
    print("   ✅ PyTorch available") 
    print("   ✅ YOLO (Ultralytics) available")
    print("   🚀 Using REAL AI model: [model_name]")

if __name__ == "__main__":
    main()