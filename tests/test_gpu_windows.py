#!/usr/bin/env python
"""
Quick GPU test script for Windows
Tests if GPU acceleration is properly configured
"""

import sys
import platform
import time

print("=" * 60)
print("GPU Configuration Test for Windows")
print("=" * 60)
print()

# System Information
print("System Information:")
print(f"  Platform: {platform.system()} {platform.release()}")
print(f"  Python: {sys.version.split()[0]}")
print()

# Check PyTorch
try:
    import torch
    print("PyTorch Status:")
    print(f"  Version: {torch.__version__}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"    Compute Capability: {props.major}.{props.minor}")
        
        # Quick performance test
        print("\nRunning quick performance test...")
        size = 5000
        
        # CPU test
        start = time.time()
        a_cpu = torch.randn(size, size)
        b_cpu = torch.randn(size, size)
        c_cpu = torch.matmul(a_cpu, b_cpu)
        cpu_time = time.time() - start
        
        # GPU test
        device = torch.device('cuda')
        start = time.time()
        a_gpu = a_cpu.to(device)
        b_gpu = b_cpu.to(device)
        c_gpu = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        print(f"\nMatrix multiplication ({size}x{size}):")
        print(f"  CPU Time: {cpu_time:.3f}s")
        print(f"  GPU Time: {gpu_time:.3f}s")
        print(f"  Speedup: {cpu_time/gpu_time:.2f}x")
    else:
        print("\n[WARNING]  CUDA is not available!")
        print("  This usually means:")
        print("  1. PyTorch CPU-only version is installed")
        print("  2. CUDA toolkit is not installed")
        print("  3. Incompatible CUDA/GPU driver versions")
        print("\n  Run install_gpu_windows.bat to fix this.")
        
except ImportError:
    print("[ERROR] PyTorch is not installed")
    print()

# Check OpenCV
print("\n" + "-" * 40)
try:
    import cv2
    print("OpenCV Status:")
    print(f"  Version: {cv2.__version__}")
    
    # Check for CUDA support
    try:
        cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
        print(f"  CUDA Devices: {cuda_count}")
    except:
        print("  CUDA Support: Not available")
        
except ImportError:
    print("[ERROR] OpenCV is not installed")

# Check YOLO/Ultralytics
print("\n" + "-" * 40)
try:
    from ultralytics import YOLO
    import ultralytics
    print("Ultralytics YOLO Status:")
    print(f"  Version: {ultralytics.__version__}")
    
    # Test YOLO with GPU
    if torch.cuda.is_available():
        print("  Testing YOLO inference...")
        model = YOLO('yolov8n.pt')
        
        # Create dummy image
        import numpy as np
        dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # CPU inference
        start = time.time()
        _ = model(dummy_img, device='cpu', verbose=False)
        cpu_time = time.time() - start
        
        # GPU inference
        start = time.time()
        _ = model(dummy_img, device=0, verbose=False)
        gpu_time = time.time() - start
        
        print(f"  CPU Inference: {cpu_time:.3f}s")
        print(f"  GPU Inference: {gpu_time:.3f}s")
        print(f"  Speedup: {cpu_time/gpu_time:.2f}x")
        
except ImportError:
    print("[ERROR] Ultralytics is not installed")

print("\n" + "=" * 60)

# Summary and recommendations
if 'torch' in sys.modules and torch.cuda.is_available():
    print("[OK] GPU acceleration is properly configured!")
    print("   Your RTX 4070 Ti SUPER is ready for processing.")
else:
    print("[WARNING]  GPU acceleration is NOT configured.")
    print("\nTo enable GPU support:")
    print("1. Run: install_gpu_windows.bat")
    print("2. Restart your application")
    print("3. Run this test again")

print("=" * 60)