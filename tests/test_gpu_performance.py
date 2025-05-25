#!/usr/bin/env python3
"""
Test GPU performance for video processing
"""
import time
import numpy as np
import sys

def test_pytorch_gpu():
    """Test PyTorch GPU performance"""
    print("\n🧪 Testing PyTorch GPU Performance...")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("❌ CUDA not available in PyTorch")
            return
        
        # Matrix multiplication benchmark
        sizes = [1024, 2048, 4096]
        
        for size in sizes:
            print(f"\n📊 Matrix multiplication {size}x{size}:")
            
            # CPU test
            a_cpu = torch.randn(size, size)
            b_cpu = torch.randn(size, size)
            
            start = time.time()
            c_cpu = torch.matmul(a_cpu, b_cpu)
            cpu_time = time.time() - start
            print(f"   CPU time: {cpu_time:.3f}s")
            
            # GPU test
            a_gpu = a_cpu.cuda()
            b_gpu = b_cpu.cuda()
            torch.cuda.synchronize()  # Ensure transfer is complete
            
            start = time.time()
            c_gpu = torch.matmul(a_gpu, b_gpu)
            torch.cuda.synchronize()  # Ensure computation is complete
            gpu_time = time.time() - start
            print(f"   GPU time: {gpu_time:.3f}s")
            print(f"   Speedup: {cpu_time/gpu_time:.1f}x")
            
        # Memory info
        print(f"\n💾 GPU Memory:")
        print(f"   Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"   Cached: {torch.cuda.memory_cached() / 1024**2:.1f} MB")
        
    except ImportError:
        print("❌ PyTorch not installed")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_opencv_gpu():
    """Test OpenCV GPU performance"""
    print("\n🧪 Testing OpenCV GPU Performance...")
    
    try:
        import cv2
        
        # Check CUDA support
        try:
            cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
            if cuda_count == 0:
                print("❌ No CUDA devices found in OpenCV")
                print("💡 OpenCV may need to be rebuilt with CUDA support")
                return
            
            print(f"✅ CUDA devices found: {cuda_count}")
            
            # Create test image
            img_size = (1920, 1080)
            img = np.random.randint(0, 255, (*img_size, 3), dtype=np.uint8)
            
            # Test Gaussian blur
            print("\n📊 Gaussian Blur Performance:")
            
            # CPU test
            start = time.time()
            for _ in range(10):
                blur_cpu = cv2.GaussianBlur(img, (31, 31), 0)
            cpu_time = time.time() - start
            print(f"   CPU time (10 iterations): {cpu_time:.3f}s")
            
            # GPU test
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img)
            
            start = time.time()
            for _ in range(10):
                gpu_blur = cv2.cuda.bilateralFilter(gpu_img, -1, 50, 50)
            cv2.cuda.Stream.waitForCompletion(cv2.cuda.Stream_Null())
            gpu_time = time.time() - start
            print(f"   GPU time (10 iterations): {gpu_time:.3f}s")
            print(f"   Speedup: {cpu_time/gpu_time:.1f}x")
            
        except AttributeError:
            print("⚠️ OpenCV installed but without CUDA support")
            print("💡 Install OpenCV with CUDA using setup_gpu_processing.py")
            
    except ImportError:
        print("❌ OpenCV not installed")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_yolo_gpu():
    """Test YOLO GPU performance"""
    print("\n🧪 Testing YOLO GPU Performance...")
    
    try:
        from ultralytics import YOLO
        import torch
        
        # Load model
        print("📥 Loading YOLOv8 model...")
        model = YOLO('yolov8n.pt')  # Nano model
        
        # Create test image
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # CPU inference
        print("\n📊 YOLO Inference Performance:")
        model.to('cpu')
        
        start = time.time()
        for _ in range(10):
            results = model(test_image, verbose=False)
        cpu_time = time.time() - start
        print(f"   CPU time (10 frames): {cpu_time:.3f}s")
        print(f"   CPU FPS: {10/cpu_time:.1f}")
        
        # GPU inference (if available)
        if torch.cuda.is_available():
            model.to('cuda')
            
            # Warmup
            model(test_image, verbose=False)
            
            start = time.time()
            for _ in range(10):
                results = model(test_image, verbose=False)
            gpu_time = time.time() - start
            print(f"   GPU time (10 frames): {gpu_time:.3f}s")
            print(f"   GPU FPS: {10/gpu_time:.1f}")
            print(f"   Speedup: {cpu_time/gpu_time:.1f}x")
        else:
            print("   ⚠️ GPU not available for YOLO")
            
    except ImportError:
        print("❌ Ultralytics (YOLO) not installed")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_video_processing():
    """Test video processing with GPU"""
    print("\n🧪 Testing Video Processing...")
    
    try:
        import cv2
        import torch
        from ultralytics import YOLO
        
        # Create a test video
        print("📹 Creating test video...")
        width, height = 640, 480
        fps = 30
        frames = 100
        
        # Generate test video in memory
        test_frames = []
        for i in range(frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            # Add moving rectangle
            x = int((i / frames) * width)
            cv2.rectangle(frame, (x-50, 200), (x+50, 280), (0, 255, 0), -1)
            cv2.putText(frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            test_frames.append(frame)
        
        # Process with YOLO
        model = YOLO('yolov8n.pt')
        
        # CPU processing
        print("\n📊 Video Processing Performance:")
        model.to('cpu')
        
        start = time.time()
        for frame in test_frames[:30]:  # Process 30 frames
            results = model(frame, verbose=False)
        cpu_time = time.time() - start
        print(f"   CPU processing (30 frames): {cpu_time:.3f}s")
        print(f"   CPU FPS: {30/cpu_time:.1f}")
        
        # GPU processing
        if torch.cuda.is_available():
            model.to('cuda')
            
            start = time.time()
            for frame in test_frames[:30]:
                results = model(frame, verbose=False)
            gpu_time = time.time() - start
            print(f"   GPU processing (30 frames): {gpu_time:.3f}s")
            print(f"   GPU FPS: {30/gpu_time:.1f}")
            print(f"   Speedup: {cpu_time/gpu_time:.1f}x")
            
            # Batch processing test
            print("\n📊 Batch Processing (8 frames at once):")
            batch = np.stack(test_frames[:8])
            
            start = time.time()
            results = model(batch, verbose=False)
            batch_time = time.time() - start
            print(f"   GPU batch time: {batch_time:.3f}s")
            print(f"   GPU batch FPS: {8/batch_time:.1f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def check_gpu_status():
    """Check GPU status and utilization"""
    print("\n📊 GPU Status:")
    
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        
        if not gpus:
            print("❌ No GPUs found")
            return
        
        for i, gpu in enumerate(gpus):
            print(f"\n🎮 GPU {i}: {gpu.name}")
            print(f"   Memory: {gpu.memoryUsed:.0f}/{gpu.memoryTotal:.0f} MB ({gpu.memoryUtil*100:.1f}%)")
            print(f"   GPU Load: {gpu.load*100:.1f}%")
            print(f"   Temperature: {gpu.temperature}°C")
            
    except ImportError:
        print("⚠️ GPUtil not installed. Install with: pip install gputil")
    except Exception as e:
        print(f"❌ Error checking GPU status: {e}")

def main():
    """Run all GPU tests"""
    print("🚀 GPU Performance Test for Video Processing")
    print("=" * 50)
    
    # Check GPU status
    check_gpu_status()
    
    # Run tests
    test_pytorch_gpu()
    test_opencv_gpu()
    test_yolo_gpu()
    test_video_processing()
    
    print("\n✅ Testing complete!")
    print("\n💡 Tips for better GPU performance:")
    print("1. Use batch processing for multiple frames")
    print("2. Keep models loaded in GPU memory")
    print("3. Use FP16 (half precision) for faster inference")
    print("4. Process video in chunks to manage memory")
    print("5. Use GPU-accelerated video decoding (NVDEC)")

if __name__ == '__main__':
    main()