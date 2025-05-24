#!/usr/bin/env python
"""
Verify GPU is being used for video processing
"""
import torch
import subprocess
import time
import threading
import sys

def monitor_gpu_usage(duration=10):
    """Monitor GPU usage for specified duration"""
    print("🔍 Monitoring GPU usage...")
    print("-" * 60)
    
    gpu_usage = []
    stop_monitoring = False
    
    def get_gpu_stats():
        while not stop_monitoring:
            try:
                # Get GPU utilization
                result = subprocess.run([
                    'nvidia-smi', 
                    '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    stats = result.stdout.strip().split(', ')
                    gpu_util = int(stats[0])
                    mem_used = int(stats[1])
                    mem_total = int(stats[2])
                    temp = int(stats[3])
                    
                    gpu_usage.append({
                        'time': time.time(),
                        'gpu_util': gpu_util,
                        'mem_used': mem_used,
                        'mem_total': mem_total,
                        'temp': temp
                    })
                    
                    # Print current status
                    print(f"\rGPU: {gpu_util}% | Memory: {mem_used}/{mem_total} MB | Temp: {temp}°C", end='')
                    
            except Exception as e:
                print(f"\nError monitoring GPU: {e}")
                break
            
            time.sleep(0.5)
    
    # Start monitoring in background thread
    monitor_thread = threading.Thread(target=get_gpu_stats)
    monitor_thread.start()
    
    # Wait for specified duration
    time.sleep(duration)
    stop_monitoring = True
    monitor_thread.join()
    
    print("\n" + "-" * 60)
    
    # Analyze results
    if gpu_usage:
        avg_util = sum(s['gpu_util'] for s in gpu_usage) / len(gpu_usage)
        max_util = max(s['gpu_util'] for s in gpu_usage)
        avg_mem = sum(s['mem_used'] for s in gpu_usage) / len(gpu_usage)
        max_mem = max(s['mem_used'] for s in gpu_usage)
        
        print(f"📊 GPU Usage Summary:")
        print(f"  Average GPU Utilization: {avg_util:.1f}%")
        print(f"  Peak GPU Utilization: {max_util}%")
        print(f"  Average Memory Usage: {avg_mem:.0f} MB")
        print(f"  Peak Memory Usage: {max_mem} MB")
        
        if avg_util > 50:
            print("  ✅ GPU is being actively used!")
        elif avg_util > 10:
            print("  ⚠️  GPU usage is low - may not be fully utilized")
        else:
            print("  ❌ GPU appears to be idle - check configuration")
    else:
        print("❌ No GPU usage data collected")

def test_gpu_processing():
    """Test if GPU is being used for YOLO inference"""
    print("\n🧪 Testing GPU Processing...")
    print("-" * 60)
    
    try:
        from ultralytics import YOLO
        import numpy as np
        
        # Check CUDA availability
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("⚠️  CUDA not available - GPU processing disabled")
            return
        
        # Load YOLO model
        print("\nLoading YOLO model...")
        model = YOLO('yolov8n.pt')
        
        # Move to GPU
        model.to('cuda:0')
        print("✅ Model loaded on GPU")
        
        # Create test batch
        batch_size = 8
        test_frames = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) for _ in range(batch_size)]
        
        print(f"\n🏃 Running inference on {batch_size} frames...")
        
        # Start GPU monitoring
        monitor_thread = threading.Thread(target=lambda: monitor_gpu_usage(5))
        monitor_thread.start()
        
        # Run inference
        start_time = time.time()
        results = model.predict(
            test_frames,
            device='cuda:0',
            verbose=False,
            half=True  # Use FP16
        )
        inference_time = time.time() - start_time
        
        # Wait for monitoring to complete
        monitor_thread.join()
        
        print(f"\n⏱️  Inference time: {inference_time:.2f}s")
        print(f"📊 FPS: {batch_size / inference_time:.1f}")
        
        # Compare with CPU
        print("\n🔄 Comparing with CPU inference...")
        model.to('cpu')
        start_time = time.time()
        results_cpu = model.predict(
            test_frames[0],  # Just one frame for CPU
            device='cpu',
            verbose=False
        )
        cpu_time = time.time() - start_time
        
        print(f"CPU inference (1 frame): {cpu_time:.2f}s")
        print(f"GPU speedup: {cpu_time * batch_size / inference_time:.1f}x")
        
    except Exception as e:
        print(f"❌ Error testing GPU: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=" * 60)
    print("GPU Usage Verification for Video Processing")
    print("=" * 60)
    
    # Check if nvidia-smi is available
    try:
        result = subprocess.run(['nvidia-smi', '--version'], capture_output=True)
        if result.returncode != 0:
            print("❌ nvidia-smi not found - cannot monitor GPU usage")
            print("Make sure NVIDIA drivers are installed")
            sys.exit(1)
    except:
        print("❌ nvidia-smi not found - cannot monitor GPU usage")
        sys.exit(1)
    
    # Test GPU processing
    test_gpu_processing()
    
    print("\n" + "=" * 60)
    print("✅ GPU verification complete!")
    print("=" * 60)