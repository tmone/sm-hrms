#!/usr/bin/env python3
"""
GPU Verification Script for StepMedia HRM
"""

import torch
import sys

def test_gpu():
    print("=" * 60)
    print("GPU VERIFICATION - StepMedia HRM")
    print("=" * 60)
    
    # Basic PyTorch info
    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version}")
    
    # CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        # GPU details
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {gpu_props.name}")
            print(f"  - Memory: {gpu_props.total_memory / 1024**3:.1f} GB")
            print(f"  - Compute Capability: {gpu_props.major}.{gpu_props.minor}")
        
        # Test GPU computation
        print("\nTesting GPU computation...")
        try:
            # Create test tensors
            device = torch.device("cuda:0")
            a = torch.randn(1000, 1000, device=device)
            b = torch.randn(1000, 1000, device=device)
            
            # Perform computation
            c = torch.mm(a, b)
            
            print(f"✅ GPU computation successful!")
            print(f"   - Tensor device: {c.device}")
            print(f"   - Result shape: {c.shape}")
            
            # Memory info
            print(f"\nGPU Memory Usage:")
            print(f"  - Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"  - Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
            
        except Exception as e:
            print(f"❌ GPU computation failed: {e}")
            return False
    else:
        print("❌ No CUDA GPU available")
        return False
    
    print("\n" + "=" * 60)
    print("✅ GPU SETUP COMPLETE - Ready for ML workloads!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    test_gpu()
