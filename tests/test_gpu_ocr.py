#!/usr/bin/env python3
"""
Test GPU availability for OCR
"""

import sys

# Test PyTorch CUDA
print("[SEARCH] Checking GPU availability for OCR...")
print("-" * 50)

try:
    import torch
    print(f"[OK] PyTorch version: {torch.__version__}")
    print(f"[GPU] CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"[GPU] CUDA device count: {torch.cuda.device_count()}")
        print(f"[GPU] Current device: {torch.cuda.current_device()}")
        print(f"[GPU] Device name: {torch.cuda.get_device_name(0)}")
        print(f"[SAVE] GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("[WARNING] CUDA is not available")
        
except ImportError as e:
    print(f"[ERROR] PyTorch not installed: {e}")

print("-" * 50)

# Test EasyOCR
try:
    import easyocr
    print("[OK] EasyOCR is installed")
    
    # Try to create reader with GPU
    try:
        print("[PROCESSING] Creating EasyOCR reader with GPU...")
        reader = easyocr.Reader(['en'], gpu=True, verbose=False)
        print("[OK] EasyOCR successfully initialized with GPU")
    except Exception as e:
        print(f"[ERROR] Failed to initialize EasyOCR with GPU: {e}")
        print("[PROCESSING] Trying CPU mode...")
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        print("[OK] EasyOCR initialized with CPU")
        
except ImportError:
    print("[ERROR] EasyOCR not installed")

print("-" * 50)

# Check environment variables
import os
print("[TRACE] Environment variables:")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'not set')}")

print("-" * 50)
print("[OK] Test complete")