#!/usr/bin/env python3
"""
Test GPU detection with person recognition integration
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from processing.gpu_enhanced_detection import extract_persons_data_gpu
    print("[OK] Successfully imported extract_persons_data_gpu")
    
    # Check function signature
    import inspect
    sig = inspect.signature(extract_persons_data_gpu)
    print(f"[TRACE] Function signature: {sig}")
    
    # Check parameters
    params = sig.parameters
    print("\n[PIN] Parameters:")
    for name, param in params.items():
        default = param.default
        if default is inspect.Parameter.empty:
            default = "Required"
        print(f"  - {name}: {default}")
    
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
except Exception as e:
    print(f"[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()