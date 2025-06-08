#!/usr/bin/env python3
"""
Batch OCR extraction for all videos that need it
Run this script to extract OCR data from all existing videos that don't have it
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the extraction script
from scripts.extract_ocr_from_existing_videos import main

if __name__ == '__main__':
    print("[START] Starting batch OCR extraction for all videos...")
    print("=" * 60)
    main()
    print("\n[OK] Batch OCR extraction completed!")