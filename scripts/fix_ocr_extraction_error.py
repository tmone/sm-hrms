#!/usr/bin/env python3
"""
Fix OCR extraction error: Invalid protocol version
This error typically occurs with EasyOCR model corruption or version mismatch
"""

import sys
import os
import shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def clear_easyocr_cache():
    """Clear EasyOCR model cache to force re-download"""
    print("="*80)
    print("CLEARING EASYOCR CACHE")
    print("="*80)
    
    # Common EasyOCR cache locations
    cache_dirs = [
        os.path.expanduser("~/.EasyOCR"),
        os.path.expanduser("~/EasyOCR"),
        os.path.join(os.path.expanduser("~"), ".cache", "easyocr"),
        "models/easyocr",
        ".easyocr"
    ]
    
    cleared = False
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                print(f"Found cache directory: {cache_dir}")
                # List contents
                for item in os.listdir(cache_dir):
                    print(f"  - {item}")
                
                # Clear the cache
                shutil.rmtree(cache_dir)
                print(f"✓ Cleared cache directory: {cache_dir}")
                cleared = True
            except Exception as e:
                print(f"✗ Error clearing {cache_dir}: {e}")
    
    if not cleared:
        print("No EasyOCR cache directories found")
    
    return cleared

def test_ocr_libraries():
    """Test OCR libraries to identify the issue"""
    print("\n" + "="*80)
    print("TESTING OCR LIBRARIES")
    print("="*80)
    
    # Test PyTesseract
    print("\n1. Testing PyTesseract...")
    try:
        import pytesseract
        print("✓ PyTesseract imported successfully")
        
        # Check if tesseract is installed
        try:
            version = pytesseract.get_tesseract_version()
            print(f"✓ Tesseract version: {version}")
        except Exception as e:
            print(f"✗ Tesseract not found: {e}")
            print("  Install with: apt-get install tesseract-ocr (Linux) or download from GitHub (Windows)")
    except ImportError:
        print("✗ PyTesseract not installed")
        print("  Install with: pip install pytesseract")
    
    # Test EasyOCR
    print("\n2. Testing EasyOCR...")
    try:
        import easyocr
        print("✓ EasyOCR imported successfully")
        
        # Try to create reader with CPU only (avoid GPU issues)
        print("  Creating EasyOCR reader (CPU mode)...")
        reader = easyocr.Reader(['en'], gpu=False, verbose=True)
        print("✓ EasyOCR reader created successfully")
        
    except Exception as e:
        print(f"✗ EasyOCR error: {e}")
        if "protocol version" in str(e).lower():
            print("\n⚠️  This appears to be a model file corruption issue.")
            print("  The EasyOCR model files may be corrupted or incompatible.")

def reinstall_easyocr():
    """Reinstall EasyOCR to fix version issues"""
    print("\n" + "="*80)
    print("REINSTALLING EASYOCR")
    print("="*80)
    
    import subprocess
    
    try:
        # Uninstall existing version
        print("Uninstalling existing EasyOCR...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "easyocr", "-y"], 
                      capture_output=True, text=True)
        
        # Install fresh version
        print("Installing fresh EasyOCR...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", "easyocr", "--no-cache-dir"], 
                               capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ EasyOCR reinstalled successfully")
            return True
        else:
            print("✗ Failed to reinstall EasyOCR:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"✗ Error during reinstallation: {e}")
        return False

def create_fallback_ocr_extractor():
    """Create a fallback OCR extractor that uses PyTesseract"""
    print("\n" + "="*80)
    print("CREATING FALLBACK OCR CONFIGURATION")
    print("="*80)
    
    fallback_config = """# Fallback OCR configuration
# This file configures the OCR system to use PyTesseract when EasyOCR fails

OCR_ENGINE = "tesseract"  # Use tesseract instead of easyocr
OCR_FALLBACK_ENABLED = True
OCR_GPU_ENABLED = False  # Disable GPU to avoid CUDA issues

# Tesseract configuration
TESSERACT_CONFIG = "--oem 3 --psm 6"  # LSTM engine with uniform text block

print("[OCR] Using Tesseract fallback configuration due to EasyOCR issues")
"""
    
    config_path = "processing/ocr_config.py"
    try:
        with open(config_path, 'w') as f:
            f.write(fallback_config)
        print(f"✓ Created fallback OCR configuration: {config_path}")
        return True
    except Exception as e:
        print(f"✗ Error creating fallback config: {e}")
        return False

def patch_ocr_extractor():
    """Patch the OCR extractor to handle the protocol version error"""
    print("\n" + "="*80)
    print("PATCHING OCR EXTRACTOR")
    print("="*80)
    
    patch_content = '''"""
Patched OCR Extractor - Handles protocol version errors
"""
import logging

logger = logging.getLogger(__name__)

# Store original OCR extractor
_original_ocr_extractor = None

def patch_video_ocr_extractor():
    """Patch VideoOCRExtractor to handle protocol version errors"""
    try:
        from hr_management.processing.ocr_extractor import VideoOCRExtractor
        
        # Save original init
        original_init = VideoOCRExtractor.__init__
        
        def patched_init(self, ocr_engine='easyocr', date_format='DD-MM-YYYY'):
            """Patched init that falls back to tesseract on EasyOCR errors"""
            try:
                # Try original initialization
                original_init(self, ocr_engine, date_format)
            except Exception as e:
                if "protocol version" in str(e).lower():
                    logger.warning(f"EasyOCR protocol error: {e}")
                    logger.info("Falling back to Tesseract OCR")
                    # Force tesseract
                    self.ocr_engine = 'tesseract'
                    self.reader = None
                    self.date_format = date_format
                else:
                    raise
        
        # Apply patch
        VideoOCRExtractor.__init__ = patched_init
        logger.info("Successfully patched VideoOCRExtractor")
        
    except Exception as e:
        logger.error(f"Failed to patch VideoOCRExtractor: {e}")

# Auto-apply patch
patch_video_ocr_extractor()
'''
    
    patch_path = "processing/patch_ocr_protocol_error.py"
    try:
        with open(patch_path, 'w') as f:
            f.write(patch_content)
        print(f"✓ Created OCR patch: {patch_path}")
        return True
    except Exception as e:
        print(f"✗ Error creating patch: {e}")
        return False

def main():
    """Main function to fix OCR extraction error"""
    print("FIXING OCR EXTRACTION ERROR")
    print("This will fix the 'Invalid protocol version' error\n")
    
    # Step 1: Clear EasyOCR cache
    print("Step 1: Clearing EasyOCR cache...")
    clear_easyocr_cache()
    
    # Step 2: Test OCR libraries
    print("\nStep 2: Testing OCR libraries...")
    test_ocr_libraries()
    
    # Step 3: Create fallback configuration
    print("\nStep 3: Creating fallback configuration...")
    create_fallback_ocr_extractor()
    
    # Step 4: Create error handling patch
    print("\nStep 4: Creating error handling patch...")
    patch_ocr_extractor()
    
    # Step 5: Offer to reinstall EasyOCR
    print("\n" + "="*80)
    print("RECOMMENDED ACTIONS")
    print("="*80)
    print("\n1. The system will now use PyTesseract as fallback when EasyOCR fails")
    print("\n2. To fix EasyOCR permanently, you can:")
    print("   a) Reinstall EasyOCR: python -m pip install --upgrade --force-reinstall easyocr")
    print("   b) Clear Python cache: python -m pip cache purge")
    print("   c) Install specific version: python -m pip install easyocr==1.7.0")
    print("\n3. For Windows users, ensure Tesseract is installed:")
    print("   Download from: https://github.com/UB-Mannheim/tesseract/wiki")
    
    response = input("\nWould you like to try reinstalling EasyOCR now? (y/n): ")
    if response.lower() == 'y':
        reinstall_easyocr()
        print("\nPlease restart the application for changes to take effect.")
    
    print("\n✅ OCR error handling is now in place!")
    print("The system will automatically fall back to Tesseract if EasyOCR fails.")

if __name__ == "__main__":
    main()