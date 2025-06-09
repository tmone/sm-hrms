#!/usr/bin/env python3
"""
Verify OCR setup and attendance data flow
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app

def check_database_fields():
    """Check if all required database fields exist"""
    print("="*80)
    print("CHECKING DATABASE FIELDS")
    print("="*80)
    
    app = create_app()
    with app.app_context():
        db = app.db
        from sqlalchemy import text
        
        # Check videos table
        print("\n1. Checking videos table for OCR fields...")
        try:
            result = db.session.execute(text("PRAGMA table_info(videos)"))
            columns = {row[1]: row[2] for row in result}
            
            required_fields = {
                'ocr_location': 'VARCHAR',
                'ocr_video_date': 'DATE', 
                'ocr_video_time': 'TIME',
                'ocr_extraction_done': 'BOOLEAN',
                'ocr_extraction_confidence': 'FLOAT'
            }
            
            for field, field_type in required_fields.items():
                if field in columns:
                    print(f"  ✓ {field} ({field_type}) - exists")
                else:
                    print(f"  ✗ {field} ({field_type}) - MISSING")
                    
        except Exception as e:
            print(f"  ✗ Error checking videos table: {e}")
        
        # Check detected_persons table
        print("\n2. Checking detected_persons table for attendance fields...")
        try:
            result = db.session.execute(text("PRAGMA table_info(detected_persons)"))
            columns = {row[1]: row[2] for row in result}
            
            required_fields = {
                'attendance_date': 'DATE',
                'attendance_time': 'TIME',
                'attendance_location': 'VARCHAR',
                'check_in_time': 'TIMESTAMP',
                'check_out_time': 'TIMESTAMP'
            }
            
            for field, field_type in required_fields.items():
                if field in columns:
                    print(f"  ✓ {field} ({field_type}) - exists")
                else:
                    print(f"  ✗ {field} ({field_type}) - MISSING")
                    
        except Exception as e:
            print(f"  ✗ Error checking detected_persons table: {e}")

def check_ocr_data():
    """Check if any videos have OCR data"""
    print("\n" + "="*80)
    print("CHECKING OCR DATA")
    print("="*80)
    
    app = create_app()
    with app.app_context():
        Video = app.Video
        DetectedPerson = app.DetectedPerson
        
        # Count videos
        total_videos = Video.query.count()
        completed_videos = Video.query.filter_by(status='completed').count()
        ocr_videos = Video.query.filter_by(ocr_extraction_done=True).count()
        
        print(f"\nVideo Statistics:")
        print(f"  Total videos: {total_videos}")
        print(f"  Completed videos: {completed_videos}")
        print(f"  Videos with OCR data: {ocr_videos}")
        
        if ocr_videos > 0:
            # Show sample OCR data
            print("\nSample OCR data from videos:")
            videos_with_ocr = Video.query.filter_by(ocr_extraction_done=True).limit(3).all()
            for video in videos_with_ocr:
                print(f"\n  Video: {video.filename}")
                print(f"    - Location: {video.ocr_location}")
                print(f"    - Date: {video.ocr_video_date}")
                print(f"    - Time: {video.ocr_video_time}")
                print(f"    - Confidence: {video.ocr_extraction_confidence:.2%}")
        
        # Check attendance data
        print("\nAttendance Data Statistics:")
        total_detections = DetectedPerson.query.count()
        detections_with_attendance = DetectedPerson.query.filter(
            DetectedPerson.attendance_date.isnot(None)
        ).count()
        
        print(f"  Total detections: {total_detections}")
        print(f"  Detections with attendance data: {detections_with_attendance}")
        
        if detections_with_attendance > 0:
            print(f"  ✓ Attendance data is available!")
        else:
            print(f"  ⚠️  No attendance data found")

def check_ocr_libraries():
    """Check OCR library status"""
    print("\n" + "="*80)
    print("CHECKING OCR LIBRARIES")
    print("="*80)
    
    # Check EasyOCR
    print("\n1. EasyOCR:")
    try:
        import easyocr
        print("  ✓ EasyOCR is installed")
        
        # Try to create reader
        try:
            reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            print("  ✓ EasyOCR reader can be initialized")
        except Exception as e:
            print(f"  ✗ EasyOCR initialization error: {e}")
            if "protocol version" in str(e).lower():
                print("    → Run: python scripts/fix_ocr_extraction_error.py")
    except ImportError:
        print("  ✗ EasyOCR not installed")
        print("    → Install with: pip install easyocr")
    
    # Check PyTesseract
    print("\n2. PyTesseract:")
    try:
        import pytesseract
        print("  ✓ PyTesseract is installed")
        
        try:
            version = pytesseract.get_tesseract_version()
            print(f"  ✓ Tesseract version: {version}")
        except Exception as e:
            print(f"  ✗ Tesseract executable not found: {e}")
            print("    → Install Tesseract OCR:")
            print("      Windows: https://github.com/UB-Mannheim/tesseract/wiki")
            print("      Linux: sudo apt-get install tesseract-ocr")
    except ImportError:
        print("  ✗ PyTesseract not installed")
        print("    → Install with: pip install pytesseract")

def check_processing_integration():
    """Check if OCR is integrated into processing"""
    print("\n" + "="*80)
    print("CHECKING PROCESSING INTEGRATION")
    print("="*80)
    
    # Check if OCR extractor exists
    try:
        from hr_management.processing.ocr_extractor import VideoOCRExtractor
        print("✓ OCR extractor module is available")
    except ImportError:
        print("✗ OCR extractor module not found")
    
    # Check if GPU detection includes OCR
    try:
        from processing.gpu_enhanced_detection import process_video_enhanced_gpu
        print("✓ GPU enhanced detection is available")
        
        # Check if it includes OCR extraction
        import inspect
        source = inspect.getsource(process_video_enhanced_gpu)
        if "ocr_extractor" in source.lower() or "videoocr" in source.lower():
            print("✓ GPU detection includes OCR extraction")
        else:
            print("⚠️  GPU detection may not include OCR extraction")
    except Exception as e:
        print(f"⚠️  Could not verify GPU detection: {e}")
    
    # Check if save function handles OCR
    try:
        from hr_management.processing.enhanced_save_detections import save_detections_with_ocr
        print("✓ Enhanced save function with OCR support is available")
    except ImportError:
        print("✗ Enhanced save function not found")

def main():
    """Main verification function"""
    print("OCR SETUP VERIFICATION")
    print("This will check if OCR is properly configured\n")
    
    # Run all checks
    check_database_fields()
    check_ocr_libraries()
    check_processing_integration()
    check_ocr_data()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\nIf you see any ✗ marks above, you may need to:")
    print("1. Run migrations: python scripts/migrate_add_ocr_video_time.py")
    print("2. Fix OCR errors: python scripts/fix_ocr_extraction_error.py")
    print("3. Process existing videos: python scripts/fix_attendance_missing_data.py")
    print("\nFor new videos, OCR extraction is automatic during processing!")

if __name__ == "__main__":
    main()