#!/usr/bin/env python3
"""
Test available video codecs for OpenCV
"""
import cv2
import numpy as np

def test_codecs():
    """Test which video codecs are available"""
    
    print("🎬 Testing available video codecs...")
    
    # Create a test frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_frame, "Test Frame", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Test codecs
    codecs_to_test = [
        # H.264 variants
        ('H264', '.mp4', 'H.264 (OpenCV)'),
        ('h264', '.mp4', 'H.264 (lowercase)'),
        ('avc1', '.mp4', 'H.264/AVC (Apple)'),
        ('AVC1', '.mp4', 'H.264/AVC (uppercase)'),
        ('x264', '.mp4', 'x264 (libx264)'),
        ('X264', '.mp4', 'X264 (uppercase)'),
        
        # MPEG-4 variants
        ('mp4v', '.mp4', 'MPEG-4'),
        ('MP4V', '.mp4', 'MPEG-4 (uppercase)'),
        ('FMP4', '.mp4', 'FFmpeg MPEG-4'),
        
        # AVI codecs
        ('MJPG', '.avi', 'Motion JPEG (AVI)'),
        ('XVID', '.avi', 'Xvid (AVI)'),
        ('DIVX', '.avi', 'DivX (AVI)'),
        ('H264', '.avi', 'H.264 in AVI'),
        
        # Other formats
        ('VP80', '.webm', 'VP8 (WebM)'),
        ('VP90', '.webm', 'VP9 (WebM)'),
    ]
    
    results = []
    
    for codec, ext, name in codecs_to_test:
        filename = f'test_{codec}{ext}'
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(filename, fourcc, 30.0, (640, 480))
            
            if out.isOpened():
                # Write a few frames
                for i in range(10):
                    out.write(test_frame)
                out.release()
                
                # Check if file was created and has size
                import os
                if os.path.exists(filename) and os.path.getsize(filename) > 0:
                    file_size = os.path.getsize(filename)
                    results.append(f"✅ {name:<25} ({codec}{ext}): SUCCESS - {file_size} bytes")
                    os.remove(filename)  # Clean up
                else:
                    results.append(f"❌ {name:<25} ({codec}{ext}): File empty or not created")
            else:
                results.append(f"❌ {name:<25} ({codec}{ext}): VideoWriter failed to open")
                
        except Exception as e:
            results.append(f"❌ {name:<25} ({codec}{ext}): ERROR - {str(e)}")
    
    print("\n📊 Codec Test Results:\n")
    for result in results:
        print(result)
    
    # Recommendations
    print("\n💡 Recommendations:")
    h264_available = any('✅' in r and 'H.264' in r for r in results)
    if h264_available:
        print("✅ H.264 codec is available - use this for best browser compatibility")
    else:
        print("❌ H.264 codec not available - install FFmpeg for better codec support")
        print("   Run: python install_video_codecs.py")
    
    mjpeg_available = any('✅' in r and 'Motion JPEG' in r for r in results)
    if mjpeg_available:
        print("✅ Motion JPEG is available - good fallback for AVI container")

if __name__ == '__main__':
    test_codecs()