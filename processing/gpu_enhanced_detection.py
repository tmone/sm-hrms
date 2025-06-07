"""
GPU-accelerated person detection module for faster video processing
"""
import cv2
import numpy as np
from datetime import datetime
import os
from pathlib import Path
import json
import glob
import uuid
import tempfile
import time

# Set up logging
try:
    from config_logging import get_logger
    logger = get_logger(__name__)
except:
    import logging
    logger = logging.getLogger(__name__)

# Import resource throttler
try:
    from processing.resource_throttler import get_throttler, safe_gpu_operation, BatchProcessor
    throttler = get_throttler()
    THROTTLING_ENABLED = True
except:
    logger.warning("Resource throttling not available")
    THROTTLING_ENABLED = False
    throttler = None

# Try to import torch and check CUDA availability
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
    if not CUDA_AVAILABLE:
        logger.info("CUDA not available - will use CPU for processing")
except ImportError:
    logger.info("PyTorch not available - will use OpenCV DNN")
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

# Try to import GPU appearance tracker
GPU_TRACKER_AVAILABLE = False
try:
    if TORCH_AVAILABLE and CUDA_AVAILABLE:
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from hr_management.processing.gpu_appearance_tracker import GPUPersonTracker
        GPU_TRACKER_AVAILABLE = True
        print("✅ GPU Appearance Tracker available")
except ImportError as e:
    print(f"⚠️ GPU Appearance Tracker not available: {e}")

# Try to import OCR extractor
OCR_AVAILABLE = False
try:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from hr_management.processing.ocr_extractor import VideoOCRExtractor
    OCR_AVAILABLE = True
    print("✅ OCR Extractor available")
except ImportError as e:
    print(f"⚠️ OCR Extractor not available: {e}")

# Try to import recognition - use venv wrapper to avoid NumPy issues
RECOGNITION_AVAILABLE = False
try:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Try direct import first
    try:
        from hr_management.processing.person_recognition_inference_simple import PersonRecognitionInferenceSimple
        RECOGNITION_AVAILABLE = True
        print("✅ Person Recognition available (direct)")
    except ImportError as e:
        # Use virtual environment wrapper as fallback
        from processing.venv_recognition_wrapper import VenvRecognitionWrapper, recognize_in_venv
        RECOGNITION_AVAILABLE = True
        print("✅ Person Recognition available (via venv wrapper)")
except Exception as e:
    print(f"⚠️ Person Recognition not available: {e}")
    RECOGNITION_AVAILABLE = False

def get_next_person_id():
    """
    Get the next available person ID by checking existing person folders
    and maintaining a persistent counter file.
    This ensures unique IDs across all video processing sessions.
    """
    persons_dir = Path('processing/outputs/persons')
    persons_dir.mkdir(parents=True, exist_ok=True)
    
    # Path to the ID counter file
    counter_file = persons_dir / 'person_id_counter.json'
    
    # First, check existing folders to find the maximum ID
    existing_persons = list(persons_dir.glob('PERSON-*'))
    max_folder_id = 0
    
    for person_folder in existing_persons:
        try:
            folder_name = person_folder.name
            if folder_name.startswith('PERSON-'):
                person_id = int(folder_name.replace('PERSON-', ''))
                max_folder_id = max(max_folder_id, person_id)
        except ValueError:
            continue
    
    # Check the counter file
    max_counter_id = 0
    if counter_file.exists():
        try:
            with open(counter_file, 'r') as f:
                data = json.load(f)
                max_counter_id = data.get('last_person_id', 0)
        except Exception as e:
            print(f"⚠️ Error reading counter file: {e}")
    
    # Use the maximum of both
    max_id = max(max_folder_id, max_counter_id)
    next_id = max_id + 1
    
    # Update the counter file
    try:
        with open(counter_file, 'w') as f:
            json.dump({
                'last_person_id': next_id,
                'updated_at': datetime.now().isoformat(),
                'total_persons': len(existing_persons)
            }, f, indent=2)
    except Exception as e:
        print(f"⚠️ Error updating counter file: {e}")
    
    print(f"📊 Found {len(existing_persons)} existing persons, next ID will be: PERSON-{next_id:04d}")
    return next_id


def update_person_id_counter(last_used_id):
    """
    Update the person ID counter after processing
    """
    persons_dir = Path('processing/outputs/persons')
    counter_file = persons_dir / 'person_id_counter.json'
    
    try:
        current_max = 0
        if counter_file.exists():
            with open(counter_file, 'r') as f:
                data = json.load(f)
                current_max = data.get('last_person_id', 0)
        
        # Only update if we used a higher ID
        if last_used_id > current_max:
            with open(counter_file, 'w') as f:
                json.dump({
                    'last_person_id': last_used_id,
                    'updated_at': datetime.now().isoformat()
                }, f, indent=2)
            print(f"📊 Updated person ID counter to: {last_used_id}")
    except Exception as e:
        print(f"⚠️ Error updating person ID counter: {e}")


def update_video_progress(video_id, progress, message="Processing...", app=None):
    """Update video processing progress in database"""
    try:
        # Try to get app context
        if app and video_id:
            with app.app_context():
                db = app.db
                Video = app.Video
                
                # Use a new session for this update
                video = db.session.get(Video, video_id)
                if video:
                    video.processing_progress = int(progress)
                    video.processing_log = f"{video.processing_log}\n[{datetime.now().strftime('%H:%M:%S')}] {message} ({progress}%)" if video.processing_log else f"[{datetime.now().strftime('%H:%M:%S')}] {message} ({progress}%)"
                    db.session.commit()
                    
                    # Important: Close the session after update
                    db.session.close()
                    db.session.remove()
                    
                    print(f"📊 Progress updated: {progress}% - {message}")
        else:
            # Just log without database update
            print(f"📊 Progress: {progress}% - {message}")
    except Exception as e:
        print(f"⚠️ Could not update progress: {e}")
        # Don't propagate the error - progress updates are non-critical

def gpu_person_detection_task(video_path, gpu_config=None, video_id=None, app=None):
    """
    GPU-accelerated person detection with optimizations for large videos
    """
    try:
        # Default GPU configuration
        if gpu_config is None:
            gpu_config = {
                'use_gpu': CUDA_AVAILABLE,
                'batch_size': 4 if CUDA_AVAILABLE else 2,  # Reduced from 8 to 4 for stability
                'device': 'cuda:0' if CUDA_AVAILABLE else 'cpu',
                'fp16': CUDA_AVAILABLE,
                'num_workers': 4
            }
        
        print(f"🎮 GPU Detection Config: {gpu_config}")
        print(f"📊 CUDA Available: {CUDA_AVAILABLE}")
        if CUDA_AVAILABLE and TORCH_AVAILABLE:
            print(f"🎮 GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Skip progress updates during processing to avoid database connections
        print("📊 Progress: 5% - Initializing GPU detection...")
        
        # Set CUDA to use less aggressive memory allocation
        if CUDA_AVAILABLE and torch.cuda.is_available():
            try:
                # Limit CUDA memory growth
                torch.cuda.set_per_process_memory_fraction(0.8)  # Use max 80% of GPU memory
                # Clear any existing cache
                torch.cuda.empty_cache()
                logger.info("CUDA memory management configured")
            except Exception as e:
                logger.warning(f"Could not configure CUDA memory: {e}")
        
        # Initialize person recognizer if available
        ui_style_recognizer = None
        if RECOGNITION_AVAILABLE:
            try:
                # Check if we need to use venv wrapper
                try:
                    # Try direct import
                    from hr_management.processing.person_recognition_inference_simple import PersonRecognitionInferenceSimple
                    
                    # Try to find the latest refined model
                    model_path = Path('models/person_recognition')
                    refined_models = list(model_path.glob('refined_*'))
                    
                    if refined_models:
                        # Use the latest refined model
                        latest_model = max(refined_models, key=lambda p: p.stat().st_mtime)
                        model_name = latest_model.name
                        print(f"🎯 Using model: {model_name}")
                    else:
                        # Try the default model from config
                        config_path = Path("models/person_recognition/config.json")
                        if config_path.exists():
                            import json
                            with open(config_path) as f:
                                config = json.load(f)
                            model_name = config.get('default_model', 'person_model_svm_20250607_181818')
                            print(f"🎯 Using default model from config: {model_name}")
                        else:
                            model_name = 'person_model_svm_20250607_181818'
                            print(f"🎯 Using fallback model: {model_name}")
                    
                    ui_style_recognizer = PersonRecognitionInferenceSimple(
                        model_name=model_name,
                        confidence_threshold=0.8  # High threshold for automatic recognition
                    )
                    print("✅ Person Recognizer initialized directly")
                except (ImportError, Exception) as e:
                    # Use venv wrapper
                    print(f"⚠️ Direct load failed: {e}")
                    print("🔄 Using virtual environment wrapper for recognition")
                    try:
                        from processing.venv_recognition_wrapper import VenvRecognitionWrapper
                        ui_style_recognizer = VenvRecognitionWrapper()
                        print("✅ Person recognition loaded via venv wrapper")
                    except Exception as venv_error:
                        print(f"❌ Venv recognition error: {venv_error}")
                        ui_style_recognizer = None
            except Exception as e:
                print(f"❌ Could not initialize any recognizer: {e}")
                ui_style_recognizer = None
        
        # Initialize video capture with backend preference
        # Try different backends for better codec support
        cap = None
        backends = [
            cv2.CAP_FFMPEG,  # FFmpeg backend (best for HEVC)
            cv2.CAP_ANY,     # Auto-select backend
            cv2.CAP_MSMF,    # Windows Media Foundation (Windows)
            cv2.CAP_DSHOW    # DirectShow (Windows)
        ]
        
        for backend in backends:
            try:
                cap = cv2.VideoCapture(video_path, backend)
                if cap.isOpened():
                    print(f"✅ Opened video with backend: {backend}")
                    break
                else:
                    cap.release()
            except Exception as e:
                print(f"⚠️ Failed with backend {backend}: {e}")
                continue
        
        if cap is None or not cap.isOpened():
            return {'error': 'Failed to open video file with any backend'}
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Get input file size
        input_size_mb = Path(video_path).stat().st_size / (1024 * 1024)
        print(f"📹 Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames, {duration:.1f}s")
        print(f"📁 Input size: {input_size_mb:.1f} MB")
        
        # Update progress: Video loaded
        if video_id:
            # Skip DB update to avoid connection pool issues
            print(f"📊 Progress: 10% - Video loaded: {total_frames} frames, {duration:.1f}s")
        
        # Extract OCR data (timestamp and location) from video
        ocr_data = None
        if OCR_AVAILABLE:
            print("\n🔤 Extracting OCR data from video...")
            if video_id:
                print("📊 Progress: 12% - Extracting timestamps and location...")
            
            try:
                ocr_extractor = VideoOCRExtractor(ocr_engine='easyocr')
                # Sample every 10 seconds for OCR (300 frames at 30fps)
                sample_interval = int(fps * 10) if fps > 0 else 300
                ocr_data = ocr_extractor.extract_video_info(video_path, sample_interval=sample_interval)
                
                if ocr_data:
                    print(f"✅ OCR extraction complete:")
                    print(f"   - Location: {ocr_data.get('location', 'Not found')}")
                    print(f"   - Video Date: {ocr_data.get('video_date', 'Not found')}")
                    print(f"   - Confidence: {ocr_data.get('confidence', 0):.2%}")
                else:
                    print("⚠️ No OCR data extracted")
            except Exception as e:
                print(f"⚠️ OCR extraction failed: {e}")
                ocr_data = None
        
        # Create output directory for annotated video
        # Use the same folder as uploads for consistency
        output_dir = Path('static/uploads')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        video_name = Path(video_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{video_name}_annotated_{timestamp}.mp4"
        
        # Process every nth frame for speed (adjust based on video length)
        # Only skip frames for very long videos to maintain quality
        skip_frames = 2 if total_frames > 3600 else 1  # Skip frames for videos > 120s @ 30fps
        
        # Initialize video writer with Windows-compatible codec
        out = None
        actual_output_path = output_path
        
        # Adjust output FPS based on skip_frames to avoid timing issues
        output_fps = fps / skip_frames if skip_frames > 1 else fps
        
        try:
            # Use Windows-compatible video writer
            from .windows_video_writer import create_windows_compatible_writer
            out, actual_output_path = create_windows_compatible_writer(output_path, output_fps, width, height)
            output_path = actual_output_path  # Update path in case extension changed
            print(f"✅ Video writer initialized successfully")
        except Exception as e:
            print(f"⚠️ Failed to use Windows-compatible writer: {e}")
            
            # Simple fallback - try mp4v directly
            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height), True)
                
                if out.isOpened():
                    print(f"✅ Using MPEG-4 codec (fallback)")
                else:
                    out.release()
                    out = None
            except Exception as e:
                print(f"⚠️ Failed to use mp4v codec: {e}")
        
        # If MP4 codecs fail, try XVID in AVI container
        if out is None:
            avi_output_path = str(output_path).replace('.mp4', '.avi')
            print(f"🔄 Trying AVI container with XVID codec: {avi_output_path}")
            
            try:
                # Use XVID for better compression
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(avi_output_path, fourcc, output_fps, (width, height), True)
                if out.isOpened():
                    output_path = Path(avi_output_path)
                    print(f"✅ Using XVID codec in AVI container")
                    print(f"📹 Output settings: {width}x{height} @ {output_fps}fps")
                    print(f"⚠️  Note: AVI files may need conversion for web playback")
                else:
                    out.release()
                    out = None
            except Exception as e:
                print(f"⚠️ Failed to use AVI container: {e}")
        
        if out is None:
            print("❌ ERROR: Could not initialize video writer with any codec!")
            print("💡 Please install FFmpeg for better codec support:")
            print("   Windows: winget install ffmpeg")
            print("   Or download from: https://ffmpeg.org/download.html")
            return {'error': 'Failed to initialize video writer. Please install FFmpeg.'}
        
        # Load YOLO model with GPU support
        try:
            from ultralytics import YOLO
            
            # Use YOLOv8 model optimized for GPU
            model = YOLO('yolov8n.pt')  # Nano model for speed
            
            # Move model to GPU if available
            if gpu_config['use_gpu']:
                try:
                    model.to(gpu_config['device'])
                    print(f"✅ YOLO model loaded on GPU: {gpu_config['device']}")
                    
                    # Enable half precision for faster inference
                    if gpu_config['fp16']:
                        model.half()
                        print("✅ Using FP16 half precision for faster GPU inference")
                except Exception as e:
                    print(f"⚠️ Failed to use GPU: {e}")
                    print("📌 Falling back to CPU mode")
                    gpu_config['use_gpu'] = False
                    gpu_config['device'] = 'cpu'
                    gpu_config['fp16'] = False
            
        except ImportError:
            print("⚠️ Ultralytics not available, using OpenCV DNN as fallback")
            # Fallback to OpenCV DNN (still GPU-accelerated if available)
            model = load_opencv_dnn_model()
            gpu_config['use_gpu'] = False
        
        # Initialize GPU appearance tracker if available
        gpu_tracker = None
        if GPU_TRACKER_AVAILABLE and gpu_config['use_gpu']:
            try:
                gpu_tracker = GPUPersonTracker(
                    appearance_weight=0.7,
                    position_weight=0.3,
                    device=gpu_config['device'],
                    initial_track_id=next_person_id
                )
                print("✅ Using GPU Appearance Tracker for enhanced person tracking")
            except Exception as e:
                print(f"⚠️ Failed to initialize GPU tracker: {e}")
                gpu_tracker = None
        
        # Process video in batches for GPU efficiency
        batch_size = gpu_config['batch_size']
        detections = []
        frame_batch = []
        frame_numbers = []
        person_tracks = {}  # Track persons across frames
        next_person_id = get_next_person_id()  # Get global next ID instead of starting from 1
        
        print(f"🚀 Starting GPU-accelerated detection with batch size: {batch_size}")
        
        frame_count = 0
        processed_frames = 0
        written_frames = 0  # Track actual frames written to output
        
        print(f"📊 Video info: {total_frames} frames @ {fps}fps = {duration:.1f}s")
        print(f"⚡ Skip frames: {skip_frames} (processing every {skip_frames} frame{'s' if skip_frames > 1 else ''})")
        if skip_frames > 1:
            print(f"📹 Output video will be @ {output_fps}fps to maintain correct timing")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames for speed - don't write skipped frames to avoid duplication
            if frame_count % skip_frames != 0:
                frame_count += 1
                # Don't write skipped frames - this was causing frame duplication
                continue
            
            # Add frame to batch
            frame_batch.append(frame)
            frame_numbers.append(frame_count)
            
            # Process batch when full or at end of video
            if len(frame_batch) >= batch_size or frame_count == total_frames - 1:
                # Apply resource throttling before GPU operations
                if THROTTLING_ENABLED and throttler:
                    throttler.throttle()
                
                # Run batch inference
                if gpu_tracker:
                    # Use GPU appearance tracker
                    if THROTTLING_ENABLED:
                        batch_detections = safe_gpu_operation(
                            process_batch_gpu_with_tracker,
                            model, frame_batch, frame_numbers, 
                            gpu_config, gpu_tracker, fps,
                            ui_style_recognizer=ui_style_recognizer,
                            batch_size=len(frame_batch)
                        )
                    else:
                        batch_detections = process_batch_gpu_with_tracker(
                            model, frame_batch, frame_numbers, 
                            gpu_config, gpu_tracker, fps,
                            ui_style_recognizer=ui_style_recognizer
                        )
                else:
                    # Use simple position-based tracking
                    if THROTTLING_ENABLED:
                        batch_detections = safe_gpu_operation(
                            process_batch_gpu,
                            model, frame_batch, frame_numbers, 
                            gpu_config, person_tracks, next_person_id,
                            ui_style_recognizer=ui_style_recognizer,
                            batch_size=len(frame_batch)
                        )
                    else:
                        batch_detections = process_batch_gpu(
                            model, frame_batch, frame_numbers, 
                            gpu_config, person_tracks, next_person_id,
                            ui_style_recognizer=ui_style_recognizer
                        )
                
                # Update next person ID
                if batch_detections:
                    # Extract numeric IDs, handling both int and string formats
                    numeric_ids = []
                    for d in batch_detections:
                        pid = d.get('person_id', 0)
                        if isinstance(pid, str) and pid.startswith('PERSON-'):
                            try:
                                numeric_ids.append(int(pid.replace('PERSON-', '')))
                            except ValueError:
                                pass
                        elif isinstance(pid, int):
                            numeric_ids.append(pid)
                    
                    if numeric_ids:
                        max_person_id = max(numeric_ids)
                        next_person_id = max(next_person_id, max_person_id + 1)
                
                detections.extend(batch_detections)
                
                # Annotate frames and write to output
                for i, (frame, frame_num) in enumerate(zip(frame_batch, frame_numbers)):
                    # Get detections for this frame
                    frame_dets = [d for d in batch_detections if d['frame_number'] == frame_num]
                    
                    # Draw bounding boxes
                    annotated_frame = draw_detections_gpu(frame, frame_dets)
                    
                    # Write annotated frame
                    if out is not None:
                        out.write(annotated_frame)
                        written_frames += 1
                        # Store last annotated frame for interpolation
                        last_annotated_frame = annotated_frame
                
                processed_frames += len(frame_batch)
                
                # Clear batch
                frame_batch = []
                frame_numbers = []
                
                # Progress update
                progress = 10 + (processed_frames / total_frames) * 80  # 10-90% for detection
                print(f"🔄 GPU Processing: {progress:.1f}% ({processed_frames}/{total_frames} frames)")
                
                # Update database progress
                if video_id:
                    print(f"📊 Progress: {int(progress)}% - Detecting persons: {processed_frames}/{total_frames} frames")
            
            frame_count += 1
        
        # Release resources
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        
        # Clear GPU cache if used
        if gpu_config['use_gpu'] and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Verify output video
        output_size_mb = output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0
        
        print(f"✅ GPU processing complete: {len(detections)} detections")
        print(f"📁 Annotated video saved: {output_path}")
        print(f"📊 Frame statistics:")
        print(f"   - Input frames: {total_frames}")
        print(f"   - Processed frames: {processed_frames}")
        print(f"   - Written frames: {written_frames}")
        print(f"   - Output size: {output_size_mb:.1f} MB")
        
        # Verify no frame duplication
        expected_written = processed_frames  # Should match exactly
        if written_frames != expected_written:
            print(f"⚠️  WARNING: Frame count mismatch! Written: {written_frames}, Expected: {expected_written}")
        else:
            print(f"✅ Frame count verified: {written_frames} frames written correctly")
        
        # Warn if output is too large
        if output_size_mb > 1000:  # More than 1GB
            print(f"⚠️  WARNING: Output video is very large ({output_size_mb:.1f} MB)")
            print(f"   Consider using better compression or reducing quality")
        
        # Convert to web format if needed (AVI or problematic MP4)
        final_output_path = output_path
        # Always convert to ensure compression and web compatibility
        # Convert if: AVI format, large size, or always for consistent compression
        if str(output_path).endswith('.avi') or output_size_mb > 100:
            print(f"\n🔄 Converting to web-compatible format...")
            try:
                from .convert_to_web import convert_video_to_web_format
                web_path = convert_video_to_web_format(output_path)
                if web_path:
                    # Remove original and use web version
                    if output_path.exists():
                        output_path.unlink()
                    final_output_path = web_path
                    final_size_mb = final_output_path.stat().st_size / (1024 * 1024) if final_output_path.exists() else 0
                    print(f"✅ Converted to web format: {final_output_path}")
                    print(f"📊 Final size: {final_size_mb:.1f} MB (compression ratio: {input_size_mb/final_size_mb:.1f}x)")
                    
                    # Verify output is smaller than input
                    if final_size_mb >= input_size_mb:
                        print(f"⚠️  WARNING: Output ({final_size_mb:.1f} MB) is not smaller than input ({input_size_mb:.1f} MB)!")
            except Exception as e:
                print(f"⚠️  Could not convert to web format: {e}")
                print("   Video may not play in browser without FFmpeg")
        
        # Extract person images before finalizing
        print(f"\n📸 Extracting person images...")
        # Keep persons in a separate directory for organization
        persons_dir = Path('processing/outputs/persons')
        persons_dir.mkdir(parents=True, exist_ok=True)
        
        # Group detections by person_id for extraction
        person_tracks = {}
        for det in detections:
            pid = det.get('person_id', 0)
            if pid not in person_tracks:
                person_tracks[pid] = []
            # Ensure person_id is numeric for consistency
            numeric_pid = pid
            if isinstance(pid, str) and pid.startswith('PERSON-'):
                try:
                    numeric_pid = int(pid.replace('PERSON-', ''))
                except ValueError:
                    numeric_pid = 0
            elif not isinstance(pid, int):
                numeric_pid = 0
                
            person_tracks[pid].append({
                'frame_number': det['frame_number'],
                'timestamp': det['timestamp'],
                'bbox': [det['x'], det['y'], det['width'], det['height']],
                'confidence': det['confidence'],
                'person_id': numeric_pid
            })
        
        # Extract person images
        extracted_count = extract_persons_data_gpu(video_path, person_tracks, persons_dir, ui_style_recognizer)
        
        # Update progress: Finalizing
        if video_id:
            print(f"📊 Progress: 95% - Finalizing: {len(detections)} detections found")
        
        # Create processing summary
        unique_persons = len(set(d.get('person_id', 0) for d in detections if d.get('person_id')))
        person_summary = {}
        for det in detections:
            pid = det.get('person_id', 0)
            if pid not in person_summary:
                person_summary[pid] = {'count': 0, 'frames': []}
            person_summary[pid]['count'] += 1
            person_summary[pid]['frames'].append(det['frame_number'])
        
        # Update the person ID counter with the highest ID used
        if person_tracks:
            max_used_id = max(pid if isinstance(pid, int) else 0 for pid in person_tracks.keys())
            update_person_id_counter(max_used_id)
        
        # Return just the filename, not the full path
        # The database expects just the filename, not the full path
        # Use final_output_path which might be the web-converted version
        annotated_filename = final_output_path.name if isinstance(final_output_path, Path) else Path(final_output_path).name
        
        return {
            'detections': detections,
            'annotated_video_path': annotated_filename,
            'persons_dir': str(persons_dir) if persons_dir.exists() else None,
            'ocr_data': ocr_data,  # Include OCR extracted data
            'processing_summary': {
                'total_frames': total_frames,
                'processed_frames': processed_frames,
                'total_detections': len(detections),
                'total_persons': unique_persons,
                'duration': duration,
                'fps': fps,
                'resolution': f"{width}x{height}",
                'person_summary': person_summary,
                'gpu_used': gpu_config['use_gpu'],
                'batch_size': batch_size,
                'skip_frames': skip_frames,
                'persons_extracted': extracted_count if 'extracted_count' in locals() else 0,
                'ocr_location': ocr_data.get('location') if ocr_data else None,
                'ocr_video_date': ocr_data.get('video_date') if ocr_data else None,
                'ocr_confidence': ocr_data.get('confidence') if ocr_data else None
            }
        }
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ GPU detection error: {str(e)}")
        print(f"📋 Trace: {error_trace}")
        return {'error': str(e), 'trace': error_trace}


def process_batch_gpu(model, frames, frame_numbers, gpu_config, person_tracks, next_person_id, ui_style_recognizer=None):
    """
    Process a batch of frames on GPU for efficiency
    """
    detections = []
    
    try:
        # Add small delay to prevent GPU overload
        if THROTTLING_ENABLED and throttler and len(frames) > 4:
            time.sleep(0.02 * len(frames))  # 20ms per frame
        
        # Run batch inference
        if hasattr(model, 'predict'):  # YOLO model
            # Process all frames at once on GPU
            results = model.predict(
                frames, 
                stream=False, 
                conf=0.5,  # Higher confidence for speed
                classes=[0],  # Person class only
                device=gpu_config['device'],
                verbose=False
            )
            
            # Extract detections from results
            for frame_idx, (result, frame_num) in enumerate(zip(results, frame_numbers)):
                timestamp = frame_num / 30.0  # Assuming 30 fps
                
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        # Calculate bounding box dimensions
                        bbox_width = int(x2 - x1)
                        bbox_height = int(y2 - y1)
                        
                        # QUALITY FILTER: Skip persons with bounding box width < 128 pixels
                        # Small bounding boxes typically contain low-quality person images
                        # that are not suitable for face recognition training
                        MIN_BBOX_WIDTH = 128
                        
                        if bbox_width < MIN_BBOX_WIDTH:
                            print(f"⚠️ Skipping person detection: bbox width {bbox_width}px < {MIN_BBOX_WIDTH}px (too small for quality face recognition)")
                            continue
                        
                        # Simple tracking based on position
                        person_id = assign_person_id(
                            x1, y1, x2, y2, frame_num, 
                            person_tracks, next_person_id
                        )
                        
                        detection = {
                            'frame_number': frame_num,
                            'timestamp': timestamp,
                            'x': int(x1),
                            'y': int(y1),
                            'width': bbox_width,
                            'height': bbox_height,
                            'confidence': confidence,
                            'person_id': person_id,
                            'track_id': f"TRACK-{person_id:04d}"
                        }
                        detections.append(detection)
        
        else:  # OpenCV DNN fallback
            for frame, frame_num in zip(frames, frame_numbers):
                frame_detections = detect_with_opencv_dnn(model, frame, frame_num)
                
                # Convert OpenCV format to our standard format
                for det in frame_detections:
                    timestamp = frame_num / 30.0  # Assuming 30 fps
                    x1, y1, x2, y2 = det['bbox']
                    
                    # Calculate bounding box dimensions
                    bbox_width = int(x2 - x1)
                    bbox_height = int(y2 - y1)
                    
                    # QUALITY FILTER 1: Skip persons with bounding box width < 128 pixels
                    # Small bounding boxes typically contain low-quality person images
                    # that are not suitable for face recognition training
                    MIN_BBOX_WIDTH = 128
                    
                    if bbox_width < MIN_BBOX_WIDTH:
                        logger.info(f"Skipping person detection: bbox width {bbox_width}px < {MIN_BBOX_WIDTH}px (too small for quality face recognition)")
                        continue
                    
                    # QUALITY FILTER 2: Skip persons where height < 2 * width
                    # This filters out poor detections where people are lying down or have incorrect bbox shapes
                    # A standing person typically has height/width ratio between 2.0 to 3.0
                    MIN_HEIGHT_WIDTH_RATIO = 2.0
                    
                    if bbox_height < (MIN_HEIGHT_WIDTH_RATIO * bbox_width):
                        logger.info(f"Skipping person detection: bbox height {bbox_height}px < {MIN_HEIGHT_WIDTH_RATIO} * width {bbox_width}px (incorrect aspect ratio)")
                        continue
                    
                    # Simple tracking based on position
                    person_id = assign_person_id(
                        x1, y1, x2, y2, frame_num,
                        person_tracks, next_person_id
                    )
                    
                    detection = {
                        'frame_number': frame_num,
                        'timestamp': timestamp,
                        'x': int(x1),
                        'y': int(y1),
                        'width': bbox_width,
                        'height': bbox_height,
                        'confidence': det.get('confidence', 0.8),
                        'person_id': person_id,
                        'track_id': f"TRACK-{person_id:04d}"
                    }
                    detections.append(detection)
    
    except Exception as e:
        print(f"⚠️ Batch processing error: {e}")
    
    return detections


def assign_person_id(x1, y1, x2, y2, frame_num, person_tracks, next_person_id):
    """
    Improved person tracking based on position proximity and size
    """
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    
    # Check if this detection matches any existing track
    best_score = float('inf')
    matched_id = None
    
    for person_id, track_info in person_tracks.items():
        last_frame = track_info['last_frame']
        last_center = track_info['last_center']
        last_size = track_info.get('last_size', (width, height))
        
        # Only consider tracks from recent frames
        frame_diff = frame_num - last_frame
        if frame_diff > 30:  # Lost track after 1 second (assuming 30fps)
            continue
            
        # Calculate position distance
        pos_distance = ((center_x - last_center[0])**2 + (center_y - last_center[1])**2)**0.5
        
        # Calculate size difference (helps when people cross)
        size_diff = abs(width - last_size[0]) + abs(height - last_size[1])
        
        # Calculate movement speed (pixels per frame)
        speed = pos_distance / max(frame_diff, 1)
        
        # Reasonable movement speed: humans typically move < 50 pixels/frame
        if speed > 50:
            continue
            
        # Combined score (lower is better)
        score = pos_distance + size_diff * 0.5
        
        if score < best_score and pos_distance < 150:  # Increased threshold
            best_score = score
            matched_id = person_id
    
    if matched_id:
        # Update existing track
        person_tracks[matched_id]['last_frame'] = frame_num
        person_tracks[matched_id]['last_center'] = (center_x, center_y)
        person_tracks[matched_id]['last_size'] = (width, height)
        return matched_id
    else:
        # Create new track with unique ID
        new_id = next_person_id
        person_tracks[new_id] = {
            'last_frame': frame_num,
            'last_center': (center_x, center_y),
            'last_size': (width, height)
        }
        return new_id


def draw_detections_gpu(frame, detections):
    """
    Draw bounding boxes and labels on frame (optimized for GPU processing)
    """
    # Create a copy to avoid modifying original
    annotated = frame.copy()
    
    for det in detections:
        x, y = det['x'], det['y']
        w, h = det['width'], det['height']
        person_id = det.get('person_id', 0)
        confidence = det['confidence']
        
        # Color based on person ID
        color = get_color_for_person(person_id)
        
        # Draw bounding box with thicker lines for visibility
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)
        
        # Create label in PERSON-XXXX format
        label = f"PERSON-{person_id:04d}"
        
        # Larger font size and thickness for better visibility
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8  # Increased from 0.5
        font_thickness = 2  # Increased from 1
        
        # Get text size
        label_size, baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        # Add padding around text
        padding = 5
        
        # Background rectangle for text (positioned above bounding box)
        label_y_top = y - label_size[1] - padding * 2
        if label_y_top < 0:  # If label would go off screen, put it inside box
            label_y_top = y + padding
            label_y_bottom = y + label_size[1] + padding * 2
        else:
            label_y_bottom = y
            
        cv2.rectangle(annotated, 
                     (x - padding, label_y_top),
                     (x + label_size[0] + padding, label_y_bottom),
                     color, -1)
        
        # Draw text in white for contrast
        text_y = label_y_bottom - padding if label_y_top < y else y - padding
        cv2.putText(annotated, label,
                   (x, text_y),
                   font,
                   font_scale, 
                   (255, 255, 255),  # White text
                   font_thickness,
                   cv2.LINE_AA)  # Anti-aliased for smoother text
        
        # Optional: Add confidence as smaller text below
        if confidence < 0.9:  # Only show confidence if it's not very high
            conf_text = f"{confidence:.0%}"
            conf_scale = 0.5
            conf_thickness = 1
            conf_size, _ = cv2.getTextSize(conf_text, font, conf_scale, conf_thickness)
            
            cv2.putText(annotated, conf_text,
                       (x + w - conf_size[0], y + h - 5),
                       font,
                       conf_scale,
                       color,
                       conf_thickness,
                       cv2.LINE_AA)
    
    return annotated


def get_color_for_person(person_id):
    """
    Generate consistent color for each person ID
    """
    # Bright, high-contrast colors that stand out well
    colors = [
        (255, 0, 0),      # Bright Red
        (0, 255, 0),      # Bright Green
        (0, 100, 255),    # Bright Blue
        (255, 255, 0),    # Yellow
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Cyan
        (255, 128, 0),    # Orange
        (128, 0, 255),    # Purple
        (255, 255, 255),  # White
        (0, 255, 128),    # Spring Green
    ]
    return colors[person_id % len(colors)]


def load_opencv_dnn_model():
    """
    Load OpenCV DNN model as fallback (can still use GPU via OpenCL)
    """
    try:
        # Try to use MobileNet SSD for person detection
        prototxt = "deploy.prototxt"
        model_path = "mobilenet_iter_73000.caffemodel"
        
        # Check if model files exist
        if os.path.exists(prototxt) and os.path.exists(model_path):
            net = cv2.dnn.readNetFromCaffe(prototxt, model_path)
            # Try to use OpenCL for GPU acceleration
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
            return net
        else:
            print("⚠️ OpenCV DNN model files not found")
            return None
    except Exception as e:
        print(f"⚠️ Failed to load OpenCV DNN model: {e}")
        return None


def detect_with_opencv_dnn(model, frame, frame_num):
    """
    Fallback detection using OpenCV's cascade classifier
    """
    if model is None:
        # Use HOG person detector as ultimate fallback
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Detect people in the frame
        (rects, weights) = hog.detectMultiScale(frame, 
                                               winStride=(4, 4),
                                               padding=(8, 8),
                                               scale=1.05)
        
        detections = []
        for i, (x, y, w, h) in enumerate(rects):
            if weights[i] > 0.5:  # Confidence threshold
                detections.append({
                    'frame': frame_num,
                    'person_id': i,
                    'bbox': [x, y, x + w, y + h],
                    'confidence': float(weights[i])
                })
        return detections
    
    # If we have a DNN model, use it
    # This is a placeholder for actual DNN inference
    return []


def validate_and_merge_tracks(person_tracks):
    """
    Validate person tracks and merge potential duplicates
    """
    # Group tracks by similar appearance timeframes
    merged_tracks = {}
    
    for person_id, detections in person_tracks.items():
        if not detections:
            continue
            
        # Sort detections by frame number
        detections.sort(key=lambda d: d['frame_number'])
        
        # Check if this track might be a duplicate of an existing one
        is_duplicate = False
        
        for existing_id, existing_detections in merged_tracks.items():
            # Check temporal overlap
            existing_frames = set(d['frame_number'] for d in existing_detections)
            current_frames = set(d['frame_number'] for d in detections)
            
            # If there's significant frame overlap, likely the same person
            overlap = len(existing_frames & current_frames)
            if overlap > min(len(existing_frames), len(current_frames)) * 0.8:  # Increased from 0.3 to prevent merging different people
                # Merge into existing track
                print(f"🔄 Merging track {person_id} into {existing_id} (overlap: {overlap} frames)")
                existing_detections.extend(detections)
                is_duplicate = True
                break
        
        if not is_duplicate:
            merged_tracks[person_id] = detections
    
    # Re-sort all detections in merged tracks
    for person_id in merged_tracks:
        merged_tracks[person_id].sort(key=lambda d: d['frame_number'])
    
    return merged_tracks


def extract_persons_data_gpu(video_path, person_tracks, persons_dir, ui_style_recognizer=None):
    """
    Extract person images and metadata to PERSON-XXXX folders
    Optimized version for GPU processing pipeline
    
    Args:
        video_path: Path to the video file
        person_tracks: Dictionary of person tracks with detections
        persons_dir: Directory to save person data
        ui_style_recognizer: Optional PersonRecognitionInferenceSimple instance for recognition
    """
    print(f"📸 Extracting person data to {persons_dir}")
    
    # Import recognition module if needed
    if ui_style_recognizer is not None:
        print("✅ Person recognition enabled during extraction")
        import tempfile
        import os as temp_os
    
    # Validate and merge potential duplicate tracks
    person_tracks = validate_and_merge_tracks(person_tracks)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video for person extraction: {video_path}")
        return
    
    extracted_count = 0
    
    for person_id, detections in person_tracks.items():
        # First, try to recognize this person before creating folder
        recognized_person_id = None
        recognition_confidence = 0.0
        
        if ui_style_recognizer and len(detections) > 0:
            print(f"🔍 Attempting recognition for tracked person {person_id}...")
            
            # Try recognition on first valid detection
            for i, detection in enumerate(detections[:3]):  # Try up to 3 frames
                frame_number = detection["frame_number"]
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                
                if ret:
                    x, y, w, h = detection["bbox"]
                    
                    # Skip small bounding boxes
                    if w < 128:
                        continue
                    
                    # Skip incorrect aspect ratios
                    if h < (2.0 * w):
                        continue
                    
                    # Extract person region
                    padding = 10
                    x1 = max(0, int(x - padding))
                    y1 = max(0, int(y - padding))
                    x2 = min(frame.shape[1], int(x + w + padding))
                    y2 = min(frame.shape[0], int(y + h + padding))
                    
                    person_img = frame[y1:y2, x1:x2]
                    
                    if person_img.size > 0:
                        try:
                            # Try recognition
                            if hasattr(ui_style_recognizer, 'recognize_person'):
                                # VenvRecognitionWrapper
                                result = ui_style_recognizer.recognize_person(person_img, 0.8)
                                if result:
                                    recognized_person_id = result['person_id']
                                    recognition_confidence = result['confidence']
                            else:
                                # Direct recognizer
                                temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                                cv2.imwrite(temp_file.name, person_img)
                                temp_file.close()
                                
                                result = ui_style_recognizer.process_cropped_image(temp_file.name)
                                os.unlink(temp_file.name)
                                
                                if result and result.get('persons'):
                                    first_person = result['persons'][0]
                                    if first_person['person_id'] != 'unknown' and first_person['confidence'] >= 0.8:
                                        recognized_person_id = first_person['person_id']
                                        recognition_confidence = first_person['confidence']
                            
                            if recognized_person_id:
                                print(f"✅ Pre-recognition successful: {recognized_person_id} ({recognition_confidence:.2%})")
                                break
                                
                        except Exception as e:
                            print(f"⚠️ Pre-recognition error: {e}")
            
            # Reset video position
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Now determine folder name based on recognition
        if recognized_person_id:
            person_id_str = recognized_person_id
            print(f"🎯 Using recognized ID for folder: {person_id_str}")
        else:
            if isinstance(person_id, int):
                person_id_str = f"PERSON-{person_id:04d}"
            else:
                person_id_str = str(person_id)
            print(f"🆕 Creating new person folder: {person_id_str}")
        
        person_dir = persons_dir / person_id_str
        person_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract all detection frames with intelligent sampling
        # Sample every N frames to avoid storing redundant consecutive frames
        # This reduces storage while maintaining diversity
        FRAME_SAMPLE_INTERVAL = 5  # Extract every 5th frame (approx 6 images per second at 30fps)
        
        # If person appears briefly, extract all frames
        if len(detections) <= 30:  # Less than 1 second of appearance
            sample_detections = detections
        else:
            # Sample frames at regular intervals
            sample_detections = detections[::FRAME_SAMPLE_INTERVAL]
            # Always include first and last detection
            if detections[0] not in sample_detections:
                sample_detections.insert(0, detections[0])
            if detections[-1] not in sample_detections:
                sample_detections.append(detections[-1])
        
        person_metadata = {
            "person_id": person_id_str,
            "original_tracking_id": person_id,
            "recognized": recognized_person_id is not None,
            "recognition_confidence": float(recognition_confidence) if recognized_person_id else 0,
            "total_detections": len(detections),
            "first_appearance": detections[0]["timestamp"],
            "last_appearance": detections[-1]["timestamp"],
            "avg_confidence": sum(d["confidence"] for d in detections) / len(detections),
            "images": [],
            "created_at": datetime.now().isoformat()
        }
        
        for i, detection in enumerate(sample_detections):
            frame_number = detection["frame_number"]
            
            # Seek to specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret:
                x, y, w, h = detection["bbox"]
                
                # QUALITY FILTER 1: Skip persons with bounding box width < 128 pixels
                # Small bounding boxes typically contain low-quality person images
                # that are not suitable for face recognition training
                MIN_BBOX_WIDTH = 128
                
                if w < MIN_BBOX_WIDTH:
                    logger.info(f"Skipping {person_id_str} frame {frame_number}: bbox width {w}px < {MIN_BBOX_WIDTH}px (too small for quality face recognition)")
                    continue
                
                # QUALITY FILTER 2: Skip persons where height < 2 * width
                # This filters out poor detections where people are lying down or have incorrect bbox shapes
                MIN_HEIGHT_WIDTH_RATIO = 2.0
                
                if h < (MIN_HEIGHT_WIDTH_RATIO * w):
                    logger.info(f"Skipping {person_id_str} frame {frame_number}: bbox height {h}px < {MIN_HEIGHT_WIDTH_RATIO} * width {w}px (incorrect aspect ratio)")
                    continue
                
                # Extract person region with some padding
                padding = 10
                x1 = max(0, int(x - padding))
                y1 = max(0, int(y - padding))
                x2 = min(frame.shape[1], int(x + w + padding))
                y2 = min(frame.shape[0], int(y + h + padding))
                
                person_img = frame[y1:y2, x1:x2]
                
                if person_img.size > 0:
                    # Use simple UUID for filename
                    img_filename = f"{uuid.uuid4()}.jpg"
                    img_path = person_dir / img_filename
                    cv2.imwrite(str(img_path), person_img)
                    
                    image_metadata = {
                        "filename": img_filename,
                        "frame_number": frame_number,
                        "timestamp": detection["timestamp"],
                        "confidence": detection["confidence"],
                        "bbox": detection["bbox"]
                    }
                    
                    # Add recognition info if available
                    if recognized_person_id:
                        image_metadata["recognized_as"] = recognized_person_id
                        image_metadata["recognition_confidence"] = float(recognition_confidence)
                    
                    person_metadata["images"].append(image_metadata)
        
        # Save metadata
        metadata_path = person_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(person_metadata, f, indent=2)
        
        if len(person_metadata["images"]) > 0:
            extracted_count += 1
            print(f"✅ Created {person_id_str} folder with {len(person_metadata['images'])} images (from {len(detections)} detections)")
        else:
            print(f"⚠️ No valid images for {person_id_str} (all too small)")
    
    cap.release()
    print(f"📸 Extracted {extracted_count} persons with valid images")
    return extracted_count



def process_batch_gpu_with_tracker(model, frames, frame_numbers, gpu_config, gpu_tracker, fps, ui_style_recognizer=None):
    """
    Process a batch of frames using GPU appearance tracker
    """
    detections = []
    
    try:
        # First, detect persons in all frames
        raw_detections_by_frame = {}
        
        if hasattr(model, "predict"):  # YOLO model
            # Process all frames at once on GPU
            results = model.predict(
                frames, 
                stream=False, 
                conf=0.5,  # Higher confidence for speed
                classes=[0],  # Person class only
                device=gpu_config["device"],
                verbose=False
            )
            
            # Extract detections from results
            for frame_idx, (result, frame_num) in enumerate(zip(results, frame_numbers)):
                frame_detections = []
                
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        # Calculate bounding box dimensions
                        bbox_width = int(x2 - x1)
                        bbox_height = int(y2 - y1)
                        
                        # QUALITY FILTER: Skip persons with bounding box width < 128 pixels
                        MIN_BBOX_WIDTH = 128
                        
                        if bbox_width < MIN_BBOX_WIDTH:
                            continue
                        
                        frame_detections.append({
                            "bbox": [float(x1), float(y1), float(bbox_width), float(bbox_height)],
                            "confidence": confidence,
                            "frame_number": frame_num,
                            "timestamp": frame_num / fps
                        })
                
                raw_detections_by_frame[frame_idx] = frame_detections
        
        # Process each frame with the tracker
        for frame_idx, (frame, frame_num) in enumerate(zip(frames, frame_numbers)):
            frame_detections = raw_detections_by_frame.get(frame_idx, [])
            
            # Update tracker with appearance features
            tracked_detections = gpu_tracker.update(frame_detections, frame, frame_num)
            
            # Convert to our standard format
            for det in tracked_detections:
                detection = {
                    "frame_number": det["frame_number"],
                    "timestamp": det["timestamp"],
                    "x": int(det["bbox"][0]),
                    "y": int(det["bbox"][1]),
                    "width": int(det["bbox"][2]),
                    "height": int(det["bbox"][3]),
                    "confidence": det["confidence"],
                    "person_id": det.get("person_id", det.get("track_id", 0)),
                    "track_id": det.get("track_id", det.get("person_id", 0))
                }
                detections.append(detection)
        
        return detections
        
    except Exception as e:
        print(f"⚠️ GPU tracker error: {e}, falling back to simple tracking")
        # Fall back to simple tracking
        return process_batch_gpu(model, frames, frame_numbers, gpu_config, {}, 1)
