"""
GPU-accelerated person detection module for faster video processing
"""
import cv2
import numpy as np
from datetime import datetime
import os
from pathlib import Path

# Try to import torch and check CUDA availability
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
    if not CUDA_AVAILABLE:
        print("⚠️ CUDA not available - will use CPU for processing")
except ImportError:
    print("⚠️ PyTorch not available - will use OpenCV DNN")
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

def update_video_progress(video_id, progress, message="Processing...", app=None):
    """Update video processing progress in database"""
    try:
        # Try to get app context
        if app:
            with app.app_context():
                db = app.db
                Video = app.Video
                video = Video.query.get(video_id)
                if video:
                    video.processing_progress = int(progress)
                    video.processing_log = f"{video.processing_log}\n[{datetime.now().strftime('%H:%M:%S')}] {message} ({progress}%)" if video.processing_log else f"[{datetime.now().strftime('%H:%M:%S')}] {message} ({progress}%)"
                    db.session.commit()
                    print(f"📊 Progress updated: {progress}% - {message}")
        else:
            print(f"⚠️ No app context for progress update: {progress}% - {message}")
    except Exception as e:
        print(f"⚠️ Could not update progress: {e}")

def gpu_person_detection_task(video_path, gpu_config=None, video_id=None, app=None):
    """
    GPU-accelerated person detection with optimizations for large videos
    """
    try:
        # Default GPU configuration
        if gpu_config is None:
            gpu_config = {
                'use_gpu': CUDA_AVAILABLE,
                'batch_size': 8 if CUDA_AVAILABLE else 4,
                'device': 'cuda:0' if CUDA_AVAILABLE else 'cpu',
                'fp16': CUDA_AVAILABLE,
                'num_workers': 4
            }
        
        print(f"🎮 GPU Detection Config: {gpu_config}")
        print(f"📊 CUDA Available: {CUDA_AVAILABLE}")
        if CUDA_AVAILABLE and TORCH_AVAILABLE:
            print(f"🎮 GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Update progress: Initializing
        if video_id:
            update_video_progress(video_id, 5, "Initializing GPU detection...", app)
        
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
        
        print(f"📹 Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames, {duration:.1f}s")
        
        # Update progress: Video loaded
        if video_id:
            update_video_progress(video_id, 10, f"Video loaded: {total_frames} frames, {duration:.1f}s", app)
        
        # Create output directory for annotated video
        output_dir = Path('processing/outputs')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        video_name = Path(video_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{video_name}_annotated_{timestamp}.mp4"
        
        # Initialize video writer with optimal compression
        # Try to use FFmpeg for better compression if available
        out = None
        
        try:
            # Import our optimized video writer
            from .video_writer_ffmpeg import create_video_writer
            
            # Create video writer with optimal settings
            out = create_video_writer(output_path, fps, width, height, use_ffmpeg=True)
            
            # For FFmpeg writer, we need to open it
            if hasattr(out, 'open'):
                if not out.open():
                    raise Exception("Failed to open FFmpeg writer")
            
            print(f"✅ Video writer initialized successfully")
            
        except Exception as e:
            print(f"⚠️  Failed to use optimized writer: {e}")
            print("🔄 Falling back to standard OpenCV writer...")
            
            # Fallback to standard OpenCV H.264
            # Calculate target bitrate based on resolution and fps
            target_bitrate = int(width * height * fps * 0.05)  # ~2-5 Mbps for HD
            
            # Try different H.264 codec options
            h264_codecs = [
                ('H264', 'H.264 (OpenCV)'),
                ('h264', 'H.264 (lowercase)'),
                ('avc1', 'H.264/AVC (Apple)'),
                ('x264', 'x264 (libx264)'),
                ('mp4v', 'MPEG-4 (fallback)')
            ]
            
            # First try with .mp4 extension (best for H.264)
            for codec, codec_name in h264_codecs:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height), True)
                    
                    # Try to set compression parameters if available
                    if out.isOpened():
                        # Set video quality (0-100, higher is better quality but larger file)
                        out.set(cv2.VIDEOWRITER_PROP_QUALITY, 80)
                        # Set bitrate if supported
                        if hasattr(cv2, 'VIDEOWRITER_PROP_BITRATE'):
                            out.set(cv2.VIDEOWRITER_PROP_BITRATE, target_bitrate)
                        
                        print(f"✅ Using {codec_name} codec for web-compatible output")
                        print(f"📹 Output settings: {width}x{height} @ {fps}fps, codec: {codec}")
                        print(f"🎯 Target bitrate: {target_bitrate/1_000_000:.2f} Mbps")
                        break
                    else:
                        out.release()
                        out = None
                except Exception as e:
                    print(f"⚠️ Failed to use {codec_name} codec: {e}")
                    continue
        
        # If H.264 codecs fail, try XVID in AVI container (better compression than MJPEG)
        if out is None:
            avi_output_path = str(output_path).replace('.mp4', '.avi')
            print(f"🔄 Trying AVI container with XVID codec: {avi_output_path}")
            
            try:
                # Use XVID for better compression than MJPEG
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(avi_output_path, fourcc, fps, (width, height), True)
                if out.isOpened():
                    output_path = Path(avi_output_path)
                    print(f"✅ Using XVID codec in AVI container (compressed)")
                    print(f"📹 Output settings: {width}x{height} @ {fps}fps")
                else:
                    out.release()
                    out = None
            except Exception as e:
                print(f"⚠️ Failed to use AVI container: {e}")
        
        if out is None:
            print("❌ ERROR: Could not initialize video writer with any codec!")
            print("💡 Tip: Install ffmpeg and opencv-python-headless for better codec support")
            return {'error': 'Failed to initialize video writer for web-compatible output'}
        
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
        
        # Process video in batches for GPU efficiency
        batch_size = gpu_config['batch_size']
        detections = []
        frame_batch = []
        frame_numbers = []
        person_tracks = {}  # Track persons across frames
        next_person_id = 1
        
        print(f"🚀 Starting GPU-accelerated detection with batch size: {batch_size}")
        
        frame_count = 0
        processed_frames = 0
        written_frames = 0  # Track actual frames written to output
        
        # Process every nth frame for speed (adjust based on video length)
        skip_frames = 2 if total_frames > 1800 else 1  # Skip frames for videos > 60s @ 30fps
        last_annotated_frame = None  # Store last annotated frame for interpolation
        
        print(f"📊 Video info: {total_frames} frames @ {fps}fps = {duration:.1f}s")
        print(f"⚡ Skip frames: {skip_frames} (processing every {skip_frames} frame{'s' if skip_frames > 1 else ''})")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames for speed
            if frame_count % skip_frames != 0:
                frame_count += 1
                # Still write the frame to output (without new annotations)
                if out is not None and last_annotated_frame is not None:
                    out.write(last_annotated_frame)
                    written_frames += 1
                continue
            
            # Add frame to batch
            frame_batch.append(frame)
            frame_numbers.append(frame_count)
            
            # Process batch when full or at end of video
            if len(frame_batch) >= batch_size or frame_count == total_frames - 1:
                # Run batch inference
                batch_detections = process_batch_gpu(
                    model, frame_batch, frame_numbers, 
                    gpu_config, person_tracks, next_person_id
                )
                
                # Update next person ID
                if batch_detections:
                    max_person_id = max(d.get('person_id', 0) for d in batch_detections)
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
                    update_video_progress(video_id, int(progress), f"Detecting persons: {processed_frames}/{total_frames} frames", app)
            
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
        
        # Warn if output is too large
        if output_size_mb > 1000:  # More than 1GB
            print(f"⚠️  WARNING: Output video is very large ({output_size_mb:.1f} MB)")
            print(f"   Consider using better compression or reducing quality")
        
        # Update progress: Finalizing
        if video_id:
            update_video_progress(video_id, 95, f"Finalizing: {len(detections)} detections found", app)
        
        # Create processing summary
        unique_persons = len(set(d.get('person_id', 0) for d in detections if d.get('person_id')))
        person_summary = {}
        for det in detections:
            pid = det.get('person_id', 0)
            if pid not in person_summary:
                person_summary[pid] = {'count': 0, 'frames': []}
            person_summary[pid]['count'] += 1
            person_summary[pid]['frames'].append(det['frame_number'])
        
        # Return just the filename, not the full path
        # The database expects just the filename, not the full path
        annotated_filename = output_path.name if isinstance(output_path, Path) else Path(output_path).name
        
        return {
            'detections': detections,
            'annotated_video_path': annotated_filename,
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
                'skip_frames': skip_frames
            }
        }
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ GPU detection error: {str(e)}")
        print(f"📋 Trace: {error_trace}")
        return {'error': str(e), 'trace': error_trace}


def process_batch_gpu(model, frames, frame_numbers, gpu_config, person_tracks, next_person_id):
    """
    Process a batch of frames on GPU for efficiency
    """
    detections = []
    
    try:
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
                            'width': int(x2 - x1),
                            'height': int(y2 - y1),
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
                        'width': int(x2 - x1),
                        'height': int(y2 - y1),
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
    Simple person tracking based on position proximity
    """
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Check if this detection matches any existing track
    min_distance = float('inf')
    matched_id = None
    
    for person_id, track_info in person_tracks.items():
        last_frame = track_info['last_frame']
        last_center = track_info['last_center']
        
        # Only consider tracks from recent frames (within 10 frames)
        if frame_num - last_frame < 10:
            distance = ((center_x - last_center[0])**2 + (center_y - last_center[1])**2)**0.5
            if distance < min_distance and distance < 100:  # Max 100 pixel movement
                min_distance = distance
                matched_id = person_id
    
    if matched_id:
        # Update existing track
        person_tracks[matched_id]['last_frame'] = frame_num
        person_tracks[matched_id]['last_center'] = (center_x, center_y)
        return matched_id
    else:
        # Create new track
        new_id = len(person_tracks) + 1
        person_tracks[new_id] = {
            'last_frame': frame_num,
            'last_center': (center_x, center_y)
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
        
        # Draw bounding box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        
        # Draw label with background
        label = f"Person {person_id} ({confidence:.2f})"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Background rectangle for text
        cv2.rectangle(annotated, 
                     (x, y - label_size[1] - 4),
                     (x + label_size[0], y),
                     color, -1)
        
        # Draw text
        cv2.putText(annotated, label,
                   (x, y - 2),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 255, 255), 1)
    
    return annotated


def get_color_for_person(person_id):
    """
    Generate consistent color for each person ID
    """
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
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