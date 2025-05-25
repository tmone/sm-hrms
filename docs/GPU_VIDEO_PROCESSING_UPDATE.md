# GPU-Accelerated Video Processing Update

## Summary of Changes

### 1. **Auto-Process on Upload**
- When a video is uploaded, it now automatically starts person extraction
- No need to manually click "Process Video" anymore
- The system immediately begins detecting and tracking persons in the video

### 2. **GPU Acceleration Support**
- Added GPU-optimized detection module (`processing/gpu_enhanced_detection.py`)
- Automatically detects and uses CUDA if available
- Falls back to CPU processing if GPU is not available
- Key optimizations:
  - Batch processing (8 frames at once on GPU)
  - Half-precision (FP16) inference for faster processing
  - Frame skipping for long videos (process every 2nd frame for videos > 60s)
  - Efficient memory management with GPU cache clearing

### 3. **Annotated Video Output**
- The system now creates an annotated video with bounding boxes
- This annotated video is used for playback (no need for conversion)
- Shows person tracking IDs and confidence scores
- Web-compatible MP4 format

### 4. **Removed Video Conversion**
- Since we use the annotated video for playback, video conversion is no longer needed
- This simplifies the workflow and reduces processing time

## How to Use

### 1. Run Database Migration
```bash
python migrate_add_annotated_video_path.py
```

### 2. Install GPU Dependencies (if you have NVIDIA GPU)
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install YOLOv8 with GPU support
pip install ultralytics

# Verify GPU is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 3. Upload Video
- Simply upload a video through the web interface
- Person extraction starts automatically
- Monitor progress on the video detail page
- Once complete, the annotated video will play showing detected persons

## GPU Performance Tips

1. **Batch Size**: Adjust in `gpu_config['batch_size']` (default: 8)
   - Increase for more GPU memory and faster processing
   - Decrease if you get out-of-memory errors

2. **Frame Skipping**: For long videos, the system automatically skips frames
   - Videos > 60s process every 2nd frame
   - Adjust `skip_frames` variable in `gpu_enhanced_detection.py`

3. **Model Selection**: 
   - Default: YOLOv8n (nano) for speed
   - Can change to YOLOv8s/m/l for better accuracy

## Workflow Changes

### Old Workflow:
1. Upload video
2. Check if conversion needed
3. Convert if necessary
4. Manually click "Process Video"
5. Wait for person extraction
6. View results

### New Workflow:
1. Upload video
2. Automatic person extraction with GPU
3. View annotated video with results

## Technical Details

### GPU Detection Process:
1. Load video and get properties
2. Initialize YOLO model on GPU
3. Process frames in batches
4. Track persons across frames
5. Draw bounding boxes
6. Save annotated video
7. Extract person data for face recognition

### Database Changes:
- Added `annotated_video_path` field to Video model
- Stores path to the annotated video
- Used for video playback instead of original/converted video

### File Structure:
```
processing/outputs/
├── {video_name}_annotated_{timestamp}.mp4  # Annotated video
└── persons/
    ├── PERSON-0001/
    │   ├── metadata.json
    │   └── frame_images...
    └── PERSON-0002/
        ├── metadata.json
        └── frame_images...
```