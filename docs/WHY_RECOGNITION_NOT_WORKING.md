# Why Recognition Is Not Working

## The Clear Answer

You're using **GPU detection** (`gpu_enhanced_detection.py`) which does NOT include recognition code. It simply creates new PERSON-XXXX IDs for everyone.

## What's Happening

1. **Your Flow**:
   ```
   Video → GPU Detection → Creates PERSON-0022, PERSON-0023 (NEW IDs)
   ```

2. **What Should Happen**:
   ```
   Video → Detection → Recognition → Reuse PERSON-0001, etc (EXISTING IDs)
   ```

## Why GPU Detection Doesn't Recognize

Looking at `gpu_enhanced_detection.py`:
- The `assign_person_id()` function only tracks by position
- It never attempts to recognize faces
- It just assigns sequential new IDs

## Solutions

### Option 1: Use Chunked Processor (Has Recognition)
Instead of GPU detection, use the chunked processor which includes recognition:
```python
# In videos.py, replace:
result = gpu_person_detection_task(video_path, gpu_config, video_obj.id, app)

# With:
from processing.chunked_video_processor import ChunkedVideoProcessor
processor = ChunkedVideoProcessor()
result = processor.process_video(video_path)
```

### Option 2: Add Recognition to GPU Detection
The GPU detection needs to be modified to:
1. Load the recognition model
2. For each detected person, try recognition BEFORE assigning ID
3. Only create new ID if recognition fails

### Option 3: Quick Workaround
After GPU detection, run a separate recognition pass:
```python
# After GPU detection
from scripts.fix_person_ids_after_detection import fix_person_ids
fix_person_ids(video_path, result['detections'])
```

## The Real Issue

Your system has TWO separate processing paths:
1. **Chunked Processor** - Has recognition, but maybe not being used
2. **GPU Detection** - No recognition, currently being used

That's why:
- UI test works (uses recognition model directly)
- Video processing doesn't recognize (uses GPU detection without recognition)

## Immediate Fix

To enable recognition RIGHT NOW:

1. Check which processor is being used:
   ```python
   # In hr_management/blueprints/videos.py
   # Look for: gpu_person_detection_task
   ```

2. Switch to chunked processor OR add recognition to GPU detection

3. Or manually merge duplicate persons after processing