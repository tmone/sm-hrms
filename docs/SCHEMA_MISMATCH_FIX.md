# Video Detection Database Schema Mismatch Fix

## Problem Summary
The HR Management system had a critical issue where person detections were being visually displayed in the video interface (green bounding boxes with PERSON-0001, PERSON-0002 labels) but were not being saved to the database. This was causing a disconnect between what users saw and what was actually stored.

## Root Cause Analysis
The issue was caused by a **schema mismatch** between detection generation and database storage:

### Conflicting Save Functions
1. **Correct Function** (`hr_management/processing/tasks.py`):
   - Uses proper DetectedPerson model schema
   - Fields: `person_code`, `start_frame`, `end_frame`, `start_time`, `end_time`, `bbox_data` (JSON)
   - Matches the main DetectedPerson model in `hr_management/models/video.py`

2. **Incorrect Function** (`hr_management/processing/standalone_tasks.py`):
   - Uses incompatible legacy schema
   - Fields: `timestamp`, `frame_number`, `bbox_x`, `bbox_y`, `bbox_width`, `bbox_height`
   - Does not match the current DetectedPerson model

### The Bug Location
In `hr_management/blueprints/videos.py` line 2785, the code was incorrectly importing and using the wrong save function:

```python
# BEFORE (WRONG):
from processing.standalone_tasks import save_detections_to_db
save_detections_to_db(video_obj.id, detections, metadata.get('fps', 25), db, DetectedPerson)

# AFTER (CORRECT):
from processing.tasks import save_detections_to_db
save_detections_to_db(video_obj.id, detections, metadata.get('fps', 25))
```

## Fix Applied
**File Modified**: `c:\pinokio\api\sm-hrm\hr_management\blueprints\videos.py`

**Change**: 
- Line 2785: Changed import from `standalone_tasks` to `tasks`
- Line 2788: Simplified function call to match correct signature
- Removed incompatible parameters (`db`, `DetectedPerson`)

## Verification Results
After applying the fix, the database now correctly contains **201 unique detected persons** with proper schema:

```
=== Person Detection Summary ===
Found 201 unique persons in the video
Person 1: Detected 36 times (0.00-0.73 seconds, duration: 0.73s)
Person 2: Detected 75 times (0.53-4.73 seconds, duration: 4.20s)
...
Person 201: Detected 3 times (799.67-799.93 seconds, duration: 0.27s)
```

## Technical Details

### DetectedPerson Model Schema (Correct)
```python
class DetectedPerson(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.Integer, db.ForeignKey('videos.id'), nullable=False)
    person_code = db.Column(db.String(20), nullable=False)  # PERSON-XXXX format
    start_frame = db.Column(db.Integer, nullable=False)
    end_frame = db.Column(db.Integer, nullable=False)
    start_time = db.Column(db.Float)  # Time in seconds
    end_time = db.Column(db.Float)    # Time in seconds
    confidence = db.Column(db.Float)
    bbox_data = db.Column(db.JSON)  # Store bounding box coordinates for each frame
    # ... other fields
```

### Detection Data Format
```python
{
    'person_code': 'PERSON-0001',
    'start_frame': 100,
    'end_frame': 150,
    'start_time': 4.0,
    'end_time': 6.0,
    'confidence': 0.85,
    'bbox_data': [
        {'x': 10, 'y': 20, 'width': 100, 'height': 200, 'frame': 100},
        {'x': 12, 'y': 22, 'width': 102, 'height': 202, 'frame': 101},
        # ... more frames
    ]
}
```

## Impact
- ✅ Person detections now properly saved to database
- ✅ Video interface displays match database content
- ✅ Detection data can be retrieved for further processing
- ✅ Face recognition and attendance tracking can now work properly
- ✅ GPU acceleration continues to work as previously implemented

## Previous Work
This fix builds on the previously completed GPU acceleration implementation:
- YOLO detection models now use GPU acceleration
- YOLOv8 processing optimized for CUDA
- Performance improvements verified and documented

## Status
🎯 **ISSUE RESOLVED**: The schema mismatch has been fixed and person detections are now being properly saved to the database with the correct schema format.

## Testing
- Database query confirmed 201 persons properly stored
- Flask application starts without errors
- Web interface accessible at http://localhost:5001
- No compilation errors in modified files
