# Visual Duplicate Image Removal

## Overview
Since images are extracted from video frames using OpenCV, traditional MD5 hash comparison won't work for duplicate detection. Even consecutive frames may have slight pixel differences due to video compression and movement. This feature uses visual similarity detection to find and remove duplicate images.

## How It Works

### 1. Visual Similarity Detection
The system uses multiple methods to detect visually similar images:

- **Perceptual Hashing**: Creates a hash based on image structure, tolerant to minor changes
- **Structural Similarity**: Compares luminance, contrast, and structure between images
- **Color Histogram Matching**: Compares color distribution across RGB channels
- **Combined Score**: Weights all methods to determine overall similarity (90% threshold)

### 2. Quality Preservation
When duplicates are found:
- Keeps the largest file (usually highest quality)
- Removes smaller/lower quality duplicates
- Creates backups before deletion

### 3. Metadata Updates
- Updates metadata.json files
- Adjusts image counts
- Records removal timestamp

## Usage

### Web Interface
1. Go to Persons page
2. Click "Select Mode"
3. Select person(s) to check
4. Click "Remove Duplicates (X)"
5. Confirm the action
6. View results summary

### Command Line (with TensorFlow)
For more accurate detection using deep learning:
```bash
python3 scripts/remove_visual_duplicates.py
```

### Simple Command Line
For basic detection without TensorFlow:
```bash
python3 scripts/remove_duplicate_images.py
```

## Technical Details

### Similarity Calculation
1. **Perceptual Hash (40% weight)**
   - Resizes image to 16x16
   - Calculates gradient-based hash
   - Allows up to 20% difference

2. **Structural Similarity (30% weight)**
   - Based on SSIM algorithm
   - Compares luminance, contrast, structure
   - Normalized to 0-1 range

3. **Color Similarity (30% weight)**
   - 64-bin histograms per channel
   - Correlation coefficient comparison
   - Averaged across RGB channels

### Threshold
- Default: 90% similarity
- Configurable in code
- Higher = stricter matching

### Performance
- Lightweight version uses only OpenCV and NumPy
- Processes ~10-20 images/second
- Memory efficient for large folders

## API Endpoint
```
POST /persons/remove-duplicates
{
  "person_ids": ["PERSON-0001", "PERSON-0002"],
  "create_backup": true
}
```

Response:
```json
{
  "success": true,
  "results": [
    {
      "person_id": "PERSON-0001",
      "duplicates_found": 5,
      "duplicates_removed": 5
    }
  ],
  "total_removed": 5,
  "backup_created": true
}
```

## Backup Location
Removed files are backed up to:
```
processing/outputs/duplicates_backup/PERSON-XXXX/
```

## Notes
- Visual similarity is more CPU intensive than hash comparison
- Works well for consecutive frames or very similar poses
- May not detect duplicates with significant pose/lighting changes
- Recommended to run periodically to clean up storage