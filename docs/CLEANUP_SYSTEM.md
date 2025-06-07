# Cleanup System Documentation

## Overview
The cleanup system automatically removes temporary files and directories created during video processing to prevent disk space issues.

## Components

### 1. Cleanup Manager (`/processing/cleanup_manager.py`)
Core cleanup functionality with methods for:
- Cleaning chunk directories
- Removing non-PERSON directories
- Deleting old temporary files
- Removing empty directories

### 2. Scheduled Cleanup (`/processing/scheduled_cleanup.py`)
Runs cleanup tasks automatically every 6 hours:
- Removes chunk directories older than 24 hours
- Cleans non-PERSON directories
- Deletes temporary files
- Removes empty directories

### 3. Integrated Cleanup
Cleanup is integrated into:
- **Video processing completion**: Chunks cleaned after successful processing
- **Chunk merging**: Chunks cleaned after merging results
- **App startup**: Immediate cleanup of leftover files

## What Gets Cleaned

### Chunk Directories
- Location: `static/uploads/chunks/*`
- Pattern: Directories containing video chunks
- Cleaned: After processing or if older than 24 hours

### Non-PERSON Directories
- Location: `processing/outputs/persons/*`
- Pattern: Any directory NOT matching `PERSON-XXXX`
- Examples of removed directories:
  - `UNKNOWN-0001`
  - `temp_processing`
  - `chunk_000`

### Temporary Files
- Patterns: `*.tmp`, `*.temp`, `*.log`, `concat_list.txt`
- Locations:
  - `processing/temp/`
  - `processing/outputs/temp/`
  - `static/uploads/temp/`
  - `static/uploads/chunks/`

### Empty Directories
- Any empty directory in:
  - `processing/outputs/`
  - `static/uploads/chunks/`

## Manual Cleanup

### Run Full Cleanup
```bash
python3 scripts/run_cleanup.py
```

### Cleanup Options
```bash
# Dry run (show what would be cleaned)
python3 scripts/run_cleanup.py --dry-run

# Clean only chunk directories
python3 scripts/run_cleanup.py --chunks-only

# Clean only non-PERSON directories
python3 scripts/run_cleanup.py --non-person-only

# Clean chunks older than 48 hours
python3 scripts/run_cleanup.py --hours-old 48
```

## Testing Cleanup

Run the test script to verify cleanup functionality:
```bash
# Full test (create test dirs and clean)
python3 test_cleanup.py

# Create test directories only
python3 test_cleanup.py --create-only

# Run cleanup only
python3 test_cleanup.py --cleanup-only
```

## Configuration

### Scheduled Cleanup Interval
Edit in `app.py`:
```python
cleanup_service = start_scheduled_cleanup(cleanup_interval_hours=6)  # Default: 6 hours
```

### Chunk Age Threshold
Edit in cleanup scripts:
```python
cleanup_manager.cleanup_old_chunks(hours_old=24)  # Default: 24 hours
```

## Important Notes

1. **PERSON Directories Protected**: Only directories matching `PERSON-XXXX` pattern are preserved
2. **Automatic Cleanup**: Runs every 6 hours automatically
3. **Safe Cleanup**: Errors are logged but don't stop the cleanup process
4. **Immediate Cleanup**: Runs on app startup to clean leftover files

## Monitoring

Check cleanup logs:
```bash
# View recent cleanup activity
grep -i "cleanup" app.log | tail -20

# Check cleanup statistics
grep -i "cleaned up" app.log | tail -10
```

## Troubleshooting

### Disk Space Issues
If disk is filling up despite cleanup:
1. Run manual cleanup: `python3 scripts/run_cleanup.py`
2. Check for stuck processing: Look for old chunks still marked as "processing"
3. Reduce chunk age threshold: `--hours-old 12`

### Protected Directories
If valid directories are being cleaned:
1. Ensure they match `PERSON-XXXX` pattern exactly
2. Check regex pattern in `cleanup_manager.py`
3. Add custom exclusions if needed