# Debug Processing UI Updates

## What We've Added:

### Backend Enhancements (`hr_management/blueprints/videos.py`):

1. **Enhanced Processing Status API** (line 922-924):
   - Added detailed logging of progress data
   - Shows database progress values
   - Logs full response details
   - Includes elapsed time information

### Frontend Enhancements (`templates/videos/detail.html`):

1. **Enhanced Data Logging** (line 975):
   - Logs full JSON response from API
   - Shows detailed data structure

2. **Enhanced UI Update Logging** (lines 1047, 1054, 1062):
   - Logs when progress bar is updated
   - Logs when progress percentage is updated  
   - Logs when progress message is updated
   - Warns if any UI elements are missing

## How to Debug:

1. **Open Browser Developer Tools** → Console tab

2. **Start Person Extraction** on a video

3. **Watch the Console Logs** for:
   ```
   📊 Person extraction status data received: {...}
   🔧 Data details: { ... full JSON ... }
   📋 Processing UI Elements found: { progressBar: true, progressPercent: true, progressMessage: true }
   📊 Updated progress bar to X%
   🔢 Updated progress text to X%
   💬 Updated progress message to: "message"
   ```

4. **Check Server Logs** for:
   ```
   🔄 Fallback processing progress for video X: Y% - message
   📊 Database progress: Y
   📡 Processing API Response for video X: status=processing, progress=Y%, message='...'
   🔧 Response details: { ... full response ... }
   ```

## Expected Flow:

1. User clicks "Process Video"
2. Page starts polling `/videos/api/{id}/processing-status` every 2 seconds
3. Backend returns progress updates from database
4. Frontend updates progress bar, percentage, and message
5. Console shows detailed logging of all updates

## Common Issues to Check:

1. **UI Elements Missing**: Look for warnings like "⚠️ Progress bar element not found!"
2. **API Errors**: Look for "❌ Error checking person extraction status"
3. **Progress Not Updating**: Check if backend logs show progress changes
4. **Stuck Processing**: Look for timeout warnings

## Files Modified:

- `hr_management/blueprints/videos.py`: Enhanced API logging
- `templates/videos/detail.html`: Enhanced frontend logging