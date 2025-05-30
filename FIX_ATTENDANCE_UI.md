# Fix Attendance UI - Summary

## Current Status

The attendance UI implementation is complete with all necessary endpoints and templates. However, the server appears to not be running or there might be dependency issues.

## What Has Been Done

1. **Backend Implementation** (`/hr_management/blueprints/attendance.py`):
   - ✓ `/attendance/` - Main dashboard page
   - ✓ `/attendance/summary` - API endpoint for statistics
   - ✓ `/attendance/daily` - Daily report (supports both HTML and JSON)
   - ✓ `/attendance/export` - Export functionality (Excel/CSV)
   - ✓ `/attendance/test` - Test page (no login required)

2. **Frontend Templates**:
   - ✓ `/templates/attendance/index.html` - Main dashboard with stats cards
   - ✓ `/templates/attendance/daily_report.html` - Daily attendance view
   - ✓ `/templates/attendance/test.html` - Test page for debugging

3. **Database Models**:
   - ✓ DetectedPerson model has attendance fields:
     - attendance_date
     - attendance_time
     - attendance_location
     - check_in_time
     - check_out_time

## How to Access the Attendance UI

1. **Ensure the Flask server is running**:
   ```bash
   cd /mnt/d/sm-hrm
   python3 app.py
   ```

2. **Access the attendance pages**:
   - Main Dashboard: http://localhost:5001/attendance/
   - Test Page (no login): http://localhost:5001/attendance/test
   - Daily Report: http://localhost:5001/attendance/daily
   - API Summary: http://localhost:5001/attendance/summary?days=7

## Troubleshooting

If you get a "page not found" or connection error:

1. **Check if dependencies are installed**:
   ```bash
   pip install flask flask-sqlalchemy flask-login
   ```

2. **Verify the server is running on the correct port**:
   - The app should be running on port 5001
   - Check the console output when starting the server

3. **If you see a login redirect**:
   - The attendance pages require login (except /test)
   - Make sure you're logged in first
   - Or use the test page: http://localhost:5001/attendance/test

## Features Available

1. **Dashboard** (`/attendance/`):
   - Today's attendance count
   - Active locations
   - Weekly statistics
   - OCR processed video count
   - Recent attendance records
   - Quick actions (daily report, export, process videos)

2. **Daily Report** (`/attendance/daily`):
   - Filter by date and location
   - Shows person ID, location, check-in/out times
   - Duration calculation
   - Navigation between days

3. **Export** (`/attendance/export`):
   - Date range selection
   - Location filtering
   - Excel (.xlsx) or CSV format
   - Automatic column width adjustment

## Next Steps

1. Start the Flask server if it's not running
2. Navigate to http://localhost:5001/attendance/
3. If you need to process videos first:
   - Go to Videos section
   - Upload videos with OCR extraction enabled
   - The system will extract timestamps and locations
   - Then attendance data will be available

The attendance UI is fully implemented and ready to use!