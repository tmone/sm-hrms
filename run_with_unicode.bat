@echo off
REM Set UTF-8 encoding for Windows console
chcp 65001 > nul

REM Set Python to use UTF-8
set PYTHONIOENCODING=utf-8

REM Run the Flask app
echo Starting HRM application with Unicode support...
python app.py

pause