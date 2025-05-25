@echo off
echo ===============================================
echo Installing OpenH264 codec for Windows
echo ===============================================
echo.

REM Create directory for OpenH264
set OPENCV_DIR=%LOCALAPPDATA%\opencv
if not exist "%OPENCV_DIR%" mkdir "%OPENCV_DIR%"

cd /d "%OPENCV_DIR%"

echo Downloading OpenH264 library...
powershell -Command "Invoke-WebRequest -Uri 'https://github.com/cisco/openh264/releases/download/v1.8.0/openh264-1.8.0-win64.dll.bz2' -OutFile 'openh264-1.8.0-win64.dll.bz2'"

echo Extracting library...
REM Use 7zip if available, otherwise use PowerShell
where 7z >nul 2>nul
if %ERRORLEVEL% == 0 (
    7z x openh264-1.8.0-win64.dll.bz2
) else (
    echo Using PowerShell to extract (this may take a moment)...
    powershell -Command "Add-Type -AssemblyName System.IO.Compression.FileSystem; [System.IO.Compression.ZipFile]::ExtractToDirectory('openh264-1.8.0-win64.dll.bz2', '.')"
)

REM Copy to Python site-packages cv2 directory
echo.
echo Locating OpenCV installation...
python -c "import cv2; import os; print(os.path.dirname(cv2.__file__))" > cv2_path.txt
set /p CV2_PATH=<cv2_path.txt
del cv2_path.txt

if exist "%CV2_PATH%" (
    echo Found OpenCV at: %CV2_PATH%
    copy openh264-1.8.0-win64.dll "%CV2_PATH%\"
    echo.
    echo ✅ OpenH264 library installed successfully!
) else (
    echo ❌ Could not find OpenCV installation directory
    echo Please copy openh264-1.8.0-win64.dll manually to your OpenCV directory
)

echo.
echo ===============================================
echo Alternative: Set environment variable
echo ===============================================
echo You can also set the OPENH264_LIBRARY environment variable:
echo.
echo setx OPENH264_LIBRARY "%OPENCV_DIR%\openh264-1.8.0-win64.dll"
echo.
pause