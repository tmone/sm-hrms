@echo off
echo ===============================================
echo GPU Support Installation for Windows
echo For NVIDIA GeForce RTX 4070 Ti SUPER
echo ===============================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

echo Current Python version:
python --version
echo.

echo Step 1: Uninstalling CPU-only PyTorch (if present)...
python -m pip uninstall -y torch torchvision torchaudio

echo.
echo Step 2: Installing PyTorch with CUDA 12.1 support...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo Step 3: Installing GPU-accelerated packages...
python -m pip install onnxruntime-gpu
python -m pip install opencv-contrib-python
python -m pip install ultralytics --upgrade

echo.
echo Step 4: Verifying GPU support...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo.
echo ===============================================
echo Installation complete!
echo Please restart your application for changes to take effect.
echo ===============================================
pause