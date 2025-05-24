from flask import Blueprint, render_template, jsonify, current_app
from flask_login import login_required
import subprocess
import sys
import os
import torch
import cv2
import platform
import json
from datetime import datetime

gpu_management_bp = Blueprint('gpu_management', __name__, url_prefix='/gpu')

def check_gpu_status():
    """Check current GPU configuration and status"""
    status = {
        'platform': platform.system(),
        'python_version': sys.version,
        'cuda_available': False,
        'cuda_version': None,
        'cuda_toolkit_version': None,
        'gpu_devices': [],
        'torch_version': None,
        'opencv_cuda': False,
        'opencv_version': cv2.__version__
    }
    
    # Check PyTorch and CUDA
    try:
        import torch
        status['torch_version'] = torch.__version__
        status['cuda_available'] = torch.cuda.is_available()
        
        if status['cuda_available']:
            status['cuda_version'] = torch.version.cuda
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                status['gpu_devices'].append({
                    'index': i,
                    'name': device_props.name,
                    'memory_total': f"{device_props.total_memory / 1024**3:.2f} GB",
                    'compute_capability': f"{device_props.major}.{device_props.minor}"
                })
    except Exception as e:
        status['torch_error'] = str(e)
    
    # Check OpenCV CUDA support
    try:
        build_info = cv2.getBuildInformation()
        status['opencv_cuda'] = 'CUDA:                      YES' in build_info
    except:
        pass
    
    # Check NVIDIA driver and CUDA toolkit
    if platform.system() != 'Darwin':  # Not macOS
        try:
            # Check nvidia-smi
            nvidia_cmd = 'nvidia-smi' if platform.system() != 'Windows' else 'nvidia-smi.exe'
            result = subprocess.run([nvidia_cmd, '--query-gpu=name,driver_version,memory.total', 
                                   '--format=csv,noheader'], 
                                  capture_output=True, text=True, shell=platform.system()=='Windows')
            if result.returncode == 0:
                status['nvidia_driver'] = result.stdout.strip()
        except:
            status['nvidia_driver'] = 'Not found'
            
        # Check CUDA toolkit version
        try:
            nvcc_cmd = 'nvcc' if platform.system() != 'Windows' else 'nvcc.exe'
            result = subprocess.run([nvcc_cmd, '--version'], 
                                  capture_output=True, text=True, shell=platform.system()=='Windows')
            if result.returncode == 0:
                # Parse CUDA version from nvcc output
                for line in result.stdout.split('\n'):
                    if 'release' in line:
                        import re
                        match = re.search(r'release (\d+\.\d+)', line)
                        if match:
                            status['cuda_toolkit_version'] = match.group(1)
                            break
        except:
            status['cuda_toolkit_version'] = None
    
    return status

@gpu_management_bp.route('/')
@login_required
def index():
    """GPU management dashboard"""
    return render_template('gpu_management/index.html')

@gpu_management_bp.route('/status')
@login_required
def status():
    """Get current GPU status"""
    return jsonify(check_gpu_status())

@gpu_management_bp.route('/install', methods=['POST'])
@login_required
def install_gpu_support():
    """Install GPU support libraries"""
    try:
        # Check if GPU is available first
        status = check_gpu_status()
        
        if platform.system() == 'Darwin':
            return jsonify({
                'success': False,
                'message': 'GPU acceleration not supported on macOS. Using CPU acceleration instead.',
                'status': status
            })
        
        # Create installation script
        install_commands = []
        
        # Determine CUDA version for PyTorch based on system
        cuda_version = 'cpu'
        if status.get('nvidia_driver'):
            # For Windows with RTX 4070 Ti SUPER, use CUDA 12.1 for best compatibility
            if platform.system() == 'Windows':
                cuda_version = 'cu121'  # CUDA 12.1 for newer GPUs
            else:
                cuda_version = 'cu118'  # CUDA 11.8 for Linux
        
        # Uninstall CPU-only PyTorch first if needed
        if status.get('torch_version') and '+cpu' in status.get('torch_version', ''):
            install_commands.append(
                f"{sys.executable} -m pip uninstall -y torch torchvision torchaudio"
            )
            
        # Install PyTorch with CUDA support
        if platform.system() == 'Windows':
            # Windows-specific installation
            install_commands.append(
                f"{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/{cuda_version}"
            )
        else:
            # Linux installation
            install_commands.append(
                f"{sys.executable} -m pip install torch torchvision --index-url https://download.pytorch.org/whl/{cuda_version}"
            )
        
        # Install other GPU-accelerated packages
        install_commands.extend([
            f"{sys.executable} -m pip install onnxruntime-gpu",
            f"{sys.executable} -m pip install opencv-contrib-python",  # Better than headless for GPU
            f"{sys.executable} -m pip install ultralytics --upgrade"
        ])
        
        # Run installation
        results = []
        for cmd in install_commands:
            try:
                # Use shell=True for Windows compatibility
                if platform.system() == 'Windows':
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                else:
                    result = subprocess.run(cmd.split(), capture_output=True, text=True)
                    
                results.append({
                    'command': cmd,
                    'success': result.returncode == 0,
                    'output': result.stdout if result.returncode == 0 else result.stderr
                })
            except Exception as e:
                results.append({
                    'command': cmd,
                    'success': False,
                    'output': str(e)
                })
        
        # Check new status
        new_status = check_gpu_status()
        
        return jsonify({
            'success': all(r['success'] for r in results),
            'results': results,
            'new_status': new_status,
            'message': 'GPU support installation completed. Please restart the application for changes to take effect.'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@gpu_management_bp.route('/test', methods=['POST'])
@login_required
def test_gpu_performance():
    """Run GPU performance tests"""
    try:
        import time
        import numpy as np
        
        tests = []
        
        # Test 1: PyTorch CUDA
        if torch.cuda.is_available():
            # Create test tensors
            size = 10000
            device = torch.device('cuda')
            
            # CPU test
            start = time.time()
            a_cpu = torch.randn(size, size)
            b_cpu = torch.randn(size, size)
            c_cpu = torch.matmul(a_cpu, b_cpu)
            cpu_time = time.time() - start
            
            # GPU test
            start = time.time()
            a_gpu = a_cpu.to(device)
            b_gpu = b_cpu.to(device)
            c_gpu = torch.matmul(a_gpu, b_gpu)
            torch.cuda.synchronize()
            gpu_time = time.time() - start
            
            tests.append({
                'name': 'PyTorch Matrix Multiplication (10000x10000)',
                'cpu_time': f"{cpu_time:.3f}s",
                'gpu_time': f"{gpu_time:.3f}s",
                'speedup': f"{cpu_time/gpu_time:.2f}x"
            })
        
        # Test 2: Video processing simulation
        try:
            # Create test image
            test_image = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)
            
            # CPU processing
            start = time.time()
            for _ in range(30):  # Simulate 30 frames
                gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 100, 200)
            cpu_video_time = time.time() - start
            
            # GPU processing (if available)
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                gpu_image = cv2.cuda_GpuMat()
                gpu_image.upload(test_image)
                
                start = time.time()
                for _ in range(30):
                    gpu_gray = cv2.cuda.cvtColor(gpu_image, cv2.COLOR_BGR2GRAY)
                    gpu_edges = cv2.cuda.Canny(gpu_gray, 100, 200)
                gpu_video_time = time.time() - start
                
                tests.append({
                    'name': 'OpenCV Video Processing (30 frames)',
                    'cpu_time': f"{cpu_video_time:.3f}s",
                    'gpu_time': f"{gpu_video_time:.3f}s",
                    'speedup': f"{cpu_video_time/gpu_video_time:.2f}x"
                })
            else:
                tests.append({
                    'name': 'OpenCV Video Processing',
                    'cpu_time': f"{cpu_video_time:.3f}s",
                    'gpu_time': 'N/A (CUDA not available)',
                    'speedup': 'N/A'
                })
                
        except Exception as e:
            tests.append({
                'name': 'OpenCV Video Processing',
                'error': str(e)
            })
        
        # Test 3: YOLO inference
        try:
            from ultralytics import YOLO
            
            # Load a small model for testing
            model = YOLO('yolov8n.pt')
            
            # Create test image
            test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # CPU inference
            start = time.time()
            results_cpu = model(test_img, device='cpu', verbose=False)
            cpu_yolo_time = time.time() - start
            
            # GPU inference
            if torch.cuda.is_available():
                start = time.time()
                results_gpu = model(test_img, device=0, verbose=False)
                gpu_yolo_time = time.time() - start
                
                tests.append({
                    'name': 'YOLO Object Detection',
                    'cpu_time': f"{cpu_yolo_time:.3f}s",
                    'gpu_time': f"{gpu_yolo_time:.3f}s",
                    'speedup': f"{cpu_yolo_time/gpu_yolo_time:.2f}x"
                })
            else:
                tests.append({
                    'name': 'YOLO Object Detection',
                    'cpu_time': f"{cpu_yolo_time:.3f}s",
                    'gpu_time': 'N/A (CUDA not available)',
                    'speedup': 'N/A'
                })
                
        except Exception as e:
            tests.append({
                'name': 'YOLO Object Detection',
                'error': str(e)
            })
        
        return jsonify({
            'success': True,
            'tests': tests,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@gpu_management_bp.route('/codec-info')
@login_required
def codec_info():
    """Get available video codecs"""
    try:
        codecs = []
        
        # Test common codecs
        test_codecs = [
            ('H264', 'H.264/AVC'),
            ('h264', 'H.264 (lowercase)'),
            ('avc1', 'H.264/AVC (Apple)'),
            ('x264', 'x264 encoder'),
            ('mp4v', 'MPEG-4'),
            ('MJPG', 'Motion JPEG'),
            ('XVID', 'Xvid'),
            ('DIVX', 'DivX')
        ]
        
        for fourcc, name in test_codecs:
            try:
                # Test if codec is available
                test_writer = cv2.VideoWriter(
                    'test.mp4',
                    cv2.VideoWriter_fourcc(*fourcc),
                    30,
                    (640, 480)
                )
                is_available = test_writer.isOpened()
                test_writer.release()
                
                if os.path.exists('test.mp4'):
                    os.remove('test.mp4')
                
                codecs.append({
                    'fourcc': fourcc,
                    'name': name,
                    'available': is_available
                })
            except:
                codecs.append({
                    'fourcc': fourcc,
                    'name': name,
                    'available': False
                })
        
        # Get OpenCV build info
        build_info = cv2.getBuildInformation()
        video_io_section = False
        video_io_info = []
        
        for line in build_info.split('\n'):
            if 'Video I/O:' in line:
                video_io_section = True
            elif video_io_section and line.strip() == '':
                break
            elif video_io_section:
                video_io_info.append(line.strip())
        
        return jsonify({
            'codecs': codecs,
            'opencv_version': cv2.__version__,
            'video_io_info': video_io_info
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500