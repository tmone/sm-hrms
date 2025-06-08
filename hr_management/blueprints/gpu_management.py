from flask import Blueprint, render_template, jsonify, current_app
from flask_login import login_required
import subprocess
import sys
import os
import platform
import json
from datetime import datetime

# Optional imports - these may not be available initially
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    cv2 = None
    OPENCV_AVAILABLE = False

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
        'opencv_version': None,
        'torch_available': TORCH_AVAILABLE,
        'opencv_available': OPENCV_AVAILABLE
    }
    
    # Check OpenCV version if available
    if OPENCV_AVAILABLE and cv2:
        status['opencv_version'] = cv2.__version__
    
    # Check PyTorch and CUDA
    if TORCH_AVAILABLE and torch:
        try:
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
    else:
        status['torch_version'] = 'Not installed'
        status['torch_error'] = 'PyTorch not available'
    
    # Check OpenCV CUDA support
    if OPENCV_AVAILABLE and cv2:
        try:
            build_info = cv2.getBuildInformation()
            status['opencv_cuda'] = 'CUDA:                      YES' in build_info
        except:
            pass
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
    status_data = check_gpu_status()
    
    # Add NVIDIA driver info
    try:
        if platform.system() == 'Windows':
            result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], 
                                  capture_output=True, text=True, shell=True)
        else:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            
        if result.returncode == 0:
            status_data['nvidia_driver'] = result.stdout.strip()
        else:
            status_data['nvidia_driver'] = 'Not found'
    except:
        status_data['nvidia_driver'] = 'Not found'
    
    return jsonify(status_data)

@gpu_management_bp.route('/diagnose')
@login_required
def diagnose_gpu():
    """Run comprehensive GPU diagnostics"""
    diagnostics = {
        'timestamp': datetime.now().isoformat(),
        'platform': platform.system(),
        'python_version': sys.version,
        'issues': [],
        'recommendations': [],
        'checks': {}
    }
    
    # 1. Check if NVIDIA GPU exists
    try:
        nvidia_result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, shell=True)
        if nvidia_result.returncode == 0:
            diagnostics['checks']['nvidia_gpus'] = nvidia_result.stdout.strip().split('\n')
            diagnostics['checks']['has_nvidia_gpu'] = True
        else:
            diagnostics['checks']['nvidia_gpus'] = []
            diagnostics['checks']['has_nvidia_gpu'] = False
            diagnostics['issues'].append("No NVIDIA GPU detected by nvidia-smi")
            diagnostics['recommendations'].append("Ensure NVIDIA drivers are installed")
    except Exception as e:
        diagnostics['checks']['nvidia_gpus'] = []
        diagnostics['checks']['has_nvidia_gpu'] = False
        diagnostics['issues'].append(f"nvidia-smi not found: {str(e)}")
        diagnostics['recommendations'].append("Install NVIDIA drivers from https://www.nvidia.com/drivers")
    
    # 2. Check PyTorch installation
    try:
        import torch
        diagnostics['checks']['torch_version'] = torch.__version__
        diagnostics['checks']['torch_cuda_available'] = torch.cuda.is_available()
        diagnostics['checks']['torch_cuda_version'] = torch.version.cuda if hasattr(torch.version, 'cuda') else None
        
        # Check if PyTorch was built with CUDA
        if 'cu' in torch.__version__ or '+cu' in torch.__version__:
            diagnostics['checks']['torch_cuda_build'] = True
            if '+' in torch.__version__:
                cuda_version = torch.__version__.split('+')[1]
                diagnostics['checks']['torch_cuda_build_version'] = cuda_version
        else:
            diagnostics['checks']['torch_cuda_build'] = False
            diagnostics['issues'].append("PyTorch installed without CUDA support")
            diagnostics['recommendations'].append("Run 'Install GPU Support' to reinstall PyTorch with CUDA")
            
        # Check CUDA runtime
        if torch.cuda.is_available():
            diagnostics['checks']['cuda_device_count'] = torch.cuda.device_count()
            diagnostics['checks']['cuda_current_device'] = torch.cuda.current_device()
            diagnostics['checks']['cuda_device_name'] = torch.cuda.get_device_name(0)
        else:
            # Detailed CUDA unavailability check
            diagnostics['checks']['cuda_is_available'] = False
            diagnostics['issues'].append("CUDA not available to PyTorch")
            
            # Check environment variables
            cuda_path = os.environ.get('CUDA_PATH')
            cuda_home = os.environ.get('CUDA_HOME')
            diagnostics['checks']['cuda_path_env'] = cuda_path
            diagnostics['checks']['cuda_home_env'] = cuda_home
            
            if not cuda_path and not cuda_home:
                diagnostics['issues'].append("CUDA_PATH/CUDA_HOME environment variables not set")
                diagnostics['recommendations'].append("Install CUDA Toolkit or set CUDA environment variables")
                
    except ImportError:
        diagnostics['checks']['torch_version'] = 'Not installed'
        diagnostics['issues'].append("PyTorch not installed")
        diagnostics['recommendations'].append("Run 'Install GPU Support' to install PyTorch")
    except Exception as e:
        diagnostics['checks']['torch_error'] = str(e)
        diagnostics['issues'].append(f"Error checking PyTorch: {str(e)}")
    
    # 3. Check CUDA Toolkit installation
    if platform.system() == 'Windows':
        # Windows CUDA paths
        cuda_paths = [
            r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA',
            r'C:\Program Files\NVIDIA Corporation\CUDA'
        ]
        found_cuda = False
        for base_path in cuda_paths:
            if os.path.exists(base_path):
                cuda_versions = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and 'v' in d]
                if cuda_versions:
                    diagnostics['checks']['cuda_toolkit_installations'] = cuda_versions
                    diagnostics['checks']['cuda_toolkit_path'] = base_path
                    found_cuda = True
                    break
        
        if not found_cuda:
            diagnostics['issues'].append("CUDA Toolkit not found in standard Windows locations")
            diagnostics['recommendations'].append("Download and install CUDA Toolkit from NVIDIA website")
    
    # 4. Final diagnosis
    if diagnostics['checks'].get('has_nvidia_gpu') and not diagnostics['checks'].get('torch_cuda_available'):
        if not diagnostics['checks'].get('torch_cuda_build'):
            diagnostics['primary_issue'] = "PyTorch installed without CUDA support"
            diagnostics['solution'] = "Click 'Install GPU Support' button to reinstall PyTorch with CUDA"
        else:
            diagnostics['primary_issue'] = "CUDA runtime not accessible despite CUDA-enabled PyTorch"
            diagnostics['solution'] = "Install CUDA Toolkit and ensure environment variables are set"
    elif not diagnostics['checks'].get('has_nvidia_gpu'):
        diagnostics['primary_issue'] = "No NVIDIA GPU detected"
        diagnostics['solution'] = "CUDA requires an NVIDIA GPU. Use CPU processing instead."
    
    return jsonify(diagnostics)

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
          # Schedule restart if installation was successful
        restart_scheduled = False
        if all(r['success'] for r in results):
            # Import here to avoid circular imports
            import threading
            import time
            
            def restart_app():
                time.sleep(5)  # Give more time for response to be sent and received
                try:
                    print("GPU installation completed. Restarting application to load new libraries...")
                    if platform.system() == 'Windows':
                        # Windows restart - use a more reliable method
                        import os
                        # Exit current process, let process manager restart
                        os._exit(0)
                    else:
                        # Linux/Mac restart
                        os.execv(sys.executable, [sys.executable] + sys.argv)
                except Exception as e:
                    print(f"Error restarting application: {e}")
            
            # Start restart in background thread
            restart_thread = threading.Thread(target=restart_app)
            restart_thread.daemon = True
            restart_thread.start()
            restart_scheduled = True
        
        return jsonify({
            'success': all(r['success'] for r in results),
            'results': results,
            'new_status': new_status,
            'restart_scheduled': restart_scheduled,
            'message': 'GPU support installation completed. Application will restart automatically...' if restart_scheduled else 'GPU support installation completed. Please restart the application for changes to take effect.'
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

@gpu_management_bp.route('/restart-status')
@login_required
def restart_status():
    """Check if application has restarted after GPU installation"""
    # This endpoint existing and responding means the app has restarted successfully
    status = check_gpu_status()
    return jsonify({
        'restarted': True,
        'timestamp': datetime.now().isoformat(),
        'gpu_status': status
    })