"""
Safe GPU wrapper with automatic fallback to CPU processing
"""
import os
import sys
import subprocess
import json
import tempfile
from pathlib import Path

try:
    from config_logging import get_logger
    logger = get_logger(__name__)
except:
    import logging
    logger = logging.getLogger(__name__)


class SafeGPUProcessor:
    """Wrap GPU operations with safety measures and CPU fallback"""
    
    def __init__(self):
        self.gpu_failed = False
        self.failure_count = 0
        self.max_failures = 3
    
    def process_with_isolation(self, video_path, video_id=None, gpu_config=None):
        """
        Process video in isolated subprocess for safety
        
        This prevents GPU crashes from taking down the main process
        """
        if self.gpu_failed and self.failure_count >= self.max_failures:
            print("[WARNING] GPU processing disabled due to repeated failures, using CPU")
            return self._process_with_cpu(video_path, video_id)
        
        # Create temporary file for results
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tf:
            result_file = tf.name
        
        try:
            # Build command to run in subprocess
            script = """
import sys
sys.path.insert(0, r'{}')
import json
from processing.gpu_enhanced_detection import gpu_person_detection_task

# Run detection
result = gpu_person_detection_task(r'{}', gpu_config={}, video_id={})

# Save result
with open(r'{}', 'w') as f:
    json.dump(result, f)
""".format(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                video_path,
                repr(gpu_config or {}),  # Fix boolean serialization
                video_id or 'None',
                result_file
            )
            
            # Run in subprocess with timeout
            process = subprocess.Popen(
                [sys.executable, '-c', script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for completion with timeout (30 minutes for large videos)
            try:
                stdout, stderr = process.communicate(timeout=1800)
                
                if process.returncode != 0:
                    raise Exception(f"Subprocess failed: {stderr}")
                
                # Read result
                with open(result_file, 'r') as f:
                    result = json.load(f)
                
                # Reset failure count on success
                self.failure_count = 0
                
                return result
                
            except subprocess.TimeoutExpired:
                process.kill()
                raise Exception("GPU processing timed out after 30 minutes")
                
        except Exception as e:
            self.failure_count += 1
            logger.error(f"GPU processing failed (attempt {self.failure_count}): {str(e)}")
            
            if self.failure_count >= self.max_failures:
                self.gpu_failed = True
                print("[ERROR] GPU processing failed multiple times, switching to CPU")
                return self._process_with_cpu(video_path, video_id)
            else:
                # Retry with more conservative settings
                if not gpu_config:
                    gpu_config = {}
                gpu_config['batch_size'] = 1
                gpu_config['fp16'] = False
                
                print(f"[INFO] Retrying with conservative settings (attempt {self.failure_count + 1})")
                return self.process_with_isolation(video_path, video_id, gpu_config)
        
        finally:
            # Clean up temp file
            try:
                os.unlink(result_file)
            except:
                pass
    
    def _process_with_cpu(self, video_path, video_id=None):
        """Fallback CPU processing when GPU fails"""
        print("[INFO] Using CPU-only processing (slower but more stable)")
        
        # Import CPU processing version
        try:
            from processing.cpu_person_detection import cpu_person_detection_task
            return cpu_person_detection_task(video_path, video_id=video_id)
        except ImportError:
            # If no CPU version, use GPU version with CPU config
            from processing.gpu_enhanced_detection import gpu_person_detection_task
            
            cpu_config = {
                'use_gpu': False,
                'batch_size': 1,
                'device': 'cpu',
                'fp16': False,
                'num_workers': 1
            }
            
            return gpu_person_detection_task(video_path, gpu_config=cpu_config, video_id=video_id)


# Global instance
_safe_processor = None

def get_safe_processor():
    """Get or create global safe processor instance"""
    global _safe_processor
    if _safe_processor is None:
        _safe_processor = SafeGPUProcessor()
    return _safe_processor


def process_video_safely(video_path, video_id=None, gpu_config=None):
    """
    Process video with maximum safety - isolated subprocess and CPU fallback
    
    This is the recommended way to process videos to prevent system crashes
    """
    processor = get_safe_processor()
    return processor.process_with_isolation(video_path, video_id, gpu_config)