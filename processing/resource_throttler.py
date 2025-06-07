"""
Resource Throttler - Prevents system overload by monitoring and limiting CPU/GPU usage
"""
import time
import psutil
import threading
from typing import Optional, Callable
import os

try:
    import torch
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False

try:
    from config_logging import get_logger
    logger = get_logger(__name__)
except:
    import logging
    logger = logging.getLogger(__name__)


class ResourceThrottler:
    """Monitors and throttles resource usage to prevent system crashes"""
    
    def __init__(self, 
                 max_cpu_percent: float = 80.0,
                 max_gpu_percent: float = 85.0,
                 max_gpu_memory_percent: float = 90.0,
                 max_temperature: float = 83.0,
                 check_interval: float = 0.5):
        """
        Initialize resource throttler
        
        Args:
            max_cpu_percent: Maximum CPU usage before throttling (default 80%)
            max_gpu_percent: Maximum GPU usage before throttling (default 85%)
            max_gpu_memory_percent: Maximum GPU memory usage (default 90%)
            max_temperature: Maximum GPU temperature in Celsius (default 83°C)
            check_interval: How often to check resources in seconds
        """
        self.max_cpu_percent = max_cpu_percent
        self.max_gpu_percent = max_gpu_percent
        self.max_gpu_memory_percent = max_gpu_memory_percent
        self.max_temperature = max_temperature
        self.check_interval = check_interval
        
        # Throttling state
        self.is_throttling = False
        self.throttle_level = 0  # 0=none, 1=light, 2=medium, 3=heavy
        
        # Initialize GPU handle if available
        self.gpu_handle = None
        if GPU_AVAILABLE:
            try:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                logger.info("GPU monitoring initialized")
            except:
                logger.warning("Failed to initialize GPU monitoring")
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        return psutil.cpu_percent(interval=0.1)
    
    def get_gpu_stats(self) -> dict:
        """Get current GPU statistics"""
        stats = {
            'usage': 0,
            'memory_used': 0,
            'memory_total': 0,
            'memory_percent': 0,
            'temperature': 0
        }
        
        if not GPU_AVAILABLE or not self.gpu_handle:
            return stats
        
        try:
            # GPU utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            stats['usage'] = util.gpu
            
            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            stats['memory_used'] = mem_info.used / 1024**2  # MB
            stats['memory_total'] = mem_info.total / 1024**2  # MB
            stats['memory_percent'] = (mem_info.used / mem_info.total) * 100
            
            # Temperature
            stats['temperature'] = pynvml.nvmlDeviceGetTemperature(
                self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU
            )
        except Exception as e:
            logger.debug(f"Error getting GPU stats: {e}")
        
        return stats
    
    def calculate_throttle_level(self) -> int:
        """Calculate required throttle level based on resource usage"""
        cpu_usage = self.get_cpu_usage()
        gpu_stats = self.get_gpu_stats()
        
        # Log current usage
        logger.debug(f"CPU: {cpu_usage:.1f}%, GPU: {gpu_stats['usage']}%, "
                    f"GPU Mem: {gpu_stats['memory_percent']:.1f}%, "
                    f"Temp: {gpu_stats['temperature']}°C")
        
        # Determine throttle level
        throttle_level = 0
        
        # Check CPU
        if cpu_usage > self.max_cpu_percent + 10:
            throttle_level = max(throttle_level, 3)  # Heavy throttle
        elif cpu_usage > self.max_cpu_percent:
            throttle_level = max(throttle_level, 2)  # Medium throttle
        elif cpu_usage > self.max_cpu_percent - 10:
            throttle_level = max(throttle_level, 1)  # Light throttle
        
        # Check GPU
        if GPU_AVAILABLE:
            if gpu_stats['usage'] > self.max_gpu_percent + 10:
                throttle_level = max(throttle_level, 3)
            elif gpu_stats['usage'] > self.max_gpu_percent:
                throttle_level = max(throttle_level, 2)
            elif gpu_stats['usage'] > self.max_gpu_percent - 10:
                throttle_level = max(throttle_level, 1)
            
            # Check GPU memory
            if gpu_stats['memory_percent'] > self.max_gpu_memory_percent:
                throttle_level = max(throttle_level, 2)
            
            # Check temperature (critical)
            if gpu_stats['temperature'] > self.max_temperature:
                throttle_level = 3  # Always heavy throttle for high temp
            elif gpu_stats['temperature'] > self.max_temperature - 5:
                throttle_level = max(throttle_level, 2)
        
        return throttle_level
    
    def get_delay_time(self) -> float:
        """Get delay time based on throttle level"""
        delays = {
            0: 0.0,    # No delay
            1: 0.05,   # 50ms delay
            2: 0.1,    # 100ms delay
            3: 0.5     # 500ms delay
        }
        return delays.get(self.throttle_level, 0.0)
    
    def throttle(self):
        """Apply throttling if needed"""
        self.throttle_level = self.calculate_throttle_level()
        
        if self.throttle_level > 0:
            if not self.is_throttling:
                logger.warning(f"🚦 Resource throttling activated (level {self.throttle_level})")
                self.is_throttling = True
            
            delay = self.get_delay_time()
            time.sleep(delay)
            
            # For heavy throttling, also yield CPU
            if self.throttle_level >= 3:
                time.sleep(0.1)  # Extra delay
                # Force garbage collection for Python
                import gc
                gc.collect()
                
                # Clear GPU cache if available
                if GPU_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            if self.is_throttling:
                logger.info("✅ Resource throttling deactivated")
                self.is_throttling = False
    
    def wait_for_resources(self, timeout: float = 30.0) -> bool:
        """
        Wait for resources to become available
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if resources available, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            self.throttle_level = self.calculate_throttle_level()
            
            if self.throttle_level <= 1:  # Light throttle or none
                return True
            
            logger.info("⏳ Waiting for resources to become available...")
            time.sleep(1.0)
        
        return False
    
    def with_throttling(self, func: Callable, *args, **kwargs):
        """Execute a function with automatic throttling"""
        self.throttle()
        result = func(*args, **kwargs)
        self.throttle()
        return result


class BatchProcessor:
    """Process data in batches with resource management"""
    
    def __init__(self, throttler: ResourceThrottler, batch_size: int = 8):
        self.throttler = throttler
        self.batch_size = batch_size
    
    def process_frames(self, frames: list, process_func: Callable) -> list:
        """Process frames in batches with throttling"""
        results = []
        total_frames = len(frames)
        
        for i in range(0, total_frames, self.batch_size):
            # Check resources before each batch
            self.throttler.throttle()
            
            batch = frames[i:i + self.batch_size]
            batch_results = process_func(batch)
            results.extend(batch_results)
            
            # Log progress
            processed = min(i + self.batch_size, total_frames)
            logger.info(f"Processed {processed}/{total_frames} frames")
            
            # Additional delay between batches
            if self.throttler.throttle_level >= 2:
                time.sleep(0.2)  # Extra 200ms between batches
        
        return results


# Global throttler instance
_throttler_instance = None

def get_throttler() -> ResourceThrottler:
    """Get or create global throttler instance"""
    global _throttler_instance
    if _throttler_instance is None:
        _throttler_instance = ResourceThrottler()
    return _throttler_instance


def safe_gpu_operation(func: Callable, *args, **kwargs):
    """
    Execute a GPU operation with error recovery
    
    Handles CUDA out of memory errors and other GPU issues
    """
    max_retries = 3
    retry_delay = 2.0
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"GPU out of memory error (attempt {attempt + 1}/{max_retries})")
                
                # Clear cache
                if GPU_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Wait before retry
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                
                # Reduce batch size if provided in kwargs
                if 'batch_size' in kwargs and attempt < max_retries - 1:
                    kwargs['batch_size'] = max(1, kwargs['batch_size'] // 2)
                    logger.info(f"Reducing batch size to {kwargs['batch_size']}")
            else:
                raise
    
    raise RuntimeError(f"GPU operation failed after {max_retries} attempts")