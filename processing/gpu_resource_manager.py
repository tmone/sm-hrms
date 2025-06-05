import os
import sys
import time
import logging
import threading
import psutil
import queue
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

# GPU monitoring imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except:
    NVML_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class GPUStatus:
    """GPU status information"""
    gpu_id: int
    name: str
    memory_used_mb: float
    memory_total_mb: float
    memory_percent: float
    temperature: float
    utilization: float
    is_available: bool
    timestamp: datetime


class GPUResourceManager:
    """Manages GPU resources for safe parallel processing"""
    
    def __init__(self, max_memory_percent: float = 80.0, 
                 max_temperature: float = 85.0,
                 check_interval: float = 2.0):
        """
        Initialize GPU Resource Manager
        
        Args:
            max_memory_percent: Maximum GPU memory usage percentage (default: 80%)
            max_temperature: Maximum GPU temperature in Celsius (default: 85°C)
            check_interval: Interval between resource checks in seconds
        """
        self.max_memory_percent = max_memory_percent
        self.max_temperature = max_temperature
        self.check_interval = check_interval
        
        # Worker management
        self.gpu_workers = {}  # worker_id -> gpu_id
        self.worker_lock = threading.Lock()
        
        # Monitoring
        self.monitoring_active = False
        self.monitor_thread = None
        self.gpu_stats_history = []
        self.max_history = 1000
        
        # GPU availability
        self.available_gpus = self._detect_gpus()
        self.gpu_queue = queue.Queue()
        
        # Initialize GPU queue
        for gpu_id in self.available_gpus:
            self.gpu_queue.put(gpu_id)
            
        logger.info(f"GPU Resource Manager initialized with {len(self.available_gpus)} GPUs")
        
    def _detect_gpus(self) -> List[int]:
        """Detect available GPUs"""
        gpus = []
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            count = torch.cuda.device_count()
            gpus = list(range(count))
            logger.info(f"Detected {count} GPUs via PyTorch")
        elif GPUTIL_AVAILABLE:
            gpu_list = GPUtil.getGPUs()
            gpus = [gpu.id for gpu in gpu_list]
            logger.info(f"Detected {len(gpus)} GPUs via GPUtil")
        
        return gpus
        
    def start_monitoring(self):
        """Start GPU monitoring thread"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("GPU monitoring started")
            
    def stop_monitoring(self):
        """Stop GPU monitoring thread"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
            logger.info("GPU monitoring stopped")
            
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect GPU stats
                gpu_stats = self._collect_gpu_stats()
                
                # Store in history
                self.gpu_stats_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'stats': gpu_stats
                })
                
                # Limit history size
                if len(self.gpu_stats_history) > self.max_history:
                    self.gpu_stats_history = self.gpu_stats_history[-self.max_history:]
                
                # Check for issues
                self._check_gpu_health(gpu_stats)
                
            except Exception as e:
                logger.error(f"Error in GPU monitoring: {e}")
                
            time.sleep(self.check_interval)
            
    def _collect_gpu_stats(self) -> List[GPUStatus]:
        """Collect current GPU statistics"""
        stats = []
        
        for gpu_id in self.available_gpus:
            try:
                stat = self._get_gpu_status(gpu_id)
                stats.append(stat)
            except Exception as e:
                logger.error(f"Failed to get stats for GPU {gpu_id}: {e}")
                
        return stats
        
    def _get_gpu_status(self, gpu_id: int) -> GPUStatus:
        """Get status for a specific GPU"""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            # Use PyTorch
            torch.cuda.set_device(gpu_id)
            
            # Memory info
            mem_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**2  # MB
            mem_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**2  # MB
            mem_total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**2  # MB
            mem_percent = (mem_allocated / mem_total) * 100
            
            # Device info
            device_name = torch.cuda.get_device_name(gpu_id)
            
            # Temperature and utilization (if NVML available)
            temperature = 0.0
            utilization = 0.0
            
            if NVML_AVAILABLE:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    utilization = util.gpu
                except:
                    pass
                    
            # Check availability
            is_available = (mem_percent < self.max_memory_percent and 
                          temperature < self.max_temperature)
                          
        elif GPUTIL_AVAILABLE:
            # Use GPUtil
            gpus = GPUtil.getGPUs()
            if gpu_id < len(gpus):
                gpu = gpus[gpu_id]
                mem_allocated = gpu.memoryUsed
                mem_total = gpu.memoryTotal
                mem_percent = gpu.memoryUtil * 100
                device_name = gpu.name
                temperature = gpu.temperature
                utilization = gpu.load * 100
                
                is_available = (mem_percent < self.max_memory_percent and 
                              temperature < self.max_temperature)
            else:
                raise ValueError(f"GPU {gpu_id} not found")
        else:
            # Fallback - assume GPU is available
            return GPUStatus(
                gpu_id=gpu_id,
                name=f"GPU {gpu_id}",
                memory_used_mb=0,
                memory_total_mb=0,
                memory_percent=0,
                temperature=0,
                utilization=0,
                is_available=True,
                timestamp=datetime.now()
            )
            
        return GPUStatus(
            gpu_id=gpu_id,
            name=device_name,
            memory_used_mb=mem_allocated,
            memory_total_mb=mem_total,
            memory_percent=mem_percent,
            temperature=temperature,
            utilization=utilization,
            is_available=is_available,
            timestamp=datetime.now()
        )
        
    def _check_gpu_health(self, gpu_stats: List[GPUStatus]):
        """Check GPU health and log warnings"""
        for stat in gpu_stats:
            # Memory warning
            if stat.memory_percent > self.max_memory_percent:
                logger.warning(f"GPU {stat.gpu_id} memory usage high: {stat.memory_percent:.1f}%")
                
            # Temperature warning
            if stat.temperature > self.max_temperature:
                logger.warning(f"GPU {stat.gpu_id} temperature high: {stat.temperature:.1f}°C")
                
            # Utilization info
            if stat.utilization > 90:
                logger.info(f"GPU {stat.gpu_id} under heavy load: {stat.utilization:.1f}%")
                
    def acquire_gpu(self, worker_id: str, timeout: float = 30.0) -> Optional[int]:
        """
        Acquire a GPU for a worker
        
        Args:
            worker_id: Unique worker identifier
            timeout: Maximum time to wait for a GPU
            
        Returns:
            GPU ID if acquired, None if timeout
        """
        try:
            gpu_id = self.gpu_queue.get(timeout=timeout)
            
            # Check if GPU is actually available
            stat = self._get_gpu_status(gpu_id)
            if not stat.is_available:
                # GPU not healthy, put it back and try another
                self.gpu_queue.put(gpu_id)
                logger.warning(f"GPU {gpu_id} not healthy, trying another...")
                return self.acquire_gpu(worker_id, timeout/2)  # Try with reduced timeout
                
            with self.worker_lock:
                self.gpu_workers[worker_id] = gpu_id
                
            logger.info(f"Worker {worker_id} acquired GPU {gpu_id}")
            return gpu_id
            
        except queue.Empty:
            logger.warning(f"Worker {worker_id} timeout waiting for GPU")
            return None
            
    def release_gpu(self, worker_id: str):
        """Release GPU from a worker"""
        with self.worker_lock:
            if worker_id in self.gpu_workers:
                gpu_id = self.gpu_workers[worker_id]
                del self.gpu_workers[worker_id]
                self.gpu_queue.put(gpu_id)
                logger.info(f"Worker {worker_id} released GPU {gpu_id}")
                
                # Clear GPU cache if using PyTorch
                if TORCH_AVAILABLE:
                    try:
                        torch.cuda.set_device(gpu_id)
                        torch.cuda.empty_cache()
                    except:
                        pass
                        
    def get_gpu_count(self) -> int:
        """Get number of available GPUs"""
        return len(self.available_gpus)
        
    def get_available_gpu_count(self) -> int:
        """Get number of currently available GPUs"""
        return self.gpu_queue.qsize()
        
    def get_worker_gpu_mapping(self) -> Dict[str, int]:
        """Get current worker to GPU mapping"""
        with self.worker_lock:
            return self.gpu_workers.copy()
            
    def get_monitoring_report(self) -> Dict:
        """Get monitoring report"""
        if not self.gpu_stats_history:
            return {}
            
        latest_stats = self.gpu_stats_history[-1]['stats']
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'monitoring_active': self.monitoring_active,
            'total_gpus': len(self.available_gpus),
            'available_gpus': self.get_available_gpu_count(),
            'active_workers': len(self.gpu_workers),
            'gpu_stats': []
        }
        
        for stat in latest_stats:
            report['gpu_stats'].append({
                'gpu_id': stat.gpu_id,
                'name': stat.name,
                'memory_used_mb': round(stat.memory_used_mb, 2),
                'memory_total_mb': round(stat.memory_total_mb, 2),
                'memory_percent': round(stat.memory_percent, 2),
                'temperature': round(stat.temperature, 1),
                'utilization': round(stat.utilization, 1),
                'is_available': stat.is_available
            })
            
        return report
        
    def save_monitoring_history(self, filepath: str):
        """Save monitoring history to file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.gpu_stats_history, f, indent=2, default=str)
            logger.info(f"Monitoring history saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save monitoring history: {e}")
            
    def should_use_cpu_fallback(self) -> bool:
        """Determine if CPU fallback should be used"""
        # No GPUs available
        if not self.available_gpus:
            return True
            
        # All GPUs are busy
        if self.get_available_gpu_count() == 0:
            return True
            
        # Check if all GPUs are unhealthy
        try:
            stats = self._collect_gpu_stats()
            healthy_gpus = sum(1 for s in stats if s.is_available)
            return healthy_gpus == 0
        except:
            return True


# Global GPU manager instance
_gpu_manager = None


def get_gpu_manager() -> GPUResourceManager:
    """Get or create global GPU manager instance"""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUResourceManager()
        _gpu_manager.start_monitoring()
    return _gpu_manager


def cleanup_gpu_manager():
    """Cleanup GPU manager"""
    global _gpu_manager
    if _gpu_manager is not None:
        _gpu_manager.stop_monitoring()
        _gpu_manager = None