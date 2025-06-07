"""
GPU Processing Queue for safe video chunk processing
Ensures only one chunk is processed at a time to prevent GPU memory issues
"""
import queue
import threading
import logging
import time
from typing import Optional, Dict, Any, Callable
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class GPUProcessingQueue:
    """Thread-safe queue for GPU video processing tasks"""
    
    def __init__(self, max_workers: int = 1):
        """
        Initialize GPU processing queue
        
        Args:
            max_workers: Maximum number of concurrent GPU workers (default: 1 for safety)
        """
        self.max_workers = max_workers
        self.task_queue = queue.Queue()
        self.active_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
        self.workers = []
        self.shutdown_event = threading.Event()
        self.lock = threading.Lock()
        self.task_counter = 0
        
        # Start worker threads
        self._start_workers()
        
        logger.info(f"GPU Processing Queue initialized with {max_workers} workers")
        
    def _start_workers(self):
        """Start worker threads"""
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(i,),
                daemon=True,
                name=f"GPUWorker-{i}"
            )
            worker.start()
            self.workers.append(worker)
            
    def _worker_loop(self, worker_id: int):
        """Worker loop that processes tasks from the queue"""
        logger.info(f"Worker {worker_id} started")
        
        while not self.shutdown_event.is_set():
            try:
                # Get task with timeout to check shutdown
                task = self.task_queue.get(timeout=1.0)
                if task is None:  # Poison pill
                    break
                    
                task_id = task['id']
                logger.info(f"Worker {worker_id} processing task {task_id}")
                
                # Mark as active
                with self.lock:
                    self.active_tasks[task_id] = {
                        'worker_id': worker_id,
                        'start_time': datetime.now(),
                        'task': task
                    }
                
                try:
                    # Execute the task
                    result = self._execute_task(task)
                    
                    # Mark as completed
                    with self.lock:
                        self.completed_tasks[task_id] = {
                            'result': result,
                            'end_time': datetime.now(),
                            'worker_id': worker_id
                        }
                        del self.active_tasks[task_id]
                        
                    logger.info(f"Task {task_id} completed successfully")
                    
                except Exception as e:
                    # Mark as failed
                    with self.lock:
                        self.failed_tasks[task_id] = {
                            'error': str(e),
                            'end_time': datetime.now(),
                            'worker_id': worker_id
                        }
                        if task_id in self.active_tasks:
                            del self.active_tasks[task_id]
                            
                    logger.error(f"Task {task_id} failed: {e}")
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                
        logger.info(f"Worker {worker_id} stopped")
        
    def _execute_task(self, task: Dict[str, Any]) -> Any:
        """Execute a processing task"""
        task_type = task.get('type')
        
        if task_type == 'process_chunk':
            return self._process_chunk(task)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
            
    def _process_chunk(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a video chunk"""
        video_path = task['video_path']
        chunk_info = task['chunk_info']
        processing_func = task['processing_func']
        app = task.get('app')
        
        logger.info(f"Processing chunk {chunk_info['index']+1}/{chunk_info['total']} of {video_path}")
        
        # Execute within app context if app is provided
        if app:
            with app.app_context():
                result = processing_func(video_path, chunk_info, app)
        else:
            # Call without app context
            result = processing_func(video_path, chunk_info, app)
        
        return result
        
    def add_chunk_task(self, video_path: str, chunk_info: Dict[str, Any], 
                      processing_func: Callable, app: Any = None) -> int:
        """
        Add a chunk processing task to the queue
        
        Args:
            video_path: Path to the video chunk
            chunk_info: Dictionary with chunk metadata (index, total, video_id, etc.)
            processing_func: Function to call for processing
            app: Flask app instance for context
            
        Returns:
            Task ID
        """
        with self.lock:
            self.task_counter += 1
            task_id = self.task_counter
            
        task = {
            'id': task_id,
            'type': 'process_chunk',
            'video_path': video_path,
            'chunk_info': chunk_info,
            'processing_func': processing_func,
            'app': app,
            'queued_time': datetime.now()
        }
        
        self.task_queue.put(task)
        logger.info(f"Added task {task_id} to queue (queue size: {self.task_queue.qsize()})")
        
        return task_id
        
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        with self.lock:
            return {
                'queue_size': self.task_queue.qsize(),
                'active_tasks': len(self.active_tasks),
                'completed_tasks': len(self.completed_tasks),
                'failed_tasks': len(self.failed_tasks),
                'workers': self.max_workers,
                'active_task_ids': list(self.active_tasks.keys())
            }
            
    def wait_for_task(self, task_id: int, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Wait for a specific task to complete
        
        Args:
            task_id: Task ID to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            Task result or error
        """
        start_time = time.time()
        
        while True:
            with self.lock:
                if task_id in self.completed_tasks:
                    return {'status': 'completed', 'result': self.completed_tasks[task_id]}
                elif task_id in self.failed_tasks:
                    return {'status': 'failed', 'error': self.failed_tasks[task_id]}
                    
            if timeout and (time.time() - start_time) > timeout:
                return {'status': 'timeout', 'error': 'Task timed out'}
                
            time.sleep(0.1)
            
    def shutdown(self, wait: bool = True):
        """Shutdown the queue and workers"""
        logger.info("Shutting down GPU processing queue...")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Send poison pills
        for _ in range(self.max_workers):
            self.task_queue.put(None)
            
        if wait:
            # Wait for workers to finish
            for worker in self.workers:
                worker.join(timeout=30)
                
        logger.info("GPU processing queue shutdown complete")


# Global instance
_gpu_queue_instance = None
_gpu_queue_lock = threading.Lock()


def get_gpu_processing_queue(max_workers: int = 1) -> GPUProcessingQueue:
    """Get or create the global GPU processing queue instance"""
    global _gpu_queue_instance
    
    with _gpu_queue_lock:
        if _gpu_queue_instance is None:
            _gpu_queue_instance = GPUProcessingQueue(max_workers=max_workers)
        return _gpu_queue_instance