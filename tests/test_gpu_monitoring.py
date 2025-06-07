#!/usr/bin/env python3
"""Test GPU resource management and monitoring"""

import os
import sys
import time
import logging
import argparse

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from processing.gpu_resource_manager import GPUResourceManager, get_gpu_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_gpu_detection():
    """Test GPU detection"""
    logger.info("Testing GPU detection...")
    
    manager = GPUResourceManager()
    gpu_count = manager.get_gpu_count()
    
    logger.info(f"Detected {gpu_count} GPUs")
    
    if gpu_count > 0:
        # Get status for each GPU
        stats = manager._collect_gpu_stats()
        for stat in stats:
            logger.info(f"GPU {stat.gpu_id}: {stat.name}")
            logger.info(f"  Memory: {stat.memory_used_mb:.1f}/{stat.memory_total_mb:.1f} MB "
                       f"({stat.memory_percent:.1f}%)")
            logger.info(f"  Temperature: {stat.temperature:.1f}°C")
            logger.info(f"  Utilization: {stat.utilization:.1f}%")
            logger.info(f"  Available: {stat.is_available}")
    
    return gpu_count > 0


def test_gpu_allocation():
    """Test GPU allocation and release"""
    logger.info("\nTesting GPU allocation...")
    
    manager = get_gpu_manager()
    
    if manager.get_gpu_count() == 0:
        logger.warning("No GPUs available for allocation test")
        return False
    
    # Test acquiring GPU
    worker_id = "test_worker_1"
    gpu_id = manager.acquire_gpu(worker_id, timeout=5.0)
    
    if gpu_id is not None:
        logger.info(f"[OK] Successfully acquired GPU {gpu_id} for {worker_id}")
        
        # Check worker mapping
        mapping = manager.get_worker_gpu_mapping()
        logger.info(f"Current worker mapping: {mapping}")
        
        # Simulate some work
        time.sleep(2)
        
        # Release GPU
        manager.release_gpu(worker_id)
        logger.info(f"[OK] Released GPU {gpu_id}")
        
        # Check available count
        available = manager.get_available_gpu_count()
        logger.info(f"Available GPUs after release: {available}")
        
        return True
    else:
        logger.error("[ERROR] Failed to acquire GPU")
        return False


def test_monitoring():
    """Test GPU monitoring"""
    logger.info("\nTesting GPU monitoring...")
    
    manager = GPUResourceManager()
    
    if manager.get_gpu_count() == 0:
        logger.warning("No GPUs available for monitoring test")
        return False
    
    # Start monitoring
    manager.start_monitoring()
    logger.info("Monitoring started, collecting data for 10 seconds...")
    
    # Let it collect some data
    for i in range(5):
        time.sleep(2)
        report = manager.get_monitoring_report()
        logger.info(f"\nMonitoring report at {i*2}s:")
        logger.info(f"  Active workers: {report['active_workers']}")
        logger.info(f"  Available GPUs: {report['available_gpus']}")
        
        for gpu_stat in report['gpu_stats']:
            logger.info(f"  GPU {gpu_stat['gpu_id']}: "
                       f"Mem {gpu_stat['memory_percent']:.1f}%, "
                       f"Temp {gpu_stat['temperature']:.1f}°C, "
                       f"Load {gpu_stat['utilization']:.1f}%")
    
    # Stop monitoring
    manager.stop_monitoring()
    
    # Save history
    manager.save_monitoring_history("test_gpu_monitoring.json")
    logger.info("[OK] Monitoring test complete, history saved")
    
    return True


def test_concurrent_allocation():
    """Test concurrent GPU allocation"""
    logger.info("\nTesting concurrent GPU allocation...")
    
    manager = get_gpu_manager()
    gpu_count = manager.get_gpu_count()
    
    if gpu_count == 0:
        logger.warning("No GPUs available for concurrent allocation test")
        return False
    
    # Try to allocate more workers than GPUs
    workers = []
    allocated = []
    
    for i in range(gpu_count + 2):  # Try to allocate 2 more than available
        worker_id = f"concurrent_worker_{i}"
        gpu_id = manager.acquire_gpu(worker_id, timeout=1.0)
        
        if gpu_id is not None:
            logger.info(f"Worker {i} acquired GPU {gpu_id}")
            workers.append(worker_id)
            allocated.append((worker_id, gpu_id))
        else:
            logger.info(f"Worker {i} could not acquire GPU (expected for workers > {gpu_count})")
    
    # Check mapping
    mapping = manager.get_worker_gpu_mapping()
    logger.info(f"Worker mapping: {mapping}")
    logger.info(f"Successfully allocated {len(allocated)} out of {gpu_count + 2} workers")
    
    # Release all
    for worker_id, gpu_id in allocated:
        manager.release_gpu(worker_id)
        logger.info(f"Released GPU {gpu_id} from {worker_id}")
    
    return len(allocated) == min(gpu_count, gpu_count + 2)


def main():
    parser = argparse.ArgumentParser(description='Test GPU monitoring and management')
    parser.add_argument('--test', choices=['detection', 'allocation', 'monitoring', 
                                          'concurrent', 'all'], 
                       default='all', help='Which test to run')
    
    args = parser.parse_args()
    
    tests_passed = 0
    tests_total = 0
    
    if args.test in ['detection', 'all']:
        tests_total += 1
        if test_gpu_detection():
            tests_passed += 1
    
    if args.test in ['allocation', 'all']:
        tests_total += 1
        if test_gpu_allocation():
            tests_passed += 1
    
    if args.test in ['monitoring', 'all']:
        tests_total += 1
        if test_monitoring():
            tests_passed += 1
    
    if args.test in ['concurrent', 'all']:
        tests_total += 1
        if test_concurrent_allocation():
            tests_passed += 1
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Test Results: {tests_passed}/{tests_total} passed")
    logger.info(f"{'='*60}")
    
    return 0 if tests_passed == tests_total else 1


if __name__ == '__main__':
    sys.exit(main())