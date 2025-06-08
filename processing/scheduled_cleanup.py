"""
Scheduled cleanup tasks for the system
"""
import logging
from datetime import datetime
from threading import Thread, Event
import time
from .cleanup_manager import get_cleanup_manager

logger = logging.getLogger(__name__)


class ScheduledCleanup:
    """Runs cleanup tasks on a schedule"""
    
    def __init__(self, cleanup_interval_hours=6):
        self.cleanup_interval_hours = cleanup_interval_hours
        self.running = False
        self.thread = None
        self.stop_event = Event()
        self.cleanup_manager = get_cleanup_manager()
        
    def start(self):
        """Start scheduled cleanup"""
        if not self.running:
            self.running = True
            self.stop_event.clear()
            self.thread = Thread(target=self._cleanup_loop, daemon=True)
            self.thread.start()
            logger.info(f"Scheduled cleanup started (runs every {self.cleanup_interval_hours} hours)")
            
    def stop(self):
        """Stop scheduled cleanup"""
        self.running = False
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        logger.info("Scheduled cleanup stopped")
        
    def _cleanup_loop(self):
        """Main cleanup loop"""
        while self.running:
            try:
                # Run cleanup
                self._run_cleanup()
                
                # Wait for next interval or stop signal
                wait_seconds = self.cleanup_interval_hours * 3600
                if self.stop_event.wait(wait_seconds):
                    break
                    
            except Exception as e:
                logger.error(f"Error in scheduled cleanup: {e}")
                # Wait a bit before retrying
                if self.stop_event.wait(300):  # 5 minutes
                    break
                    
    def _run_cleanup(self):
        """Run cleanup tasks"""
        logger.info(f"Running scheduled cleanup at {datetime.now()}")
        
        try:
            # Perform full cleanup
            stats = self.cleanup_manager.perform_full_cleanup()
            
            # Log results
            logger.info(f"Scheduled cleanup complete:")
            logger.info(f"  - Cleaned {stats['non_person_dirs']} non-person directories")
            logger.info(f"  - Cleaned {stats['old_chunks']} old chunk directories")
            logger.info(f"  - Cleaned {stats['temp_files']} temporary files")
            logger.info(f"  - Cleaned {stats['empty_dirs']} empty directories")
            logger.info(f"  - Total: {stats['total']} items cleaned")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            
    def run_immediate(self):
        """Run cleanup immediately"""
        self._run_cleanup()


# Global instance
_scheduled_cleanup = None


def get_scheduled_cleanup(cleanup_interval_hours=6):
    """Get or create scheduled cleanup instance"""
    global _scheduled_cleanup
    if _scheduled_cleanup is None:
        _scheduled_cleanup = ScheduledCleanup(cleanup_interval_hours)
    return _scheduled_cleanup


def start_scheduled_cleanup(cleanup_interval_hours=6):
    """Start the scheduled cleanup service"""
    cleanup = get_scheduled_cleanup(cleanup_interval_hours)
    cleanup.start()
    return cleanup


def stop_scheduled_cleanup():
    """Stop the scheduled cleanup service"""
    global _scheduled_cleanup
    if _scheduled_cleanup:
        _scheduled_cleanup.stop()
        _scheduled_cleanup = None