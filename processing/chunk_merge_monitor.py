"""
Monitor and merge chunk processing results
"""

import time
import logging
from datetime import datetime, timedelta
from threading import Thread
from .cleanup_manager import get_cleanup_manager

logger = logging.getLogger(__name__)


class ChunkMergeMonitor:
    """Monitor chunk processing and merge results when all chunks are complete"""
    
    def __init__(self, app, check_interval=10):
        self.app = app
        self.check_interval = check_interval
        self.running = False
        self.thread = None
        
    def start(self):
        """Start monitoring in background thread"""
        if not self.running:
            self.running = True
            self.thread = Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()
            logger.info("Chunk merge monitor started")
            
    def stop(self):
        """Stop monitoring"""
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("Chunk merge monitor stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                self._check_and_merge_videos()
            except Exception as e:
                logger.error(f"Error in chunk merge monitor: {e}")
            
            time.sleep(self.check_interval)
            
    def _check_and_merge_videos(self):
        """Check for videos that need chunk merging"""
        with self.app.app_context():
            db = self.app.db
            Video = self.app.Video
            DetectedPerson = self.app.DetectedPerson
            
            # Find parent videos with status 'chunking_complete'
            parent_videos = Video.query.filter_by(
                status='chunking_complete',
                is_chunk=False
            ).filter(Video.parent_video_id.is_(None)).all()
            
            from processing.video_chunk_manager import VideoChunkManager
            chunk_manager = VideoChunkManager()
            
            for parent_video in parent_videos:
                logger.info(f"Checking parent video {parent_video.id} for chunk completion")
                
                # Try to merge results
                if chunk_manager.merge_chunk_results(parent_video, db, Video, DetectedPerson):
                    logger.info(f"Successfully merged results for video {parent_video.id}")
                    
                    # Clean up chunk files after successful merge
                    cleanup_manager = get_cleanup_manager()
                    cleaned = cleanup_manager.cleanup_video_chunks(parent_video.id)
                    logger.info(f"Cleaned up {cleaned} chunk directories for video {parent_video.id}")
                else:
                    # Check if chunks are stuck
                    chunks = Video.query.filter_by(
                        parent_video_id=parent_video.id,
                        is_chunk=True
                    ).all()
                    
                    stuck_chunks = []
                    for chunk in chunks:
                        if chunk.status == 'processing':
                            # Check if processing for too long (> 30 minutes)
                            if chunk.processing_started_at:
                                processing_time = datetime.utcnow() - chunk.processing_started_at
                                if processing_time > timedelta(minutes=30):
                                    stuck_chunks.append(chunk)
                                    
                    if stuck_chunks:
                        logger.warning(f"Found {len(stuck_chunks)} stuck chunks for video {parent_video.id}")
                        # Could restart stuck chunks here if needed
                        

# Global instance
_monitor = None

def get_chunk_monitor(app):
    """Get or create the global chunk monitor instance"""
    global _monitor
    if _monitor is None:
        _monitor = ChunkMergeMonitor(app)
        _monitor.start()
    return _monitor