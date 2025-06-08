"""
Checkpoint Manager - Saves processing state to allow resumption after crashes
"""
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import pickle
import shutil

try:
    from config_logging import get_logger
    logger = get_logger(__name__)
except:
    import logging
    logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoints for video processing to enable crash recovery"""
    
    def __init__(self, checkpoint_dir: str = "processing/checkpoints"):
        """
        Initialize checkpoint manager
        
        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoint manager initialized at {self.checkpoint_dir}")
    
    def create_checkpoint(self, video_id: int, checkpoint_data: Dict[str, Any]) -> str:
        """
        Create a checkpoint for video processing
        
        Args:
            video_id: Video ID being processed
            checkpoint_data: Data to save in checkpoint
            
        Returns:
            Checkpoint file path
        """
        checkpoint_id = f"video_{video_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.json"
        
        # Add metadata
        checkpoint_data['checkpoint_id'] = checkpoint_id
        checkpoint_data['video_id'] = video_id
        checkpoint_data['created_at'] = datetime.now().isoformat()
        checkpoint_data['checkpoint_version'] = "1.0"
        
        # Save checkpoint
        try:
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
            logger.info(f"Checkpoint created: {checkpoint_path}")
            
            # Also save a backup
            backup_path = checkpoint_path.with_suffix('.json.bak')
            shutil.copy2(checkpoint_path, backup_path)
            
            return str(checkpoint_path)
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            raise
    
    def update_checkpoint(self, video_id: int, updates: Dict[str, Any]) -> bool:
        """
        Update existing checkpoint for a video
        
        Args:
            video_id: Video ID
            updates: Data to update in checkpoint
            
        Returns:
            True if updated successfully
        """
        checkpoint_path = self.get_latest_checkpoint(video_id)
        if not checkpoint_path:
            logger.warning(f"No checkpoint found for video {video_id}")
            return False
        
        try:
            # Load existing checkpoint
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            # Update data
            checkpoint_data.update(updates)
            checkpoint_data['updated_at'] = datetime.now().isoformat()
            
            # Save updated checkpoint
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
            
            logger.info(f"Checkpoint updated: {checkpoint_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to update checkpoint: {e}")
            return False
    
    def get_latest_checkpoint(self, video_id: int) -> Optional[Path]:
        """
        Get the latest checkpoint file for a video
        
        Args:
            video_id: Video ID
            
        Returns:
            Path to latest checkpoint or None
        """
        pattern = f"video_{video_id}_*.json"
        checkpoints = list(self.checkpoint_dir.glob(pattern))
        
        if not checkpoints:
            return None
        
        # Get most recent checkpoint
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return latest
    
    def load_checkpoint(self, video_id: int) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint data for a video
        
        Args:
            video_id: Video ID
            
        Returns:
            Checkpoint data or None
        """
        checkpoint_path = self.get_latest_checkpoint(video_id)
        if not checkpoint_path:
            return None
        
        try:
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
            logger.info(f"Checkpoint loaded for video {video_id}")
            return data
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            # Try backup
            backup_path = checkpoint_path.with_suffix('.json.bak')
            if backup_path.exists():
                try:
                    with open(backup_path, 'r') as f:
                        data = json.load(f)
                    logger.info(f"Checkpoint loaded from backup for video {video_id}")
                    return data
                except:
                    pass
            return None
    
    def delete_checkpoint(self, video_id: int) -> bool:
        """
        Delete checkpoint files for a video (after successful completion)
        
        Args:
            video_id: Video ID
            
        Returns:
            True if deleted successfully
        """
        pattern = f"video_{video_id}_*"
        checkpoints = list(self.checkpoint_dir.glob(pattern))
        
        if not checkpoints:
            return True
        
        try:
            for checkpoint in checkpoints:
                checkpoint.unlink()
                logger.info(f"Deleted checkpoint: {checkpoint}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete checkpoints: {e}")
            return False
    
    def cleanup_old_checkpoints(self, days: int = 7) -> int:
        """
        Clean up checkpoints older than specified days
        
        Args:
            days: Delete checkpoints older than this many days
            
        Returns:
            Number of checkpoints deleted
        """
        from datetime import timedelta
        cutoff_time = datetime.now() - timedelta(days=days)
        deleted_count = 0
        
        for checkpoint in self.checkpoint_dir.glob("*.json"):
            try:
                # Check file modification time
                mtime = datetime.fromtimestamp(checkpoint.stat().st_mtime)
                if mtime < cutoff_time:
                    checkpoint.unlink()
                    # Also delete backup
                    backup = checkpoint.with_suffix('.json.bak')
                    if backup.exists():
                        backup.unlink()
                    deleted_count += 1
                    logger.info(f"Deleted old checkpoint: {checkpoint}")
            except Exception as e:
                logger.error(f"Error deleting checkpoint {checkpoint}: {e}")
        
        return deleted_count
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all checkpoints with their status
        
        Returns:
            List of checkpoint information
        """
        checkpoints = []
        
        for checkpoint_path in self.checkpoint_dir.glob("*.json"):
            try:
                with open(checkpoint_path, 'r') as f:
                    data = json.load(f)
                
                info = {
                    'video_id': data.get('video_id'),
                    'checkpoint_id': data.get('checkpoint_id'),
                    'created_at': data.get('created_at'),
                    'updated_at': data.get('updated_at'),
                    'last_processed_frame': data.get('last_processed_frame', 0),
                    'total_frames': data.get('total_frames', 0),
                    'status': data.get('status', 'unknown'),
                    'file_path': str(checkpoint_path)
                }
                checkpoints.append(info)
            except Exception as e:
                logger.error(f"Error reading checkpoint {checkpoint_path}: {e}")
        
        return sorted(checkpoints, key=lambda x: x['created_at'], reverse=True)


class VideoProcessingCheckpoint:
    """Helper class for video processing checkpoints"""
    
    def __init__(self, manager: CheckpointManager, video_id: int):
        self.manager = manager
        self.video_id = video_id
        self.checkpoint_interval = 100  # Save every 100 frames
        self.last_checkpoint_frame = 0
    
    def should_checkpoint(self, current_frame: int) -> bool:
        """Check if we should create a checkpoint"""
        return current_frame - self.last_checkpoint_frame >= self.checkpoint_interval
    
    def save_progress(self, data: Dict[str, Any]) -> bool:
        """Save processing progress"""
        current_frame = data.get('last_processed_frame', 0)
        
        if self.should_checkpoint(current_frame):
            success = self.manager.update_checkpoint(self.video_id, data)
            if success:
                self.last_checkpoint_frame = current_frame
            return success
        return True
    
    def get_resume_point(self) -> Optional[Dict[str, Any]]:
        """Get resume point from checkpoint"""
        return self.manager.load_checkpoint(self.video_id)


# Global checkpoint manager instance
_checkpoint_manager_instance = None

def get_checkpoint_manager() -> CheckpointManager:
    """Get or create global checkpoint manager instance"""
    global _checkpoint_manager_instance
    if _checkpoint_manager_instance is None:
        _checkpoint_manager_instance = CheckpointManager()
    return _checkpoint_manager_instance