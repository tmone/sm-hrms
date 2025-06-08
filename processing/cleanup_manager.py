"""
Cleanup Manager - Handles cleanup of temporary files and directories
"""
import os
import shutil
import logging
from pathlib import Path
from datetime import datetime, timedelta
import re

logger = logging.getLogger(__name__)


class CleanupManager:
    """Manages cleanup of temporary files and directories"""
    
    def __init__(self):
        self.logger = logger
        
    def cleanup_chunk_directory(self, chunk_dir: Path) -> bool:
        """
        Clean up a chunk directory and all its contents
        
        Args:
            chunk_dir: Path to chunk directory
            
        Returns:
            True if cleaned up successfully
        """
        try:
            if chunk_dir.exists() and chunk_dir.is_dir():
                # Check if it's a chunk directory (contains 'chunk' in name)
                if 'chunk' in str(chunk_dir).lower():
                    shutil.rmtree(chunk_dir)
                    self.logger.info(f"Cleaned up chunk directory: {chunk_dir}")
                    return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to clean up chunk directory {chunk_dir}: {e}")
            return False
            
    def cleanup_video_chunks(self, video_id: int, upload_dir: str = 'static/uploads') -> int:
        """
        Clean up all chunk directories for a specific video
        
        Args:
            video_id: Video ID
            upload_dir: Upload directory path
            
        Returns:
            Number of directories cleaned
        """
        cleaned_count = 0
        upload_path = Path(upload_dir)
        
        # Find all chunk directories for this video
        # Pattern: chunks/*_videoname_*
        chunks_dir = upload_path / 'chunks'
        if chunks_dir.exists():
            for chunk_dir in chunks_dir.iterdir():
                if chunk_dir.is_dir():
                    # Clean up directory
                    if self.cleanup_chunk_directory(chunk_dir):
                        cleaned_count += 1
                        
        return cleaned_count
        
    def cleanup_non_person_directories(self, persons_dir: str = 'processing/outputs/persons') -> int:
        """
        Clean up directories that don't match PERSON-XXXX pattern
        
        Args:
            persons_dir: Directory containing person folders
            
        Returns:
            Number of directories cleaned
        """
        cleaned_count = 0
        persons_path = Path(persons_dir)
        
        if not persons_path.exists():
            return 0
            
        # Pattern for valid PERSON directories
        person_pattern = re.compile(r'^PERSON-\d{4}$')
        
        for item in persons_path.iterdir():
            if item.is_dir():
                # Check if directory name matches PERSON-XXXX pattern
                if not person_pattern.match(item.name):
                    try:
                        shutil.rmtree(item)
                        self.logger.info(f"Cleaned up non-person directory: {item}")
                        cleaned_count += 1
                    except Exception as e:
                        self.logger.error(f"Failed to clean up directory {item}: {e}")
                        
        return cleaned_count
        
    def cleanup_old_chunks(self, upload_dir: str = 'static/uploads', hours_old: int = 24) -> int:
        """
        Clean up chunk directories older than specified hours
        
        Args:
            upload_dir: Upload directory path
            hours_old: Age threshold in hours
            
        Returns:
            Number of directories cleaned
        """
        cleaned_count = 0
        upload_path = Path(upload_dir)
        chunks_dir = upload_path / 'chunks'
        
        if not chunks_dir.exists():
            return 0
            
        # Calculate cutoff time
        cutoff_time = datetime.now() - timedelta(hours=hours_old)
        
        for chunk_dir in chunks_dir.iterdir():
            if chunk_dir.is_dir():
                try:
                    # Check directory modification time
                    dir_mtime = datetime.fromtimestamp(chunk_dir.stat().st_mtime)
                    
                    if dir_mtime < cutoff_time:
                        shutil.rmtree(chunk_dir)
                        self.logger.info(f"Cleaned up old chunk directory: {chunk_dir}")
                        cleaned_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to clean up old directory {chunk_dir}: {e}")
                    
        return cleaned_count
        
    def cleanup_empty_directories(self, base_dir: str) -> int:
        """
        Clean up empty directories recursively
        
        Args:
            base_dir: Base directory to search
            
        Returns:
            Number of directories cleaned
        """
        cleaned_count = 0
        base_path = Path(base_dir)
        
        if not base_path.exists():
            return 0
            
        # Walk directory tree bottom-up
        for dirpath, dirnames, filenames in os.walk(base_dir, topdown=False):
            dir_path = Path(dirpath)
            
            # Skip if directory has files or subdirectories
            if not filenames and not dirnames:
                try:
                    dir_path.rmdir()
                    self.logger.info(f"Removed empty directory: {dir_path}")
                    cleaned_count += 1
                except Exception as e:
                    self.logger.debug(f"Could not remove directory {dir_path}: {e}")
                    
        return cleaned_count
        
    def cleanup_temp_files(self, patterns: list = None) -> int:
        """
        Clean up temporary files matching patterns
        
        Args:
            patterns: List of file patterns to clean (e.g., ['*.tmp', '*.temp'])
            
        Returns:
            Number of files cleaned
        """
        if patterns is None:
            patterns = ['*.tmp', '*.temp', '*.log', 'concat_list.txt']
            
        cleaned_count = 0
        
        # Search in common temp directories
        temp_dirs = [
            'processing/temp',
            'processing/outputs/temp',
            'static/uploads/temp',
            'static/uploads/chunks'
        ]
        
        for temp_dir in temp_dirs:
            temp_path = Path(temp_dir)
            if temp_path.exists():
                for pattern in patterns:
                    for file_path in temp_path.rglob(pattern):
                        try:
                            file_path.unlink()
                            self.logger.info(f"Removed temp file: {file_path}")
                            cleaned_count += 1
                        except Exception as e:
                            self.logger.debug(f"Could not remove file {file_path}: {e}")
                            
        return cleaned_count
        
    def perform_full_cleanup(self) -> dict:
        """
        Perform comprehensive cleanup of all temporary files and directories
        
        Returns:
            Dictionary with cleanup statistics
        """
        self.logger.info("Starting full cleanup...")
        
        stats = {
            'non_person_dirs': 0,
            'old_chunks': 0,
            'temp_files': 0,
            'empty_dirs': 0,
            'total': 0
        }
        
        # Clean up non-person directories
        stats['non_person_dirs'] = self.cleanup_non_person_directories()
        
        # Clean up old chunk directories (older than 24 hours)
        stats['old_chunks'] = self.cleanup_old_chunks(hours_old=24)
        
        # Clean up temporary files
        stats['temp_files'] = self.cleanup_temp_files()
        
        # Clean up empty directories
        stats['empty_dirs'] = self.cleanup_empty_directories('processing/outputs')
        stats['empty_dirs'] += self.cleanup_empty_directories('static/uploads/chunks')
        
        # Calculate total
        stats['total'] = sum(stats.values()) - stats['total']
        
        self.logger.info(f"Cleanup complete: {stats['total']} items cleaned")
        self.logger.info(f"  - Non-person directories: {stats['non_person_dirs']}")
        self.logger.info(f"  - Old chunk directories: {stats['old_chunks']}")
        self.logger.info(f"  - Temporary files: {stats['temp_files']}")
        self.logger.info(f"  - Empty directories: {stats['empty_dirs']}")
        
        return stats


# Singleton instance
_cleanup_manager = None


def get_cleanup_manager() -> CleanupManager:
    """Get or create cleanup manager instance"""
    global _cleanup_manager
    if _cleanup_manager is None:
        _cleanup_manager = CleanupManager()
    return _cleanup_manager