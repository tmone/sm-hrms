"""
Video Chunk Manager - Handles splitting large videos into chunks
and queuing them for processing
"""

import os
import subprocess
from pathlib import Path
from datetime import datetime
import shutil
from .cleanup_manager import get_cleanup_manager

# Set up logging
try:
    from config_logging import get_logger
    from utils.progress_logger import VideoProcessingProgress, simple_progress
    logger = get_logger(__name__)
except:
    import logging
    logger = logging.getLogger(__name__)


class VideoChunkManager:
    """Manages video chunking and queue creation"""
    
    def __init__(self, chunk_duration=30, chunks_folder='chunks'):
        self.chunk_duration = chunk_duration
        self.chunks_folder = chunks_folder
        self.cleanup_manager = get_cleanup_manager()
        self._gpu_available = None  # Cache GPU availability check
        
    def get_video_duration(self, video_path):
        """Get video duration in seconds"""
        cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', video_path
        ]
        try:
            # Use UTF-8 encoding to handle non-ASCII characters
            output = subprocess.check_output(cmd, encoding='utf-8', errors='replace')
            duration = float(output.strip())
            return duration
        except Exception as e:
            logger.error(f"Failed to get video duration: {e}")
            return None
            
    def should_chunk_video(self, video_path, threshold=60):
        """Check if video should be chunked based on duration"""
        duration = self.get_video_duration(video_path)
        if duration is None:
            return False
        return duration > threshold
        
    def split_video_to_chunks(self, video_path, output_dir):
        """Split video into chunks using ffmpeg with copy codec"""
        duration = self.get_video_duration(video_path)
        if duration is None:
            return []
            
        # Calculate number of chunks
        num_chunks = int((duration + self.chunk_duration - 1) // self.chunk_duration)
        logger.info(f"Video duration: {duration:.1f}s, will create {num_chunks} chunks")
        
        # Create chunks directory
        video_name = Path(video_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chunks_dir = Path(output_dir) / self.chunks_folder / f"{video_name}_{timestamp}"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        
        chunk_paths = []
        
        for i in range(num_chunks):
            start_time = i * self.chunk_duration
            # Keep original extension to maintain format
            original_ext = Path(video_path).suffix
            chunk_filename = f"{video_name}_chunk_{i:03d}{original_ext}"
            chunk_path = chunks_dir / chunk_filename
            
            # Use stream copy (no re-encoding) for fastest chunking
            # This preserves original video format and quality
            cmd = [
                'ffmpeg',
                '-ss', str(start_time),  # Seek BEFORE input (much faster)
                '-i', str(video_path),
                '-t', str(self.chunk_duration),
                '-c', 'copy',  # Copy all streams without re-encoding
                '-avoid_negative_ts', 'make_zero',
                '-y',  # Overwrite
                str(chunk_path)
            ]
            logger.info(f"📋 Creating chunk {i+1}/{num_chunks} with stream copy (no re-encoding)")
            
            try:
                logger.info(f"Creating chunk {i+1}/{num_chunks}: {chunk_filename}")
                # Use encoding='utf-8' and errors='replace' to handle non-ASCII characters
                result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
                chunk_paths.append(str(chunk_path))
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to create chunk {i}: {e.stderr}")
                # Try alternative method without avoid_negative_ts
                cmd_alt = [
                    'ffmpeg',
                    '-ss', str(start_time),  # Input seeking
                    '-i', str(video_path),
                    '-t', str(self.chunk_duration),
                    '-c', 'copy',  # Still use stream copy in fallback
                    '-y',
                    str(chunk_path)
                ]
                try:
                    subprocess.run(cmd_alt, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
                    chunk_paths.append(str(chunk_path))
                    logger.info(f"✅ Chunk {i+1} created with alternative method")
                except subprocess.CalledProcessError as e2:
                    # Last resort: try with re-encoding if copy fails
                    logger.warning(f"Stream copy failed, trying with re-encoding for chunk {i}")
                    cmd_encode = [
                        'ffmpeg',
                        '-ss', str(start_time),
                        '-i', str(video_path),
                        '-t', str(self.chunk_duration),
                        '-c:v', 'libx264',
                        '-preset', 'ultrafast',
                        '-crf', '28',  # Lower quality for speed
                        '-c:a', 'aac',
                        '-y',
                        str(chunk_path)
                    ]
                    try:
                        subprocess.run(cmd_encode, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
                        chunk_paths.append(str(chunk_path))
                        logger.info(f"✅ Chunk {i+1} created with re-encoding")
                    except subprocess.CalledProcessError as e3:
                        logger.error(f"All methods failed for chunk {i}: {e3.stderr}")
                    
        logger.info(f"Successfully created {len(chunk_paths)} chunks")
        return chunk_paths
        
    def create_chunk_entries(self, parent_video, chunk_paths, db, Video, upload_dir='static/uploads'):
        """Create database entries for each chunk"""
        chunk_videos = []
        
        for idx, chunk_path in enumerate(chunk_paths):
            # Convert to relative path for storage
            # Extract the relative path from uploads folder
            chunk_path_obj = Path(chunk_path)
            upload_dir_path = Path(upload_dir)
            
            # Get relative path from uploads directory
            try:
                relative_path = chunk_path_obj.relative_to(upload_dir_path)
            except ValueError:
                # If can't get relative, use the chunk directory + filename
                relative_path = Path(self.chunks_folder) / chunk_path_obj.parent.name / chunk_path_obj.name
            
            # Create a new video entry for each chunk
            chunk_video = Video(
                filename=os.path.basename(chunk_path),
                file_path=str(relative_path).replace('\\', '/'),  # Store relative path with forward slashes
                status='uploaded',
                parent_video_id=parent_video.id,
                chunk_index=idx,
                total_chunks=len(chunk_paths),
                is_chunk=True,
                employee_id=parent_video.employee_id
            )
            
            db.session.add(chunk_video)
            chunk_videos.append(chunk_video)
            
        # Update parent video status
        parent_video.status = 'chunking_complete'
        parent_video.total_chunks = len(chunk_paths)
        
        db.session.commit()
        logger.info(f"Created {len(chunk_videos)} chunk entries in database")
        
        return chunk_videos
        
    def merge_chunk_results(self, parent_video, db, Video, DetectedPerson):
        """Merge results from all processed chunks back to parent video"""
        # Get all chunks for this parent video
        chunks = Video.query.filter_by(
            parent_video_id=parent_video.id,
            is_chunk=True
        ).order_by(Video.chunk_index).all()
        
        if not chunks:
            logger.error(f"No chunks found for parent video {parent_video.id}")
            return False
            
        # Check if all chunks are processed
        incomplete_chunks = [c for c in chunks if c.status != 'completed']
        if incomplete_chunks:
            logger.info(f"{len(incomplete_chunks)} chunks still processing")
            return False
            
        logger.info(f"All {len(chunks)} chunks completed, merging results...")
        
        # Get all detections from chunks
        all_detections = []
        total_persons = set()
        
        for chunk in chunks:
            # Get detections for this chunk
            chunk_detections = DetectedPerson.query.filter_by(video_id=chunk.id).all()
            
            # Adjust timestamps based on chunk index
            time_offset = chunk.chunk_index * self.chunk_duration
            
            for detection in chunk_detections:
                # Calculate adjusted frame number
                # Use parent video fps if chunk fps is not available
                fps = chunk.fps or parent_video.fps or 30.0  # Default to 30 fps if not available
                adjusted_frame = detection.frame_number + int(chunk.chunk_index * fps * self.chunk_duration)
                
                # Create new detection for parent video
                parent_detection = DetectedPerson(
                    video_id=parent_video.id,
                    person_id=detection.person_id,
                    frame_number=adjusted_frame,
                    timestamp=detection.timestamp + time_offset,
                    bbox_x=detection.bbox_x,
                    bbox_y=detection.bbox_y,
                    bbox_width=detection.bbox_width,
                    bbox_height=detection.bbox_height,
                    confidence=detection.confidence,
                    track_id=detection.track_id,
                    attendance_location=detection.attendance_location,
                    attendance_date=detection.attendance_date,
                    attendance_time=detection.attendance_time,
                    check_in_time=detection.check_in_time
                )
                
                db.session.add(parent_detection)
                all_detections.append(parent_detection)
                
                if detection.person_id:
                    total_persons.add(detection.person_id)
                    
        # Merge annotated videos if they exist
        annotated_videos = []
        upload_dir = 'static/uploads'
        for chunk in chunks:
            if chunk.annotated_video_path:
                # Convert relative path to full path
                full_path = os.path.join(upload_dir, chunk.annotated_video_path)
                if os.path.exists(full_path):
                    annotated_videos.append(full_path)
                else:
                    logger.warning(f"Chunk annotated video not found: {full_path}")
                
        if annotated_videos:
            # Create merged annotated video
            merged_video_path = self._merge_annotated_videos(parent_video, annotated_videos)
            if merged_video_path:
                parent_video.annotated_video_path = merged_video_path
                
        # Update parent video status
        parent_video.status = 'completed'
        parent_video.processing_progress = 100
        parent_video.processing_completed_at = datetime.utcnow()
        
        # Copy OCR data from first chunk
        first_chunk = chunks[0]
        parent_video.ocr_location = first_chunk.ocr_location
        parent_video.ocr_video_date = first_chunk.ocr_video_date
        parent_video.ocr_extraction_done = first_chunk.ocr_extraction_done
        
        db.session.commit()
        
        logger.info(f"Merged {len(all_detections)} detections from {len(chunks)} chunks")
        logger.info(f"Total unique persons: {len(total_persons)}")
        
        # Clean up chunk files if needed
        self._cleanup_chunks(chunks)
        
        return True
        
    def _merge_annotated_videos(self, parent_video, video_paths):
        """Merge multiple annotated videos into one"""
        if not video_paths:
            return None
            
        # Create output path
        output_dir = Path(video_paths[0]).parent
        output_filename = f"{parent_video.filename.rsplit('.', 1)[0]}_annotated_merged.mp4"
        output_path = output_dir / output_filename
        
        # Create concat list file
        concat_file = output_dir / "concat_list.txt"
        with open(concat_file, 'w') as f:
            for video_path in video_paths:
                f.write(f"file '{os.path.abspath(video_path)}'\n")
                
        # Use ffmpeg to concatenate and ensure H.264 output
        cmd = [
            'ffmpeg', '-f', 'concat', '-safe', '0',
            '-i', str(concat_file),
            '-c:v', 'libx264',  # Force H.264 codec
            '-preset', 'fast',  # Fast encoding
            '-crf', '23',  # Good quality
            '-pix_fmt', 'yuv420p',  # Compatibility
            '-movflags', '+faststart',  # Web optimization
            '-c:a', 'aac',  # AAC audio if present
            '-y',
            str(output_path)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
            logger.info(f"Merged annotated video created: {output_path}")
            
            # Clean up concat file
            concat_file.unlink()
            
            # Return relative path from static/uploads
            try:
                # Try to get relative path from uploads directory
                uploads_dir = Path('static/uploads')
                return str(output_path.relative_to(uploads_dir.resolve()))
            except ValueError:
                # If not in uploads, return the filename only
                return output_filename
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to merge annotated videos: {e.stderr}")
            return None
            
    def _cleanup_chunks(self, chunks):
        """Clean up chunk files and directories"""
        cleaned_count = 0
        
        # Get chunk directory
        if chunks and chunks[0].file_path:
            # Build full path
            upload_dir = 'static/uploads'
            chunk_path = os.path.join(upload_dir, chunks[0].file_path)
            chunk_dir = Path(chunk_path).parent
            
            # Use cleanup manager to remove chunk directory
            if self.cleanup_manager.cleanup_chunk_directory(chunk_dir):
                cleaned_count += 1
                
        # Also clean up any temporary files in the chunks folder
        temp_cleaned = self.cleanup_manager.cleanup_temp_files()
        logger.info(f"Cleaned up {cleaned_count} chunk directories and {temp_cleaned} temp files")