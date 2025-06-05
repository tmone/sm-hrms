#!/usr/bin/env python3
"""Test the chunked video processing for large files"""

import os
import sys
import argparse
import logging

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from processing.chunked_video_processor import ChunkedVideoProcessor
from processing.enhanced_detection import process_video_with_enhanced_detection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_chunked_processor(video_path, output_dir):
    """Test the chunked video processor directly"""
    logger.info("Testing chunked video processor...")
    
    # Create processor
    processor = ChunkedVideoProcessor(max_workers=2, chunk_duration=30)
    
    # Process video
    result = processor.process_video(video_path, output_dir)
    
    if result['status'] == 'success':
        logger.info(f"✅ Chunked processing successful!")
        logger.info(f"   Total detections: {len(result['detections'])}")
        logger.info(f"   Unique persons: {result['metadata']['unique_persons']}")
        logger.info(f"   Processing time: {result['metadata']['processing_time']:.1f}s")
        logger.info(f"   Annotated video: {result['annotated_video']}")
        
        # Check quality of extracted persons
        persons_with_quality = []
        for det in result['detections']:
            if det.get('quality_score', 0) > 70:
                persons_with_quality.append(det['person_id'])
        
        unique_quality_persons = len(set(persons_with_quality))
        logger.info(f"   High quality person crops: {unique_quality_persons} persons")
        
        return True
    else:
        logger.error(f"❌ Chunked processing failed: {result.get('error')}")
        return False


def test_enhanced_detection(video_path, output_dir):
    """Test the enhanced detection with chunking for all videos"""
    logger.info("Testing enhanced detection with chunking...")
    
    # Get file size
    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    logger.info(f"Video file size: {file_size_mb:.1f} MB")
    
    # Process video
    result = process_video_with_enhanced_detection(video_path, output_dir)
    
    if result and result.get('success'):
        logger.info(f"✅ Enhanced detection successful!")
        logger.info(f"   Processing method: {result['summary']['processing_method']}")
        logger.info(f"   Total persons: {result['summary']['total_persons']}")
        logger.info(f"   Annotated video: {result['annotated_video']}")
        
        # Check person tracks
        for person_id, tracks in result['person_tracks'].items():
            if len(tracks) > 10:  # Only show persons with significant appearances
                logger.info(f"   {person_id}: {len(tracks)} detections, "
                          f"duration: {tracks[-1]['timestamp'] - tracks[0]['timestamp']:.1f}s")
        
        return True
    else:
        logger.error(f"❌ Enhanced detection failed: {result.get('error') if result else 'Unknown error'}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Test chunked video processing')
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument('--output-dir', default='./test_chunked_output',
                       help='Output directory (default: ./test_chunked_output)')
    parser.add_argument('--mode', choices=['chunked', 'enhanced', 'both'], default='both',
                       help='Test mode (default: both)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        logger.error(f"Video file not found: {args.video_path}")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    success = True
    
    if args.mode in ['chunked', 'both']:
        chunked_output = os.path.join(args.output_dir, 'chunked_test')
        os.makedirs(chunked_output, exist_ok=True)
        
        logger.info("\n" + "="*60)
        logger.info("TESTING CHUNKED PROCESSOR")
        logger.info("="*60)
        
        if not test_chunked_processor(args.video_path, chunked_output):
            success = False
    
    if args.mode in ['enhanced', 'both']:
        enhanced_output = os.path.join(args.output_dir, 'enhanced_test')
        os.makedirs(enhanced_output, exist_ok=True)
        
        logger.info("\n" + "="*60)
        logger.info("TESTING ENHANCED DETECTION WITH AUTO-CHUNKING")
        logger.info("="*60)
        
        if not test_enhanced_detection(args.video_path, enhanced_output):
            success = False
    
    if success:
        logger.info("\n✅ All tests passed!")
        return 0
    else:
        logger.error("\n❌ Some tests failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())