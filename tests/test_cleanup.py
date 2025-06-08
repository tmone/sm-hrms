#!/usr/bin/env python3
"""
Test cleanup functionality
"""
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from processing.cleanup_manager import get_cleanup_manager


def create_test_directories():
    """Create test directories for cleanup testing"""
    print("Creating test directories...")
    
    # Create test chunk directories
    chunks_dir = Path("static/uploads/chunks")
    chunks_dir.mkdir(parents=True, exist_ok=True)
    
    test_chunk_dir = chunks_dir / "test_video_20250101_120000"
    test_chunk_dir.mkdir(exist_ok=True)
    (test_chunk_dir / "chunk_000.mp4").touch()
    (test_chunk_dir / "chunk_001.mp4").touch()
    print(f"  [OK] Created test chunk directory: {test_chunk_dir}")
    
    # Create test non-PERSON directories
    persons_dir = Path("processing/outputs/persons")
    persons_dir.mkdir(parents=True, exist_ok=True)
    
    # Valid PERSON directory (should NOT be deleted)
    valid_person = persons_dir / "PERSON-0001"
    valid_person.mkdir(exist_ok=True)
    (valid_person / "test.jpg").touch()
    print(f"  [OK] Created valid PERSON directory: {valid_person}")
    
    # Invalid directories (should be deleted)
    invalid_dirs = [
        "UNKNOWN-0001",
        "temp_processing",
        "chunk_000",
        "test_directory"
    ]
    
    for dir_name in invalid_dirs:
        invalid_dir = persons_dir / dir_name
        invalid_dir.mkdir(exist_ok=True)
        (invalid_dir / "test.jpg").touch()
        print(f"  [OK] Created invalid directory: {invalid_dir}")
        
    # Create temp files
    temp_files = [
        chunks_dir / "concat_list.txt",
        chunks_dir / "test.tmp",
        chunks_dir / "processing.log"
    ]
    
    for temp_file in temp_files:
        temp_file.touch()
        print(f"  [OK] Created temp file: {temp_file}")
        
    print("\nTest directories created successfully!")


def run_cleanup_test():
    """Run cleanup and show results"""
    print("\n" + "="*60)
    print("Running Cleanup Test")
    print("="*60 + "\n")
    
    cleanup_manager = get_cleanup_manager()
    
    # Show what exists before cleanup
    print("Before cleanup:")
    chunks_dir = Path("static/uploads/chunks")
    persons_dir = Path("processing/outputs/persons")
    
    if chunks_dir.exists():
        print(f"\nChunk directories: {list(chunks_dir.iterdir())}")
    
    if persons_dir.exists():
        print(f"\nPerson directories: {list(persons_dir.iterdir())}")
        
    # Perform cleanup
    print("\n" + "-"*40)
    print("Performing cleanup...")
    print("-"*40 + "\n")
    
    stats = cleanup_manager.perform_full_cleanup()
    
    # Show results
    print("\nCleanup Results:")
    print(f"  • Non-person directories cleaned: {stats['non_person_dirs']}")
    print(f"  • Old chunk directories cleaned: {stats['old_chunks']}")
    print(f"  • Temporary files cleaned: {stats['temp_files']}")
    print(f"  • Empty directories cleaned: {stats['empty_dirs']}")
    print(f"  • Total items cleaned: {stats['total']}")
    
    # Show what remains after cleanup
    print("\nAfter cleanup:")
    
    if chunks_dir.exists():
        remaining_chunks = list(chunks_dir.iterdir())
        print(f"\nRemaining in chunks: {remaining_chunks if remaining_chunks else 'Empty'}")
    
    if persons_dir.exists():
        remaining_persons = list(persons_dir.iterdir())
        print(f"\nRemaining in persons: {remaining_persons}")
        
        # Verify PERSON-0001 was NOT deleted
        if (persons_dir / "PERSON-0001").exists():
            print("  [OK] Valid PERSON-0001 directory was preserved!")
        else:
            print("  [ERROR] ERROR: PERSON-0001 was incorrectly deleted!")
            
    print("\n" + "="*60)
    print("Cleanup test completed!")
    print("="*60)


def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test cleanup functionality')
    parser.add_argument('--create-only', action='store_true',
                       help='Only create test directories without cleanup')
    parser.add_argument('--cleanup-only', action='store_true',
                       help='Only run cleanup without creating test directories')
    
    args = parser.parse_args()
    
    if args.cleanup_only:
        run_cleanup_test()
    elif args.create_only:
        create_test_directories()
    else:
        # Run full test
        create_test_directories()
        print("\nPress Enter to run cleanup test...")
        input()
        run_cleanup_test()


if __name__ == "__main__":
    main()