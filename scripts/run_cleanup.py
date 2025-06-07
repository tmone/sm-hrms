#!/usr/bin/env python3
"""
Run cleanup of temporary files and directories
Can be run manually or scheduled as a cron job
"""
import os
import sys
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processing.cleanup_manager import get_cleanup_manager


def main():
    parser = argparse.ArgumentParser(description='Clean up temporary files and directories')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be cleaned without actually cleaning')
    parser.add_argument('--chunks-only', action='store_true',
                       help='Only clean up chunk directories')
    parser.add_argument('--non-person-only', action='store_true',
                       help='Only clean up non-PERSON directories')
    parser.add_argument('--hours-old', type=int, default=24,
                       help='Clean chunks older than this many hours (default: 24)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Cleanup Manager - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    cleanup_manager = get_cleanup_manager()
    
    if args.dry_run:
        print("[SEARCH] DRY RUN MODE - No files will be deleted\n")
        
    if args.chunks_only:
        print("[DELETE]  Cleaning up chunk directories...")
        if not args.dry_run:
            count = cleanup_manager.cleanup_old_chunks(hours_old=args.hours_old)
            print(f"[OK] Cleaned up {count} old chunk directories")
            
    elif args.non_person_only:
        print("[DELETE]  Cleaning up non-PERSON directories...")
        if not args.dry_run:
            count = cleanup_manager.cleanup_non_person_directories()
            print(f"[OK] Cleaned up {count} non-person directories")
            
    else:
        print("[DELETE]  Performing full cleanup...")
        if not args.dry_run:
            stats = cleanup_manager.perform_full_cleanup()
            
            print("\n[INFO] Cleanup Summary:")
            print(f"  • Non-person directories: {stats['non_person_dirs']}")
            print(f"  • Old chunk directories: {stats['old_chunks']}")
            print(f"  • Temporary files: {stats['temp_files']}")
            print(f"  • Empty directories: {stats['empty_dirs']}")
            print(f"  • Total items cleaned: {stats['total']}")
            
    print(f"\n{'='*60}")
    print("[OK] Cleanup complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()