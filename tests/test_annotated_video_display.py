#!/usr/bin/env python3
"""
Test script to verify annotated video display functionality
Updates a video record to point to an annotated video for testing
"""

import sys
import os
import sqlite3

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def find_database():
    """Find the database file"""
    possible_paths = [
        'instance/stepmedia_hrm.db',
        'stepmedia_hrm.db',
        'hr_management.db'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def get_available_videos(db_path):
    """Get list of available videos"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, filename, status, annotated_video_path FROM videos ORDER BY id")
    videos = cursor.fetchall()
    
    conn.close()
    return videos

def get_available_outputs():
    """Get list of available annotated video outputs"""
    outputs_dir = 'processing/outputs'
    if not os.path.exists(outputs_dir):
        return []
    
    outputs = []
    for folder in os.listdir(outputs_dir):
        folder_path = os.path.join(outputs_dir, folder)
        if os.path.isdir(folder_path) and folder.startswith('detected_'):
            # Look for video file in the folder
            for file in os.listdir(folder_path):
                if file.endswith('.mp4') and file.startswith('detected_'):
                    video_path = os.path.join(folder, file)
                    outputs.append(video_path)
                    break
    
    return outputs

def update_video_annotated_path(db_path, video_id, annotated_path):
    """Update a video record with annotated video path"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute(
        "UPDATE videos SET annotated_video_path = ? WHERE id = ?",
        (annotated_path, video_id)
    )
    
    conn.commit()
    rows_affected = cursor.rowcount
    conn.close()
    
    return rows_affected > 0

def main():
    print("🧪 Testing Annotated Video Display Functionality")
    print("=" * 50)
    
    # Find database
    db_path = find_database()
    if not db_path:
        print("❌ Database not found!")
        return False
    
    print(f"📁 Found database: {db_path}")
    
    # Get available videos
    videos = get_available_videos(db_path)
    if not videos:
        print("❌ No videos found in database!")
        return False
    
    print(f"\n📊 Available videos:")
    for video in videos:
        status_icon = "✅" if video[3] else "⚪"
        print(f"   {status_icon} Video {video[0]}: {video[1]} (status: {video[2]})")
        if video[3]:
            print(f"      📁 Annotated: {video[3]}")
    
    # Get available outputs
    outputs = get_available_outputs()
    if not outputs:
        print("❌ No annotated video outputs found!")
        return False
    
    print(f"\n📁 Available annotated outputs:")
    for i, output in enumerate(outputs):
        print(f"   {i+1}. {output}")
    
    # Update first video without annotated path
    target_video = None
    for video in videos:
        if not video[3]:  # No annotated_video_path
            target_video = video
            break
    
    if not target_video:
        print("ℹ️ All videos already have annotated paths!")
        return True
    
    # Use first available output
    if outputs:
        annotated_path = outputs[0]
        print(f"\n🔧 Updating video {target_video[0]} with annotated path: {annotated_path}")
        
        success = update_video_annotated_path(db_path, target_video[0], annotated_path)
        
        if success:
            print("✅ Video record updated successfully!")
            print(f"\n🎯 Test Instructions:")
            print(f"1. Start your Flask app: python3 app.py")
            print(f"2. Navigate to video {target_video[0]} detail page")
            print(f"3. You should see the enhanced detection video player")
            print(f"4. The video should show: 🎯 Enhanced Detection Video")
            print(f"5. Below should show enhanced detection statistics")
            return True
        else:
            print("❌ Failed to update video record!")
            return False
    
    return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Annotated video display test setup complete!")
        print("Start your Flask app to see the enhanced video player in action!")
    else:
        print("\n⚠️ Test setup failed. Check the errors above.")
    
    sys.exit(0 if success else 1)