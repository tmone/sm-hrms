"""
Manual fix for video annotated path - run this with flask shell
"""

# Run these commands in flask shell:
# flask shell
# Then paste:

from app import db, Video

# Get video ID 1
video = Video.query.get(1)
if video:
    print(f"Current annotated_video_path: {video.annotated_video_path}")
    
    # Set the correct annotated video path
    video.annotated_video_path = "3c63c24a-a120-43c3-a21a-a7fa6c84d9e9_TANG_TRET_84A_Tret_84A_Tret_20250512085459_20250512091458_472401_annotated_20250524_210010.mp4"
    
    # Also set processed_path if not set
    if not video.processed_path:
        video.processed_path = video.annotated_video_path
    
    db.session.commit()
    print(f"Updated annotated_video_path to: {video.annotated_video_path}")
else:
    print("Video ID 1 not found")