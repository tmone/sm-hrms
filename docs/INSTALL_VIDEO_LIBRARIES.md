# Video Processing Libraries Installation Guide

## 📦 Quick Installation

### Option 1: Install All Dependencies (Recommended)
```bash
pip install -r requirements.txt
```

### Option 2: Install Individual Libraries
```bash
# Core video processing
pip install moviepy>=1.0.3
pip install opencv-python>=4.8.0

# Supporting libraries  
pip install imageio>=2.31.1
pip install imageio-ffmpeg>=0.4.7
pip install pillow>=9.0.0
pip install numpy>=1.21.0
```

### Option 3: Via Web Interface
1. Go to your video management page
2. Click "Install Video Dependencies" button
3. Libraries will be installed in the background

## 🎯 What Gets Installed

### MoviePy
- **Purpose**: Easy video editing and conversion
- **Best for**: Simple format conversions, video editing
- **Supports**: Most common video formats including IMKH

### OpenCV
- **Purpose**: Computer vision and video processing
- **Best for**: Frame-by-frame processing, advanced video analysis
- **Supports**: Wide range of video formats

### ImageIO + FFmpeg
- **Purpose**: Backend video processing engine
- **Best for**: Fast, efficient video conversion
- **Supports**: Virtually all video formats

## 🔧 How It Works

### Automatic Format Detection
The system automatically detects video formats:
- ✅ **Standard formats** (MP4, WebM) → Play directly
- ⚠️ **Non-standard formats** (IMKH, AVI) → Convert automatically
- ❌ **Unsupported formats** → Show conversion options

### Conversion Process
1. **Upload**: Video is uploaded to the server
2. **Detection**: System checks if format is web-compatible
3. **Conversion**: If needed, converts to MP4 in background
4. **Playback**: Web-compatible video plays in browser

### Supported Input Formats
- IMKH (proprietary surveillance format)
- AVI, MOV, MKV, WMV, FLV
- MP4, WebM (already web-compatible)
- And many more...

### Output Format
- **Video**: H.264 encoded MP4
- **Audio**: AAC encoded
- **Optimization**: Web-optimized for streaming
- **Quality**: Configurable (low/medium/high)

## 🚀 Usage

### Converting Existing Videos
1. Go to video details page
2. Click "🔄 Convert to Web-Compatible MP4"
3. Wait for conversion to complete
4. Video will automatically play in browser

### Automatic Conversion on Upload
- Future uploads can be configured to auto-convert
- Non-compatible formats are detected and queued for conversion
- Progress is shown in real-time

### Manual Conversion
```bash
# Using the video processor directly
python utils/video_processor.py input_video.mp4

# Using the video converter
python utils/video_converter.py input_video.mp4 output.mp4 --quality high
```

## ⚡ Performance Tips

### For Large Videos (>1GB)
- Use "low" quality setting for faster conversion
- Consider splitting large files
- Ensure sufficient disk space (2x original file size)

### For Multiple Videos
- Process videos one at a time to avoid memory issues
- Use background processing (automatic in web interface)
- Monitor system resources

### Quality Settings
- **High**: Best quality, slower conversion, larger files
- **Medium**: Balanced quality and speed (recommended)
- **Low**: Fast conversion, smaller files, lower quality

## 🔍 Troubleshooting

### "No conversion libraries available"
- Install the required libraries: `pip install -r requirements.txt`
- Restart your Flask application

### "FFmpeg not found"
- Install FFmpeg from https://ffmpeg.org/download.html
- Ensure FFmpeg is in your system PATH

### "Conversion failed"
- Check video file is not corrupted
- Ensure sufficient disk space
- Try different quality settings
- Check application logs for detailed errors

### "Video still won't play"
- Verify conversion completed successfully
- Clear browser cache
- Try different browser
- Check network console for errors

## 📊 Library Comparison

| Library | Speed | Quality | Formats | Memory |
|---------|-------|---------|---------|--------|
| FFmpeg  | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| MoviePy | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| OpenCV  | ⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐ |

## 🎉 Success!

Once installed, your HRM system will:
- ✅ Automatically detect video formats
- ✅ Convert incompatible videos to MP4
- ✅ Display conversion progress in real-time
- ✅ Play videos directly in web browsers
- ✅ Handle IMKH and other proprietary formats

Your IMKH video will now convert automatically and play in the browser! 🎥