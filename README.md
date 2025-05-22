# StepMedia HRM - AI-Powered Human Resource Management System

A modern Flask-based HRM system with advanced video processing and facial recognition capabilities.

## Features

### Core HRM Features
- **Employee Directory**: Comprehensive employee management with profiles, departments, and contact information
- **Attendance Tracking**: Digital attendance system with check-in/check-out functionality
- **Dashboard Analytics**: Real-time insights and reporting on HR metrics
- **Multi-language Support**: English and Vietnamese language options

### AI & Video Processing
- **Video Upload System**: Support for MP4, AVI, MOV, MKV, and WEBM formats
- **Person Detection**: Automatic person detection using YOLO/SAM models
- **Face Recognition**: Train custom models for employee identification
- **Real-time Processing**: Background video processing with live progress updates
- **Face Dataset Management**: Extract and organize facial data for training

### Technical Features
- **Modern UI**: Responsive design with light/dark themes
- **Real-time Updates**: WebSocket integration for live notifications
- **RESTful API**: Complete API for external integrations
- **Background Processing**: Celery-based task queue for heavy operations
- **Database Management**: SQLite with SQLAlchemy ORM

## Quick Start

### Prerequisites
- Python 3.8+
- Redis (for Celery)
- FFmpeg (for video processing)

### Installation

#### Quick Start (Recommended)
```bash
# Install core dependencies
python3 -m pip install Flask Flask-SQLAlchemy Flask-Login Flask-WTF

# Run the application
python3 app.py
```

#### Full Installation
```bash
# Install all dependencies (some optional features may fail gracefully)
python3 -m pip install -r requirements.txt

# Run the application
python3 app.py

# Optional: Start Redis for advanced features (if needed)
redis-server                                    # In separate terminal
```

#### Access the Application
- Open http://localhost:5000
- Click "Demo Login" to access the system with sample data

> **Note**: The application has smart dependency detection. It will work with minimal Flask installation and progressively enable features as more packages are available.

## Project Structure

**Clean Python project with single entry point:**

```
stepmedia_hrm/
├── app.py                          # Main Flask application (single entry point, includes config)
├── requirements.txt                # Python dependencies (one file only)
├── models/                         # Database models
│   ├── employee.py                 # Employee and attendance models
│   ├── video.py                    # Video and detection models
│   └── face_recognition.py         # AI model management
├── blueprints/                     # Flask blueprints
│   ├── auth.py                     # Authentication
│   ├── dashboard.py                # Main dashboard
│   ├── employees.py                # Employee management
│   ├── videos.py                   # Video processing
│   ├── face_recognition.py         # AI model management
│   └── api.py                      # REST API endpoints
├── templates/                      # Jinja2 templates
│   ├── base.html                   # Base template
│   ├── partials/                   # Reusable components
│   ├── auth/                       # Authentication pages
│   ├── dashboard/                  # Dashboard templates
│   ├── employees/                  # Employee management
│   ├── videos/                     # Video processing UI
│   └── face_recognition/           # AI model interfaces
├── static/                         # Static assets
│   ├── css/custom.css              # Custom styling
│   └── js/app.js                   # JavaScript application
├── processing/                     # Background processing
│   └── tasks.py                    # Celery tasks
├── uploads/                        # File uploads
├── datasets/                       # AI training data
└── models/                         # Trained AI models
```

## Configuration

### Environment Variables
Create a `.env` file with:
```bash
SECRET_KEY=your-secret-key-here
DATABASE_URL=sqlite:///stepmedia_hrm.db
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
```

### File Upload Settings
- Maximum file size: 500MB
- Supported video formats: MP4, AVI, MOV, MKV, WEBM
- Face image resolution: 128x128 pixels

## API Endpoints

### Authentication
- `POST /auth/login` - User login
- `GET /auth/logout` - User logout

### Employees
- `GET /api/employees` - List employees
- `GET /employees/<id>` - Employee details
- `POST /employees/create` - Create employee
- `PUT /employees/<id>/edit` - Update employee

### Videos
- `GET /api/videos` - List videos
- `POST /videos/api/upload` - Upload video
- `GET /videos/<id>` - Video details
- `POST /videos/<id>/reprocess` - Reprocess video

### Face Recognition
- `GET /api/models` - List AI models
- `GET /api/datasets` - List face datasets
- `POST /face-recognition/recognition/video/<id>` - Run recognition

### System
- `GET /api/stats` - Dashboard statistics
- `GET /api/system/health` - System health check

## Video Processing Pipeline

1. **Upload**: Videos uploaded via web interface or API
2. **Metadata Extraction**: Duration, FPS, resolution analysis
3. **Person Detection**: YOLO/SAM-based person detection
4. **Face Extraction**: Extract faces at 128x128 resolution
5. **Dataset Creation**: Organize faces for training
6. **Model Training**: Train custom recognition models
7. **Recognition**: Apply trained models to new videos

## AI Model Integration

### Supported Models
- **YOLO**: Person detection in videos
- **SAM/SAM2**: Segmentation and tracking
- **Custom Face Recognition**: Employee identification

### Training Workflow
1. Upload and process videos
2. Extract faces from detected persons
3. Create labeled datasets
4. Train recognition models
5. Deploy models for production use

## Development

### Running in Development Mode
```bash
export FLASK_ENV=development
python app.py
```

### Database Migrations
```bash
# Create migration
flask db migrate -m "Description"

# Apply migration
flask db upgrade
```

### Adding New Features
1. Create database models in `models/`
2. Add blueprints in `blueprints/`
3. Create templates in `templates/`
4. Add API endpoints in `blueprints/api.py`
5. Update JavaScript in `static/js/app.js`

## Deployment

### Production Setup
1. Use Gunicorn WSGI server
2. Setup Redis for production
3. Configure proper database (PostgreSQL recommended)
4. Setup reverse proxy (Nginx)
5. Configure SSL certificates

### Docker Deployment
```bash
# Build image
docker build -t stepmedia-hrm .

# Run with docker-compose
docker-compose up -d
```

## Security Considerations

- File upload validation and sanitization
- SQL injection protection via SQLAlchemy
- CSRF protection with Flask-WTF
- Session security with secure cookies
- Input validation and sanitization

## Performance Optimization

- Background video processing with Celery
- Database indexing for large datasets
- Caching with Redis
- Optimized video processing pipelines
- Progressive image loading

## Troubleshooting

### Common Issues

1. **Video processing fails**
   - Check FFmpeg installation
   - Verify file permissions
   - Check Celery worker status

2. **Face recognition not working**
   - Ensure models are trained
   - Check dataset quality
   - Verify GPU availability

3. **Real-time updates not working**
   - Check Socket.IO connection
   - Verify Redis connectivity
   - Check browser WebSocket support

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the troubleshooting guide

---

**StepMedia HRM** - Transforming HR management with AI-powered video analysis and face recognition technology.