# **App Name**: StepMedia HRM

## Core Features:

- **Dashboard Overview**: Interactive dashboard displaying employee information and summary metrics.
- **Employee Directory**: View and manage employee details (name, department, contact info).
- **Attendance Tracking**: Record daily attendance with check-in/check-out times.
- **Reporting and Analytics**: Visualize attendance and leave data through charts and graphs.
- **Leave Request Management**: Submit and manage time off requests, and display approval status.

## Video Processing and Facial Recognition:

- **Video Upload System**: 
  - Allow users to upload video files in common formats (MP4, AVI, MOV)
  - Support for batch uploading and organizing videos in collections
  - Preview functionality for uploaded videos

- **Person Detection and Tracking**:
  - Implement automatic person detection using models like SAM, SAM2, or YOLO
  - Track and identify when a person appears and disappears in the video (frame ranges)
  - Assign unique identifier codes to each detected person (e.g., "PERSON-0001", "PERSON-0002")
  - Create an interactive index of people appearing in videos with timestamps

- **Video Navigation by Person**:
  - Allow clicking on a person identifier to automatically play video from when they first appear
  - Stop playback when the person no longer appears in the frame
  - Include thumbnail previews of each person detected

- **Face Extraction and Dataset Creation**:
  - Extract facial images from identified persons at 128x128 resolution
  - Organize extracted faces into YOLO-compatible dataset format
  - Support for reviewing and cleaning extracted facial data
  - Export functionality for the created datasets

- **Face Recognition Model Training**:
  - Interface for training facial recognition models using extracted datasets
  - Training progress monitoring and performance metrics
  - Model versioning and management
  - Export trained models for deployment

- **Recognition Application**:
  - Apply trained models to new videos for automatic person identification
  - Real-time recognition using device cameras (mobile or webcam)
  - Match detected faces to previously labeled PERSON-XXXX identifiers
  - Integration with employee directory for potential automatic attendance marking

## Style Guidelines:

- Primary color: Navy blue (#2E3192) for a professional and trustworthy feel.
- Secondary color: Light gray (#F5F5F5) for backgrounds and subtle UI elements.
- Accent: Teal (#008080) for interactive elements and highlights.
- Clean and modern sans-serif fonts for readability.
- Simple and consistent line icons for navigation and actions.
- Card-based layout with clear sections for different modules.
- Subtle transitions and animations for a smooth user experience.
- Light and Dark themes.
- Admin Dashboard layout
- Full responsive for desktop, tablet and mobile
- Multi language: English, Vietnamese

## Technical Stack:

### Frontend (Python Flask):
- **Framework**: Flask for web application framework
- **Template Engine**: Jinja2 for dynamic HTML rendering
- **Styling**: 
  - Tailwind CSS for responsive design
  - Bootstrap 5 for UI components
  - Custom CSS for theme implementation
- **JavaScript Libraries**:
  - Alpine.js for reactive UI components
  - Video.js for advanced video player controls
  - Chart.js for analytics visualization
  - Socket.io client for real-time updates
- **HTMX**: For seamless AJAX interactions without full page reloads
- **Internationalization**: Flask-Babel for multi-language support
- **Form Handling**: WTForms for form validation and rendering

### Backend (Python Flask + Pinokio):
- **Web Framework**: Flask with Blueprints for modular architecture
- **Database**: SQLite with SQLAlchemy ORM
  - Fast setup and minimal configuration
  - Single file database for easy backup and portability
  - Models for employees, videos, detected persons, face datasets, and models
- **File Storage**:
  - Disk-based storage for videos and extracted face images
  - Organized directory structure for easy access during training
  - Naming convention to connect database records with image files
- **Video Processing Pipeline**:
  - Celery for background task management
  - Redis for task queue and caching
  - Independent Python scripts for each processing stage
  - Parallel processing architecture for handling heavy video files
- **AI/ML Integration**:
  - Pinokio for AI model serving
  - SAM/SAM2 or YOLO for person detection
  - Custom facial recognition models
  - OpenCV for video processing
  - PyTorch/TensorFlow for model training
- **Real-time Communication**: Flask-SocketIO for WebSocket connections
- **Authentication**: Flask-Login with session management
- **API**: Flask-RESTful for API endpoints

## Key Python Libraries:

### Core Flask Stack:
```python
# Core framework
Flask==2.3.3
Flask-SQLAlchemy==3.0.5
Flask-Login==0.6.3
Flask-WTF==1.1.1
WTForms==3.0.1
Flask-Babel==3.1.0
Flask-SocketIO==5.3.6

# Background processing
Celery==5.3.2
Redis==4.6.0

# Video and image processing
OpenCV-python==4.8.1.78
Pillow==10.0.1
FFmpeg-python==0.2.0

# AI/ML libraries
torch==2.0.1
torchvision==0.15.2
ultralytics==8.0.196  # for YOLO
segment-anything==1.0  # for SAM

# Data handling
pandas==2.1.1
numpy==1.24.4
sqlite3  # built-in Python module
```

## Data Storage Schema:

### SQLAlchemy Models:
```python
# Employee model
class Employee(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    department = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone = db.Column(db.String(20))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Video processing models
class Video(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(500), nullable=False)
    duration = db.Column(db.Float)
    resolution = db.Column(db.String(20))
    status = db.Column(db.String(50), default='uploaded')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class DetectedPerson(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.Integer, db.ForeignKey('video.id'), nullable=False)
    person_code = db.Column(db.String(20), nullable=False)  # PERSON-XXXX
    start_frame = db.Column(db.Integer, nullable=False)
    end_frame = db.Column(db.Integer, nullable=False)
    confidence = db.Column(db.Float)
    employee_id = db.Column(db.Integer, db.ForeignKey('employee.id'))
```

### Disk Storage Organization:
- **/uploads/videos/**: Original uploaded video files
- **/processing/temp/**: Temporary files during processing
- **/datasets/faces/**: Extracted face images organized by person ID
  - Directory structure: `/datasets/faces/PERSON-XXXX/`
  - Images stored as `face_XXXX_frame_YYYY.jpg`
- **/datasets/yolo/**: YOLO-formatted datasets for training
- **/models/**: Trained model files and checkpoints
- **/static/**: CSS, JavaScript, and static assets
- **/templates/**: Jinja2 HTML templates

## Flask Application Structure:

```
stepmedia_hrm/
├── app.py                          # Main Flask application
├── config.py                       # Configuration settings
├── requirements.txt                # Python dependencies
├── celery_app.py                   # Celery configuration
├── models/                         # SQLAlchemy models
│   ├── __init__.py
│   ├── employee.py
│   ├── video.py
│   └── face_recognition.py
├── blueprints/                     # Flask blueprints
│   ├── __init__.py
│   ├── dashboard.py
│   ├── employees.py
│   ├── videos.py
│   ├── face_recognition.py
│   └── api.py
├── templates/                      # Jinja2 templates
│   ├── base.html
│   ├── dashboard/
│   ├── employees/
│   ├── videos/
│   └── face_recognition/
├── static/                         # Static files
│   ├── css/
│   ├── js/
│   └── images/
├── processing/                     # Background processing scripts
│   ├── __init__.py
│   ├── video_processor.py
│   ├── person_detector.py
│   ├── face_extractor.py
│   └── model_trainer.py
├── utils/                          # Utility functions
│   ├── __init__.py
│   ├── file_handler.py
│   └── ai_models.py
└── migrations/                     # Database migrations
```

## Processing Flow:

1. **Video Upload Handling** (Flask + Celery):
   ```python
   @app.route('/upload_video', methods=['POST'])
   def upload_video():
       # Handle file upload
       # Create database entry
       # Queue background processing task
       process_video.delay(video_id)
       return jsonify({'status': 'uploaded', 'video_id': video_id})
   ```

2. **Background Processing** (Celery Tasks):
   ```python
   @celery.task
   def process_video(video_id):
       # Parallel processing using multiprocessing
       # Update database with progress
       # Emit real-time updates via SocketIO
       return processing_result
   ```

3. **Real-time Updates** (Flask-SocketIO):
   ```python
   @socketio.on('connect')
   def handle_connect():
       # Client connection handling
       
   def emit_progress_update(video_id, progress):
       socketio.emit('processing_progress', {
           'video_id': video_id,
           'progress': progress
       })
   ```

## Template Structure (Jinja2):

### Base Template with Theme Support:
```html
<!-- base.html -->
<!DOCTYPE html>
<html data-theme="{{ session.get('theme', 'light') }}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}StepMedia HRM{% endblock %}</title>
    <link href="{{ url_for('static', filename='css/tailwind.min.css') }}" rel="stylesheet">
    <script src="{{ url_for('static', filename='js/alpine.min.js') }}" defer></script>
</head>
<body class="bg-gray-100 dark:bg-gray-900 text-gray-800 dark:text-white">
    {% include 'partials/navbar.html' %}
    {% include 'partials/sidebar.html' %}
    
    <main class="ml-64 p-6">
        {% block content %}{% endblock %}
    </main>
    
    {% block scripts %}{% endblock %}
</body>
</html>
```

## Integration Points:

- SQLAlchemy ORM for database operations
- Celery for background video processing
- Flask-SocketIO for real-time progress updates
- Pinokio integration for AI model serving
- RESTful API endpoints for mobile integration
- Export functionality to standard formats

## Deployment Considerations:

- Single Python environment for consistency
- Gunicorn for production WSGI server
- Redis for Celery message broker and caching
- GPU acceleration for AI processing
- Docker containerization option
- SQLite database backup strategies
- Disk space management for video storage

## Development Workflow:

1. **Setup Flask Application**:
   ```bash
   pip install -r requirements.txt
   flask db init
   flask db migrate
   flask db upgrade
   celery -A celery_app worker --loglevel=info
   redis-server
   flask run
   ```

2. **Template Development**: Create responsive HTML templates with Jinja2
3. **API Development**: Build RESTful endpoints using Flask-RESTful
4. **Background Processing**: Implement video processing with Celery
5. **AI Integration**: Connect with Pinokio for model serving
6. **Testing**: Unit and integration testing with pytest
