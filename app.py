import os
import sys

# Fix Unicode output on Windows before anything else
try:
    from utils.fix_unicode_output import fix_unicode_output
    fix_unicode_output()
except Exception as e:
    print(f"Warning: Could not fix Unicode output: {e}")

from flask import Flask

# Set up logging before anything else
try:
    from config_logging import setup_logging, get_logger
    progress_logger = setup_logging()
    logger = get_logger(__name__)
    logger.info("Logging system initialized")
except Exception as e:
    print(f"Failed to setup logging: {e}")
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Try to import optional dependencies
SOCKETIO_AVAILABLE = False
BABEL_AVAILABLE = False
CELERY_AVAILABLE = False

try:
    from flask_socketio import SocketIO
    SOCKETIO_AVAILABLE = True
except ImportError:
    logger.warning("Flask-SocketIO not available. Real-time features disabled.")
    SocketIO = None

try:
    from flask_babel import Babel
    BABEL_AVAILABLE = True
except ImportError:
    logger.warning("Flask-Babel not available. Multi-language support disabled.")
    Babel = None

try:
    import celery
    CELERY_AVAILABLE = True
except ImportError:
    logger.warning("Celery not available. Background processing disabled.")

# Core required imports
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from datetime import datetime, timedelta

# Initialize extensions
db = SQLAlchemy()
login_manager = LoginManager()
socketio = SocketIO() if SOCKETIO_AVAILABLE else None
babel = Babel() if BABEL_AVAILABLE else None

def create_app(config_name=None):
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    # Use instance folder for database
    instance_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance')
    if not os.path.exists(instance_path):
        os.makedirs(instance_path)
    db_path = os.path.join(instance_path, 'stepmedia_hrm.db')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', f'sqlite:///{db_path}')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Import database configuration for connection pool
    try:
        from config_database import SQLALCHEMY_ENGINE_OPTIONS
        app.config['SQLALCHEMY_ENGINE_OPTIONS'] = SQLALCHEMY_ENGINE_OPTIONS
        logger.info("Using enhanced database connection pool settings")
    except ImportError:
        # Default connection pool settings
        app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
            'pool_size': 10,
            'max_overflow': 20,
            'pool_timeout': 45,
            'pool_recycle': 3600,
            'pool_pre_ping': True,
        }
        logger.info("Using default database connection pool settings")
    app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB file upload limit
    
    # Optional configuration
    app.config['UPLOAD_FOLDER'] = 'static/uploads'
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)
    
    # Development mode
    if os.environ.get('FLASK_ENV') == 'development':
        app.config['DEBUG'] = True
    
    # Initialize core extensions
    db.init_app(app)
    login_manager.init_app(app)
    
    # Initialize optional extensions
    if SOCKETIO_AVAILABLE and socketio:
        socketio.init_app(app, cors_allowed_origins="*")
    
    if BABEL_AVAILABLE and babel:
        babel.init_app(app)
    
    # Configure login manager
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access this page.'
    
    # Create directories if they don't exist
    directories = [
        'uploads',
        'processing/temp',
        'datasets/faces',
        'datasets/yolo',
        'models',
        'static/css',
        'static/js',
        'static/images'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Define models directly here to avoid import issues
    from flask_login import UserMixin
    
    class Employee(db.Model, UserMixin):
        __tablename__ = 'employees'
        
        id = db.Column(db.Integer, primary_key=True)
        name = db.Column(db.String(100), nullable=False)
        department = db.Column(db.String(50), nullable=False)
        email = db.Column(db.String(120), unique=True, nullable=False, index=True)
        phone = db.Column(db.String(20))
        position = db.Column(db.String(100))
        employee_id = db.Column(db.String(20), unique=True)
        hire_date = db.Column(db.Date)
        status = db.Column(db.String(20), default='active')
        created_at = db.Column(db.DateTime, default=datetime.utcnow)
        updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        
        def __repr__(self):
            return f'<Employee {self.name}>'
        
        def to_dict(self):
            return {
                'id': self.id,
                'name': self.name,
                'department': self.department,
                'email': self.email,
                'phone': self.phone,
                'position': self.position,
                'employee_id': self.employee_id,
                'hire_date': self.hire_date.isoformat() if self.hire_date else None,
                'status': self.status,
                'created_at': self.created_at.isoformat()
            }

    class AttendanceRecord(db.Model):
        __tablename__ = 'attendance_records'
        
        id = db.Column(db.Integer, primary_key=True)
        employee_id = db.Column(db.Integer, db.ForeignKey('employees.id'), nullable=False)
        date = db.Column(db.Date, nullable=False)
        check_in_time = db.Column(db.DateTime)
        check_out_time = db.Column(db.DateTime)
        status = db.Column(db.String(20), default='present')
        notes = db.Column(db.Text)
        created_at = db.Column(db.DateTime, default=datetime.utcnow)
        
        employee = db.relationship('Employee', backref='attendance_records')
        
        __table_args__ = (db.UniqueConstraint('employee_id', 'date', name='unique_employee_date'),)
        
        def __repr__(self):
            return f'<AttendanceRecord {self.employee.name} - {self.date}>'
        
        def to_dict(self):
            return {
                'id': self.id,
                'employee_id': self.employee_id,
                'date': self.date.isoformat(),
                'check_in_time': self.check_in_time.isoformat() if self.check_in_time else None,
                'check_out_time': self.check_out_time.isoformat() if self.check_out_time else None,
                'status': self.status,
                'notes': self.notes
            }
    
    # Video processing models (optional feature)
    class Video(db.Model):
        __tablename__ = 'videos'
        
        id = db.Column(db.Integer, primary_key=True)
        filename = db.Column(db.String(255), nullable=False)
        title = db.Column(db.String(200))
        description = db.Column(db.Text)
        file_path = db.Column(db.String(500), nullable=False)
        file_size = db.Column(db.BigInteger)
        duration = db.Column(db.Float)
        resolution = db.Column(db.String(20))
        fps = db.Column(db.Float)
        codec = db.Column(db.String(50))
        
        # Processing related fields
        status = db.Column(db.String(20), default='uploaded', index=True)
        priority = db.Column(db.String(20), default='normal')
        processing_started_at = db.Column(db.DateTime)
        processing_completed_at = db.Column(db.DateTime)
        processed_path = db.Column(db.String(500))
        annotated_video_path = db.Column(db.String(500))  # Enhanced detection annotated video
        processing_log = db.Column(db.Text)
        error_message = db.Column(db.Text)
        task_id = db.Column(db.String(100))  # Celery task ID
        processing_progress = db.Column(db.Integer, default=0)
        
        # Detection statistics
        person_count = db.Column(db.Integer, default=0)
        frame_count = db.Column(db.Integer)
        processed_frames = db.Column(db.Integer)
        
        # OCR extracted fields
        ocr_location = db.Column(db.String(100))
        ocr_video_date = db.Column(db.Date)
        ocr_video_time = db.Column(db.Time)
        ocr_extraction_done = db.Column(db.Boolean, default=False)
        ocr_extraction_confidence = db.Column(db.Float)
        
        # Video chunking fields
        parent_video_id = db.Column(db.Integer, db.ForeignKey('videos.id'))
        chunk_index = db.Column(db.Integer)
        total_chunks = db.Column(db.Integer)
        is_chunk = db.Column(db.Boolean, default=False)
        
        # Employee relationship (uploader)
        employee_id = db.Column(db.Integer, db.ForeignKey('employees.id'))
        
        # Metadata
        created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
        updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        
        def __repr__(self):
            return f'<Video {self.filename}>'
        
        def to_dict(self):
            return {
                'id': self.id,
                'filename': self.filename,
                'title': self.title,
                'description': self.description,
                'file_size': self.file_size,
                'duration': self.duration,
                'resolution': self.resolution,
                'fps': self.fps,
                'codec': self.codec,
                'status': self.status,
                'priority': self.priority,
                'processing_started_at': self.processing_started_at.isoformat() if self.processing_started_at else None,
                'processing_completed_at': self.processing_completed_at.isoformat() if self.processing_completed_at else None,
                'processed_path': self.processed_path,
                'annotated_video_path': self.annotated_video_path,
                'person_count': self.person_count,
                'frame_count': self.frame_count,
                'processed_frames': self.processed_frames,
                'created_at': self.created_at.isoformat(),
                'updated_at': self.updated_at.isoformat()
            }

    class DetectedPerson(db.Model):
        __tablename__ = 'detected_persons'
        
        id = db.Column(db.Integer, primary_key=True)
        video_id = db.Column(db.Integer, db.ForeignKey('videos.id'), nullable=False)
        employee_id = db.Column(db.Integer, db.ForeignKey('employees.id'))
        
        # Detection data
        timestamp = db.Column(db.Float, nullable=False)
        frame_number = db.Column(db.Integer)
        confidence = db.Column(db.Float, default=0.0)
        
        # Person tracking (for multi-frame tracking)
        person_id = db.Column(db.String(50))  # e.g., "PERSON-0001"
        track_id = db.Column(db.Integer)      # Internal tracking ID
        
        # Bounding box coordinates
        bbox_x = db.Column(db.Integer)
        bbox_y = db.Column(db.Integer)
        bbox_width = db.Column(db.Integer)
        bbox_height = db.Column(db.Integer)
        
        # Recognition data
        is_identified = db.Column(db.Boolean, default=False)
        manual_identification = db.Column(db.Boolean, default=False)
        
        # Face encoding (for face recognition)
        face_encoding = db.Column(db.Text)
        
        # OCR-based attendance fields
        attendance_date = db.Column(db.Date)
        attendance_time = db.Column(db.Time)
        attendance_location = db.Column(db.String(100))
        check_in_time = db.Column(db.DateTime)
        check_out_time = db.Column(db.DateTime)
        
        created_at = db.Column(db.DateTime, default=datetime.utcnow)
        
        # Relationships
        video = db.relationship('Video', backref='detected_persons')
        employee = db.relationship('Employee', backref='detections')
        
        def __repr__(self):
            return f'<DetectedPerson {self.id}>'
        
        def to_dict(self):
            return {
                'id': self.id,
                'video_id': self.video_id,
                'employee_id': self.employee_id,
                'timestamp': self.timestamp,
                'frame_number': self.frame_number,
                'confidence': self.confidence,
                'person_id': self.person_id,
                'track_id': self.track_id,
                'bbox_x': self.bbox_x,
                'bbox_y': self.bbox_y,
                'bbox_width': self.bbox_width,
                'bbox_height': self.bbox_height,
                'is_identified': self.is_identified,
                'manual_identification': self.manual_identification,
                'created_at': self.created_at.isoformat(),
                'employee_name': self.employee.name if self.employee else None
            }

    # Face recognition models (optional feature)
    class FaceDataset(db.Model):
        __tablename__ = 'face_datasets'
        
        id = db.Column(db.Integer, primary_key=True)
        name = db.Column(db.String(100), nullable=False)
        description = db.Column(db.Text)
        dataset_path = db.Column(db.String(500), nullable=False)
        format_type = db.Column(db.String(20), default='yolo')
        
        # Statistics
        person_count = db.Column(db.Integer, default=0)
        image_count = db.Column(db.Integer, default=0)
        
        # Status
        status = db.Column(db.String(20), default='created')
        
        created_at = db.Column(db.DateTime, default=datetime.utcnow)
        updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        
        def __repr__(self):
            return f'<FaceDataset {self.name}>'
        
        def to_dict(self):
            return {
                'id': self.id,
                'name': self.name,
                'description': self.description,
                'dataset_path': self.dataset_path,
                'format_type': self.format_type,
                'person_count': self.person_count,
                'image_count': self.image_count,
                'status': self.status,
                'created_at': self.created_at.isoformat(),
                'updated_at': self.updated_at.isoformat()
            }

    class TrainedModel(db.Model):
        __tablename__ = 'trained_models'
        
        id = db.Column(db.Integer, primary_key=True)
        name = db.Column(db.String(100), nullable=False)
        description = db.Column(db.Text)
        model_type = db.Column(db.String(50), default='face_recognition')
        dataset_id = db.Column(db.Integer, db.ForeignKey('face_datasets.id'))
        
        # Model configuration
        model_path = db.Column(db.String(500))
        version = db.Column(db.String(20), default='1.0.0')
        framework = db.Column(db.String(50), default='sklearn')
        
        # Training parameters
        epochs = db.Column(db.Integer, default=10)
        batch_size = db.Column(db.Integer, default=32)
        learning_rate = db.Column(db.Float, default=0.001)
        
        # Performance metrics
        accuracy = db.Column(db.Float)
        validation_accuracy = db.Column(db.Float)
        loss = db.Column(db.Float)
        validation_loss = db.Column(db.Float)
        
        # Status and deployment
        status = db.Column(db.String(20), default='created')
        is_active = db.Column(db.Boolean, default=False)
        deployed_at = db.Column(db.DateTime)
        
        created_at = db.Column(db.DateTime, default=datetime.utcnow)
        updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        
        # Relationship
        dataset = db.relationship('FaceDataset', backref='models')
        
        def __repr__(self):
            return f'<TrainedModel {self.name}>'
        
        def to_dict(self):
            return {
                'id': self.id,
                'name': self.name,
                'description': self.description,
                'model_type': self.model_type,
                'dataset_id': self.dataset_id,
                'version': self.version,
                'framework': self.framework,
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'accuracy': self.accuracy,
                'validation_accuracy': self.validation_accuracy,
                'loss': self.loss,
                'validation_loss': self.validation_loss,
                'status': self.status,
                'is_active': self.is_active,
                'deployed_at': self.deployed_at.isoformat() if self.deployed_at else None,
                'created_at': self.created_at.isoformat(),
                'updated_at': self.updated_at.isoformat()
            }

    class RecognitionResult(db.Model):
        __tablename__ = 'recognition_results'
        
        id = db.Column(db.Integer, primary_key=True)
        video_id = db.Column(db.Integer, db.ForeignKey('videos.id'), nullable=False)
        model_id = db.Column(db.Integer, db.ForeignKey('trained_models.id'))
        detected_person_id = db.Column(db.Integer, db.ForeignKey('detected_persons.id'))
        employee_id = db.Column(db.Integer, db.ForeignKey('employees.id'))
        
        # Recognition results
        confidence = db.Column(db.Float, default=0.0)
        is_verified = db.Column(db.Boolean, default=False)
        
        created_at = db.Column(db.DateTime, default=datetime.utcnow)
        
        # Relationships
        video = db.relationship('Video', backref='recognition_results')
        model = db.relationship('TrainedModel', backref='recognition_results')
        detected_person = db.relationship('DetectedPerson', backref='recognition_results')
        employee = db.relationship('Employee', backref='recognition_results')
        
        def __repr__(self):
            return f'<RecognitionResult {self.id}>'
        
        def to_dict(self):
            return {
                'id': self.id,
                'video_id': self.video_id,
                'model_id': self.model_id,
                'detected_person_id': self.detected_person_id,
                'employee_id': self.employee_id,
                'confidence': self.confidence,
                'is_verified': self.is_verified,
                'created_at': self.created_at.isoformat(),
                'employee_name': self.employee.name if self.employee else None,
                'model_name': self.model.name if self.model else None
            }

    # System Settings Model
    class SystemSettings(db.Model):
        """System-wide configuration settings"""
        __tablename__ = 'system_settings'
        
        id = db.Column(db.Integer, primary_key=True)
        key = db.Column(db.String(100), unique=True, nullable=False)
        value = db.Column(db.Text)
        value_type = db.Column(db.String(20), default='string')
        category = db.Column(db.String(50), default='general')
        description = db.Column(db.Text)
        is_sensitive = db.Column(db.Boolean, default=False)
        created_at = db.Column(db.DateTime, default=datetime.utcnow)
        updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        
        # Default settings
        DEFAULT_SETTINGS = {
            'general': {
                'app_name': {'value': 'StepMedia HRM', 'type': 'string', 'description': 'Application name'},
                'timezone': {'value': 'Asia/Bangkok', 'type': 'string', 'description': 'Default timezone'},
                'language': {'value': 'en', 'type': 'string', 'description': 'Default language'}
            },
            'date_time': {
                'date_format': {'value': 'DD-MM-YYYY', 'type': 'string', 'description': 'Date format for display'},
                'time_format': {'value': '24h', 'type': 'string', 'description': 'Time format (12h or 24h)'},
                'ocr_date_format': {'value': 'DD-MM-YYYY', 'type': 'string', 'description': 'Expected date format in OCR'}
            },
            'video_processing': {
                'max_upload_size_mb': {'value': '2048', 'type': 'integer', 'description': 'Max video size in MB'},
                'auto_process_on_upload': {'value': 'true', 'type': 'boolean', 'description': 'Auto process videos'},
                'default_fps': {'value': '30', 'type': 'integer', 'description': 'Default FPS'},
                'ocr_sample_interval': {'value': '10', 'type': 'integer', 'description': 'OCR sampling interval (seconds)'}
            },
            'attendance': {
                'work_start_time': {'value': '08:00', 'type': 'string', 'description': 'Work start time'},
                'work_end_time': {'value': '17:00', 'type': 'string', 'description': 'Work end time'},
                'late_threshold_minutes': {'value': '15', 'type': 'integer', 'description': 'Late threshold (minutes)'},
                'minimum_presence_seconds': {'value': '5', 'type': 'integer', 'description': 'Min presence time'}
            }
        }
        
        def get_typed_value(self):
            """Get value converted to its proper type"""
            if self.value_type == 'integer':
                return int(self.value) if self.value else 0
            elif self.value_type == 'boolean':
                return self.value.lower() in ('true', '1', 'yes', 'on')
            else:
                return self.value
        
        @classmethod
        def get_setting(cls, key, default=None):
            """Get a setting value by key"""
            setting = cls.query.filter_by(key=key).first()
            return setting.get_typed_value() if setting else default
        
        @classmethod
        def set_setting(cls, key, value, value_type='string', category='general', description=None):
            """Set a setting value"""
            setting = cls.query.filter_by(key=key).first()
            if not setting:
                setting = cls(key=key, value_type=value_type, category=category, description=description)
                db.session.add(setting)
            
            if value_type == 'boolean':
                setting.value = 'true' if value else 'false'
            else:
                setting.value = str(value)
            
            setting.value_type = value_type
            if category:
                setting.category = category
            if description:
                setting.description = description
            
            db.session.commit()
            return setting
        
        @classmethod
        def initialize_defaults(cls):
            """Initialize default settings if they don't exist"""
            for category, settings in cls.DEFAULT_SETTINGS.items():
                for key, config in settings.items():
                    existing = cls.query.filter_by(key=key).first()
                    if not existing:
                        cls.set_setting(key=key, value=config['value'], value_type=config['type'],
                                      category=category, description=config['description'])
            db.session.commit()
        
        @classmethod
        def get_all_by_category(cls):
            """Get all settings grouped by category"""
            settings = cls.query.order_by(cls.category, cls.key).all()
            grouped = {}
            for setting in settings:
                if setting.category not in grouped:
                    grouped[setting.category] = []
                grouped[setting.category].append(setting)
            return grouped
    
    # Store model and db references in app for blueprints to access
    app.db = db
    app.Employee = Employee
    app.AttendanceRecord = AttendanceRecord
    app.Video = Video
    app.DetectedPerson = DetectedPerson
    app.FaceDataset = FaceDataset
    app.TrainedModel = TrainedModel
    app.RecognitionResult = RecognitionResult
    app.SystemSettings = SystemSettings
      # Set up user loader now that Employee model is defined
    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(Employee, int(user_id))
    
    # Initialize database and create tables
    with app.app_context():
        db.create_all()
        
        # Initialize default settings
        SystemSettings.initialize_defaults()
        print("[OK] System settings initialized")
        
        # Create demo admin user if it doesn't exist
        admin = Employee.query.filter_by(email='admin@stepmedia.com').first()
        if not admin:
            admin = Employee(
                name='Admin User',
                email='admin@stepmedia.com',
                department='IT',
                position='System Administrator',
                employee_id='EMP001',
                status='active'
            )
            db.session.add(admin)
            db.session.commit()
            print("[OK] Demo admin user created: admin@stepmedia.com")
      # Add a simple test route
    @app.route('/test')
    def test():
        return "<h1>[OK] App is working!</h1><p><a href='/auth/login'>Go to Login</a></p>"
    
    # Add static file serving for uploads
    @app.route('/static/uploads/<path:filename>')
    def uploaded_file(filename):
        from flask import send_from_directory
        upload_folder = app.config.get('UPLOAD_FOLDER', 'static/uploads')
        return send_from_directory(upload_folder, filename)
    
    # Add static file serving for processing outputs (detected videos)
    @app.route('/processing/outputs/<path:filename>')
    def processing_output_file(filename):
        from flask import send_from_directory
        import os
        outputs_dir = os.path.join('processing', 'outputs')
        return send_from_directory(outputs_dir, filename)
      # Register blueprints
    blueprint_errors = []
    
    # Essential blueprints (required for basic functionality)
    try:
        from hr_management.blueprints.auth import auth_bp
        app.register_blueprint(auth_bp, url_prefix='/auth')
        print("[OK] Auth blueprint registered")
    except ImportError as e:
        blueprint_errors.append(f"Auth blueprint: {e}")
        
    try:
        from hr_management.blueprints.dashboard import dashboard_bp
        app.register_blueprint(dashboard_bp, url_prefix='/')
        print("[OK] Dashboard blueprint registered")
    except ImportError as e:
        blueprint_errors.append(f"Dashboard blueprint: {e}")
        
    # Optional blueprints
    optional_blueprints = [
        ('hr_management.blueprints.employees', 'employees_bp', '/employees', 'Employees'),
        ('hr_management.blueprints.videos', 'videos_bp', '/videos', 'Videos'),
        ('hr_management.blueprints.face_recognition', 'face_recognition_bp', '/face-recognition', 'Face Recognition'),
        ('hr_management.blueprints.api', 'api_bp', '/api', 'API'),
        ('hr_management.blueprints.persons', 'persons_bp', '/persons', 'Persons'),
        ('hr_management.blueprints.person_review', 'person_review_bp', '/persons/review', 'Person Review'),
        ('hr_management.blueprints.attendance', 'attendance_bp', '/attendance', 'Attendance'),
        ('hr_management.blueprints.settings', 'settings_bp', '/settings', 'Settings'),
        ('hr_management.blueprints.person_recognition', 'person_recognition_bp', '/person-recognition', 'Person Recognition'),
    ]
    
    for module_name, blueprint_name, url_prefix, display_name in optional_blueprints:
        try:
            module = __import__(module_name, fromlist=[blueprint_name])
            blueprint = getattr(module, blueprint_name)
            app.register_blueprint(blueprint, url_prefix=url_prefix)
            print(f"[OK] {display_name} blueprint registered")
        except ImportError as e:
            blueprint_errors.append(f"{display_name} blueprint: {e}")
        except AttributeError as e:
            blueprint_errors.append(f"{display_name} blueprint: {e}")
      # GPU management blueprint (special import)
    try:
        from hr_management.blueprints import gpu_management_bp
        if gpu_management_bp is not None:
            app.register_blueprint(gpu_management_bp, url_prefix='/gpu')
            print("[OK] GPU Management blueprint registered")
        else:
            blueprint_errors.append("GPU Management blueprint: Dependencies not available")
    except ImportError as e:
        blueprint_errors.append(f"GPU Management blueprint: {e}")
    
    if blueprint_errors:
        print("[WARNING]  Some blueprints failed to load:")
        for error in blueprint_errors:
            print(f"   - {error}")
        print("The application will run with limited functionality")
    else:
        print("[OK] All blueprints registered successfully")
    
    return app

def main():
    """Main application entry point with error handling"""
    print("Starting StepMedia HRM...")
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required")
        sys.exit(1)
    
    print(f"Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create Flask application
    try:
        app = create_app()
        print("Application created successfully")
    except Exception as e:
        print(f"Error: Failed to create application: {e}")
        sys.exit(1)
    
    # Print status
    print("\n" + "="*50)
    print("StepMedia HRM is ready!")
    print("Access: http://localhost:5000")
    print("Demo Login: Click 'Demo Login' button")
    print("Admin Email: admin@stepmedia.com")
    
    # Print feature status
    features = [
        ("Core HRM", True),
        ("Real-time Updates", SOCKETIO_AVAILABLE),
        ("Multi-language", BABEL_AVAILABLE),
        ("Background Processing", CELERY_AVAILABLE)
    ]
    
    print("\nAvailable Features:")
    for feature, available in features:
        status = "[OK]" if available else "[WARN]"
        print(f"   {status} {feature}")
    
    print("="*50)
    
    # OCR saving is now integrated into the processing workflow
    # No patch needed - OCR data is saved automatically during processing
    
    # Start chunk merge monitor
    try:
        from processing.chunk_merge_monitor import get_chunk_monitor
        monitor = get_chunk_monitor(app)
        print("[OK] Chunk merge monitor started")
    except Exception as e:
        print(f"[WARNING] Could not start chunk merge monitor: {e}")
        
    # Start scheduled cleanup service
    try:
        from processing.scheduled_cleanup import start_scheduled_cleanup
        cleanup_service = start_scheduled_cleanup(cleanup_interval_hours=6)  # Run every 6 hours
        print("[OK] Scheduled cleanup service started (runs every 6 hours)")
        
        # Run immediate cleanup on startup to clean any leftover files
        cleanup_service.run_immediate()
    except Exception as e:
        print(f"[WARNING] Could not start scheduled cleanup: {e}")
    
    # Start the application
    try:
        if SOCKETIO_AVAILABLE and socketio:
            print("Starting with WebSocket support...")
            socketio.run(app, debug=True, host='0.0.0.0', port=5001, allow_unsafe_werkzeug=True)
        else:
            print("Starting in standard mode...")
            app.run(debug=True, host='0.0.0.0', port=5001)
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"\nApplication error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()