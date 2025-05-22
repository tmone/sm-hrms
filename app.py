import os
import sys
from flask import Flask

# Try to import optional dependencies
SOCKETIO_AVAILABLE = False
BABEL_AVAILABLE = False
CELERY_AVAILABLE = False

try:
    from flask_socketio import SocketIO
    SOCKETIO_AVAILABLE = True
except ImportError:
    print("⚠️  Flask-SocketIO not available. Real-time features disabled.")
    SocketIO = None

try:
    from flask_babel import Babel
    BABEL_AVAILABLE = True
except ImportError:
    print("⚠️  Flask-Babel not available. Multi-language support disabled.")
    Babel = None

try:
    import celery
    CELERY_AVAILABLE = True
except ImportError:
    print("⚠️  Celery not available. Background processing disabled.")

# Core required imports
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from datetime import timedelta

# Initialize extensions
db = SQLAlchemy()
login_manager = LoginManager()
socketio = SocketIO() if SOCKETIO_AVAILABLE else None
babel = Babel() if BABEL_AVAILABLE else None

def create_app(config_name=None):
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///stepmedia_hrm.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Optional configuration
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
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
    
    @login_manager.user_loader
    def load_user(user_id):
        from models.employee import Employee
        return Employee.query.get(int(user_id))
    
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
    
    # Initialize models with database
    with app.app_context():
        # Create models with the database instance
        from models.employee import create_employee_models
        Employee, AttendanceRecord = create_employee_models(db)
        
        # Store model references in app for blueprints to access
        app.Employee = Employee
        app.AttendanceRecord = AttendanceRecord
        
        # Create database tables
        db.create_all()
        
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
            print("✅ Demo admin user created: admin@stepmedia.com")
    
    # Register blueprints
    try:
        from blueprints.dashboard import dashboard_bp
        from blueprints.employees import employees_bp  
        from blueprints.videos import videos_bp
        from blueprints.face_recognition import face_recognition_bp
        from blueprints.auth import auth_bp
        from blueprints.api import api_bp
        
        app.register_blueprint(dashboard_bp, url_prefix='/')
        app.register_blueprint(employees_bp, url_prefix='/employees')
        app.register_blueprint(videos_bp, url_prefix='/videos')
        app.register_blueprint(face_recognition_bp, url_prefix='/face-recognition')
        app.register_blueprint(auth_bp, url_prefix='/auth')
        app.register_blueprint(api_bp, url_prefix='/api')
        
        print("✅ All blueprints registered successfully")
        
    except ImportError as e:
        print(f"⚠️  Some blueprints failed to load: {e}")
        print("The application will run with limited functionality")
    
    return app

def main():
    """Main application entry point with error handling"""
    print("🚀 Starting StepMedia HRM...")
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create Flask application
    try:
        app = create_app()
        print("✅ Application created successfully")
    except Exception as e:
        print(f"❌ Failed to create application: {e}")
        sys.exit(1)
    
    # Print status
    print("\n" + "="*50)
    print("📱 StepMedia HRM is ready!")
    print("🌐 Access: http://localhost:5000")
    print("🔑 Demo Login: Click 'Demo Login' button")
    print("👤 Admin Email: admin@stepmedia.com")
    
    # Print feature status
    features = [
        ("Core HRM", True),
        ("Real-time Updates", SOCKETIO_AVAILABLE),
        ("Multi-language", BABEL_AVAILABLE),
        ("Background Processing", CELERY_AVAILABLE)
    ]
    
    print("\n📋 Available Features:")
    for feature, available in features:
        status = "✅" if available else "⚠️ "
        print(f"   {status} {feature}")
    
    print("="*50)
    
    # Start the application
    try:
        if SOCKETIO_AVAILABLE and socketio:
            print("🔌 Starting with WebSocket support...")
            socketio.run(app, debug=True, host='0.0.0.0', port=5000)
        else:
            print("🌐 Starting in standard mode...")
            app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Application error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()