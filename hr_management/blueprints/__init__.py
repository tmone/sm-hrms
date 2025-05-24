# Flask blueprints initialization
from .auth import auth_bp
from .dashboard import dashboard_bp
from .employees import employees_bp
from .api import api_bp
from .face_recognition import face_recognition_bp
from .videos import videos_bp
from .gpu_management import gpu_management_bp

__all__ = [
    'auth_bp',
    'dashboard_bp', 
    'employees_bp',
    'api_bp',
    'face_recognition_bp',
    'videos_bp',
    'gpu_management_bp'
]