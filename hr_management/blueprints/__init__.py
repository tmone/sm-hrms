# Flask blueprints initialization
# Only import blueprints that don't require external dependencies by default
from .auth import auth_bp
from .dashboard import dashboard_bp
from .employees import employees_bp
from .api import api_bp
from .videos import videos_bp

# Try to import blueprints that may have dependencies
try:
    from .face_recognition import face_recognition_bp
except ImportError:
    face_recognition_bp = None

try:
    from .gpu_management import gpu_management_bp
except ImportError:
    gpu_management_bp = None

__all__ = [
    'auth_bp',
    'dashboard_bp', 
    'employees_bp',
    'api_bp',
    'videos_bp',
    'face_recognition_bp',
    'gpu_management_bp'
]