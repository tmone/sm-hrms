"""
Logging configuration for HRM system
Separates logs into different files and controls console output
"""
import os
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path

# Create logs directory
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Log file paths
LOG_FILES = {
    'api': LOGS_DIR / f'api_{datetime.now().strftime("%Y%m%d")}.log',
    'background': LOGS_DIR / f'background_{datetime.now().strftime("%Y%m%d")}.log',
    'video_processing': LOGS_DIR / f'video_processing_{datetime.now().strftime("%Y%m%d")}.log',
    'gpu': LOGS_DIR / f'gpu_{datetime.now().strftime("%Y%m%d")}.log',
    'database': LOGS_DIR / f'database_{datetime.now().strftime("%Y%m%d")}.log',
    'app': LOGS_DIR / f'app_{datetime.now().strftime("%Y%m%d")}.log',
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'format': '[%(asctime)s] %(levelname)-8s [%(name)s:%(lineno)d] %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'simple': {
            'format': '%(levelname)-8s %(message)s'
        },
        'console': {
            'format': '%(message)s'  # Clean console output
        }
    },
    'filters': {
        'important_only': {
            '()': 'config_logging.ImportantOnlyFilter',
        }
    },
    'handlers': {
        # Console handler - only important messages
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'WARNING',  # Only warnings and errors on console
            'formatter': 'console',
            'filters': ['important_only']
        },
        # File handlers for different components
        'api_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': str(LOG_FILES['api']),
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'detailed',
            'level': 'DEBUG'
        },
        'background_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': str(LOG_FILES['background']),
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'detailed',
            'level': 'DEBUG'
        },
        'video_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': str(LOG_FILES['video_processing']),
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'detailed',
            'level': 'DEBUG'
        },
        'gpu_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': str(LOG_FILES['gpu']),
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'detailed',
            'level': 'DEBUG'
        },
        'database_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': str(LOG_FILES['database']),
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'detailed',
            'level': 'DEBUG'
        },
        'app_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': str(LOG_FILES['app']),
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'detailed',
            'level': 'INFO'
        }
    },
    'loggers': {
        # API endpoints
        'hr_management.blueprints.api': {
            'handlers': ['api_file', 'console'],
            'level': 'DEBUG',
            'propagate': False
        },
        'blueprints.api': {
            'handlers': ['api_file', 'console'],
            'level': 'DEBUG',
            'propagate': False
        },
        # Background tasks
        'processing': {
            'handlers': ['background_file', 'console'],
            'level': 'DEBUG',
            'propagate': False
        },
        'hr_management.processing': {
            'handlers': ['background_file', 'console'],
            'level': 'DEBUG',
            'propagate': False
        },
        # Video processing
        'processing.gpu_enhanced_detection': {
            'handlers': ['video_file', 'console'],
            'level': 'DEBUG',
            'propagate': False
        },
        'processing.video_chunk_manager': {
            'handlers': ['video_file', 'console'],
            'level': 'DEBUG',
            'propagate': False
        },
        'hr_management.blueprints.videos': {
            'handlers': ['video_file', 'console'],
            'level': 'DEBUG',
            'propagate': False
        },
        # GPU specific
        'processing.gpu_processing_queue': {
            'handlers': ['gpu_file', 'console'],
            'level': 'DEBUG',
            'propagate': False
        },
        'processing.gpu_resource_manager': {
            'handlers': ['gpu_file', 'console'],
            'level': 'DEBUG',
            'propagate': False
        },
        # Database
        'sqlalchemy': {
            'handlers': ['database_file'],
            'level': 'WARNING',
            'propagate': False
        },
        # Root logger
        'root': {
            'handlers': ['app_file', 'console'],
            'level': 'INFO'
        }
    }
}


class ImportantOnlyFilter(logging.Filter):
    """Filter to only show important messages on console"""
    
    # Keywords that indicate important messages
    IMPORTANT_KEYWORDS = [
        'error', 'failed', 'exception', 'critical',
        'completed successfully', 'finished', 'started',
        'gpu', 'cuda', 'uploaded', 'processed'
    ]
    
    def filter(self, record):
        # Always show warnings and errors
        if record.levelno >= logging.WARNING:
            return True
            
        # Check if message contains important keywords
        message_lower = record.getMessage().lower()
        return any(keyword in message_lower for keyword in self.IMPORTANT_KEYWORDS)


def setup_logging():
    """Initialize logging configuration"""
    import logging.config
    
    # Apply configuration
    logging.config.dictConfig(LOGGING_CONFIG)
    
    # Create a simple progress logger for console
    progress_logger = logging.getLogger('progress')
    progress_logger.setLevel(logging.INFO)
    progress_logger.propagate = False
    
    # Add only console handler for progress
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    progress_logger.addHandler(console_handler)
    
    return progress_logger


# Progress bar utilities
class ProgressBar:
    """Simple progress bar for console output"""
    
    def __init__(self, total, width=50, prefix='Progress'):
        self.total = total
        self.width = width
        self.prefix = prefix
        self.current = 0
        
    def update(self, current):
        """Update progress bar"""
        self.current = min(current, self.total)
        percent = self.current / self.total if self.total > 0 else 0
        filled = int(self.width * percent)
        bar = '█' * filled + '░' * (self.width - filled)
        
        # Use carriage return to update same line
        print(f'\r{self.prefix}: |{bar}| {percent*100:.1f}% ({self.current}/{self.total})', end='', flush=True)
        
        if self.current >= self.total:
            print()  # New line when complete
            
    def finish(self):
        """Mark as complete"""
        self.update(self.total)


def get_logger(name):
    """Get a logger instance with proper configuration"""
    return logging.getLogger(name)


def log_to_file(message, log_type='app'):
    """Convenience function to log directly to a specific file"""
    logger = logging.getLogger(f'file.{log_type}')
    
    # Ensure handler exists
    if not logger.handlers:
        handler = logging.handlers.RotatingFileHandler(
            str(LOG_FILES.get(log_type, LOG_FILES['app'])),
            maxBytes=10485760,
            backupCount=5
        )
        handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    
    logger.info(message)