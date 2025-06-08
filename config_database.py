"""
Database configuration for handling connection pool issues
"""

# SQLAlchemy connection pool configuration
SQLALCHEMY_ENGINE_OPTIONS = {
    'pool_size': 20,          # Increase from default 5 to 20
    'max_overflow': 30,       # Increase from default 10 to 30
    'pool_timeout': 60,       # Increase timeout from 30 to 60 seconds
    'pool_recycle': 3600,     # Recycle connections after 1 hour
    'pool_pre_ping': True,    # Check connection health before using
}

# Additional database settings
SQLALCHEMY_TRACK_MODIFICATIONS = False  # Disable event system to save resources
SQLALCHEMY_ECHO = False  # Set to True for SQL query debugging