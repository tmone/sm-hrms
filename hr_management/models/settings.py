"""
System Settings Model
"""

from datetime import datetime
from hr_management.models.base import db

class SystemSettings(db.Model):
    """System-wide configuration settings"""
    __tablename__ = 'system_settings'
    
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(100), unique=True, nullable=False)
    value = db.Column(db.Text)
    value_type = db.Column(db.String(20), default='string')  # string, integer, boolean, json
    category = db.Column(db.String(50), default='general')
    description = db.Column(db.Text)
    is_sensitive = db.Column(db.Boolean, default=False)  # For passwords, API keys, etc.
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Default settings
    DEFAULT_SETTINGS = {
        'general': {
            'app_name': {
                'value': 'StepMedia HRM',
                'type': 'string',
                'description': 'Application name displayed in UI'
            },
            'timezone': {
                'value': 'Asia/Bangkok',
                'type': 'string',
                'description': 'Default timezone for the application'
            },
            'language': {
                'value': 'en',
                'type': 'string',
                'description': 'Default language (en, th, vi)'
            }
        },
        'date_time': {
            'date_format': {
                'value': 'DD-MM-YYYY',
                'type': 'string',
                'description': 'Date format for display (DD-MM-YYYY, MM-DD-YYYY, YYYY-MM-DD)'
            },
            'time_format': {
                'value': '24h',
                'type': 'string',
                'description': 'Time format (12h or 24h)'
            },
            'ocr_date_format': {
                'value': 'DD-MM-YYYY',
                'type': 'string',
                'description': 'Expected date format in OCR extraction'
            }
        },
        'video_processing': {
            'max_upload_size_mb': {
                'value': '2048',
                'type': 'integer',
                'description': 'Maximum video upload size in MB'
            },
            'auto_process_on_upload': {
                'value': 'true',
                'type': 'boolean',
                'description': 'Automatically process videos after upload'
            },
            'default_fps': {
                'value': '30',
                'type': 'integer',
                'description': 'Default FPS for video processing'
            },
            'ocr_sample_interval': {
                'value': '10',
                'type': 'integer',
                'description': 'OCR sampling interval in seconds'
            }
        },
        'attendance': {
            'work_start_time': {
                'value': '08:00',
                'type': 'string',
                'description': 'Standard work start time'
            },
            'work_end_time': {
                'value': '17:00',
                'type': 'string',
                'description': 'Standard work end time'
            },
            'late_threshold_minutes': {
                'value': '15',
                'type': 'integer',
                'description': 'Minutes after work start to consider late'
            },
            'minimum_presence_seconds': {
                'value': '5',
                'type': 'integer',
                'description': 'Minimum seconds to count as present'
            }
        }
    }
    
    def __repr__(self):
        return f'<SystemSettings {self.key}={self.value}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'key': self.key,
            'value': self.get_typed_value(),
            'value_type': self.value_type,
            'category': self.category,
            'description': self.description,
            'is_sensitive': self.is_sensitive,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    def get_typed_value(self):
        """Get value converted to its proper type"""
        if self.value_type == 'integer':
            return int(self.value) if self.value else 0
        elif self.value_type == 'boolean':
            return self.value.lower() in ('true', '1', 'yes', 'on')
        elif self.value_type == 'json':
            import json
            try:
                return json.loads(self.value)
            except:
                return {}
        else:
            return self.value
    
    @classmethod
    def get_setting(cls, key, default=None):
        """Get a setting value by key"""
        setting = cls.query.filter_by(key=key).first()
        if setting:
            return setting.get_typed_value()
        return default
    
    @classmethod
    def set_setting(cls, key, value, value_type='string', category='general', description=None):
        """Set a setting value"""
        setting = cls.query.filter_by(key=key).first()
        
        if not setting:
            setting = cls(key=key, value_type=value_type, category=category, description=description)
            db.session.add(setting)
        
        # Convert value to string for storage
        if value_type == 'boolean':
            setting.value = 'true' if value else 'false'
        elif value_type == 'json':
            import json
            setting.value = json.dumps(value)
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
                    cls.set_setting(
                        key=key,
                        value=config['value'],
                        value_type=config['type'],
                        category=category,
                        description=config['description']
                    )
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