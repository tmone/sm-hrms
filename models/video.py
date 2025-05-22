from datetime import datetime
from .base import db

class Video(db.Model):
    __tablename__ = 'videos'
    
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(500), nullable=False)
    duration = db.Column(db.Float)
    resolution = db.Column(db.String(20))
    fps = db.Column(db.Float)
    file_size = db.Column(db.BigInteger)
    status = db.Column(db.String(50), default='uploaded')  # uploaded, processing, completed, failed
    processing_progress = db.Column(db.Integer, default=0)
    error_message = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    detected_persons = db.relationship('DetectedPerson', backref='video', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Video {self.filename}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'original_filename': self.original_filename,
            'filepath': self.filepath,
            'duration': self.duration,
            'resolution': self.resolution,
            'fps': self.fps,
            'file_size': self.file_size,
            'status': self.status,
            'processing_progress': self.processing_progress,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat(),
            'detected_persons_count': len(self.detected_persons)
        }

class DetectedPerson(db.Model):
    __tablename__ = 'detected_persons'
    
    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.Integer, db.ForeignKey('videos.id'), nullable=False)
    person_code = db.Column(db.String(20), nullable=False)  # PERSON-XXXX format
    start_frame = db.Column(db.Integer, nullable=False)
    end_frame = db.Column(db.Integer, nullable=False)
    start_time = db.Column(db.Float)  # Time in seconds
    end_time = db.Column(db.Float)    # Time in seconds
    confidence = db.Column(db.Float)
    bbox_data = db.Column(db.JSON)  # Store bounding box coordinates for each frame
    thumbnail_path = db.Column(db.String(500))
    face_count = db.Column(db.Integer, default=0)
    employee_id = db.Column(db.Integer, db.ForeignKey('employees.id'), nullable=True)
    is_identified = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (db.Index('idx_video_person', 'video_id', 'person_code'),)
    
    def __repr__(self):
        return f'<DetectedPerson {self.person_code} in Video {self.video_id}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'video_id': self.video_id,
            'person_code': self.person_code,
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'confidence': self.confidence,
            'bbox_data': self.bbox_data,
            'thumbnail_path': self.thumbnail_path,
            'face_count': self.face_count,
            'employee_id': self.employee_id,
            'is_identified': self.is_identified,
            'created_at': self.created_at.isoformat(),
            'employee_name': self.employee.name if self.employee else None
        }