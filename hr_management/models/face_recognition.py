from datetime import datetime
from .base import db

class FaceDataset(db.Model):
    __tablename__ = 'face_datasets'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    dataset_path = db.Column(db.String(500), nullable=False)
    image_count = db.Column(db.Integer, default=0)
    person_count = db.Column(db.Integer, default=0)
    format_type = db.Column(db.String(20), default='yolo')  # yolo, coco, etc.
    status = db.Column(db.String(50), default='created')  # created, processing, ready, error
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    trained_models = db.relationship('TrainedModel', backref='dataset', lazy=True)
    
    def __repr__(self):
        return f'<FaceDataset {self.name}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'dataset_path': self.dataset_path,
            'image_count': self.image_count,
            'person_count': self.person_count,
            'format_type': self.format_type,
            'status': self.status,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class TrainedModel(db.Model):
    __tablename__ = 'trained_models'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    model_type = db.Column(db.String(50), nullable=False)  # face_recognition, yolo, etc.
    model_path = db.Column(db.String(500), nullable=False)
    config_path = db.Column(db.String(500))
    dataset_id = db.Column(db.Integer, db.ForeignKey('face_datasets.id'), nullable=False)
    
    # Training parameters
    epochs = db.Column(db.Integer)
    batch_size = db.Column(db.Integer)
    learning_rate = db.Column(db.Float)
    
    # Performance metrics
    accuracy = db.Column(db.Float)
    loss = db.Column(db.Float)
    validation_accuracy = db.Column(db.Float)
    validation_loss = db.Column(db.Float)
    
    # Training status
    status = db.Column(db.String(50), default='created')  # created, training, completed, failed
    training_progress = db.Column(db.Integer, default=0)
    training_log = db.Column(db.Text)
    error_message = db.Column(db.Text)
    
    # Version and deployment
    version = db.Column(db.String(20), default='1.0.0')
    is_active = db.Column(db.Boolean, default=False)
    deployed_at = db.Column(db.DateTime)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<TrainedModel {self.name}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'model_type': self.model_type,
            'model_path': self.model_path,
            'config_path': self.config_path,
            'dataset_id': self.dataset_id,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'accuracy': self.accuracy,
            'loss': self.loss,
            'validation_accuracy': self.validation_accuracy,
            'validation_loss': self.validation_loss,
            'status': self.status,
            'training_progress': self.training_progress,
            'version': self.version,
            'is_active': self.is_active,
            'deployed_at': self.deployed_at.isoformat() if self.deployed_at else None,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class RecognitionResult(db.Model):
    __tablename__ = 'recognition_results'
    
    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.Integer, db.ForeignKey('videos.id'), nullable=False)
    model_id = db.Column(db.Integer, db.ForeignKey('trained_models.id'), nullable=False)
    detected_person_id = db.Column(db.Integer, db.ForeignKey('detected_persons.id'), nullable=False)
    employee_id = db.Column(db.Integer, db.ForeignKey('employees.id'), nullable=True)
    confidence = db.Column(db.Float, nullable=False)
    frame_number = db.Column(db.Integer)
    bbox_coordinates = db.Column(db.JSON)
    is_verified = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    video = db.relationship('Video', backref='recognition_results')
    model = db.relationship('TrainedModel', backref='recognition_results')
    detected_person = db.relationship('DetectedPerson', backref='recognition_results')
    employee = db.relationship('Employee', backref='recognition_results')
    
    def __repr__(self):
        return f'<RecognitionResult Video:{self.video_id} Person:{self.detected_person_id}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'video_id': self.video_id,
            'model_id': self.model_id,
            'detected_person_id': self.detected_person_id,
            'employee_id': self.employee_id,
            'confidence': self.confidence,
            'frame_number': self.frame_number,
            'bbox_coordinates': self.bbox_coordinates,
            'is_verified': self.is_verified,
            'created_at': self.created_at.isoformat(),
            'employee_name': self.employee.name if self.employee else None
        }