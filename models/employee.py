from datetime import datetime
from flask_login import UserMixin

# Models will be initialized by app.py
class Employee(UserMixin):
    pass

class AttendanceRecord:
    pass

def create_employee_models(db):
    """Create Employee models with database instance"""
    
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
    
    return Employee, AttendanceRecord