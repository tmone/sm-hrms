from datetime import datetime
from flask_login import UserMixin
from flask import current_app

def get_db():
    """Get the database instance from current app context"""
    return current_app.extensions['sqlalchemy']

class Employee(UserMixin):
    """Employee model - will be properly initialized when app starts"""
    pass

class AttendanceRecord:
    """Attendance record model - will be properly initialized when app starts"""
    pass