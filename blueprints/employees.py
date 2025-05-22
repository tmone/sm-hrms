from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_required
from models.employee import Employee, AttendanceRecord
from models.base import db
from datetime import datetime, date, timedelta
from sqlalchemy import or_

employees_bp = Blueprint('employees', __name__)

@employees_bp.route('/')
@login_required
def index():
    # Get filter parameters
    search = request.args.get('search', '')
    department = request.args.get('department', '')
    status = request.args.get('status', '')
    page = request.args.get('page', 1, type=int)
    per_page = 10
    
    # Build query
    query = Employee.query
    
    if search:
        query = query.filter(or_(
            Employee.name.contains(search),
            Employee.email.contains(search),
            Employee.employee_id.contains(search)
        ))
    
    if department:
        query = query.filter_by(department=department)
    
    if status:
        query = query.filter_by(status=status)
    
    # Paginate results
    employees = query.paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    # Get departments for filter dropdown
    departments = db.session.query(Employee.department.distinct()).all()
    departments = [d[0] for d in departments if d[0]]
    
    return render_template('employees/index.html',
                         employees=employees,
                         departments=departments,
                         search=search,
                         department=department,
                         status=status)

@employees_bp.route('/create', methods=['GET', 'POST'])
@login_required
def create():
    if request.method == 'POST':
        employee = Employee(
            name=request.form['name'],
            department=request.form['department'],
            email=request.form['email'],
            phone=request.form.get('phone', ''),
            position=request.form.get('position', ''),
            employee_id=request.form.get('employee_id', ''),
            hire_date=datetime.strptime(request.form['hire_date'], '%Y-%m-%d').date() if request.form.get('hire_date') else None,
            status=request.form.get('status', 'active')
        )
        
        try:
            db.session.add(employee)
            db.session.commit()
            flash(f'Employee {employee.name} created successfully!', 'success')
            return redirect(url_for('employees.detail', id=employee.id))
        except Exception as e:
            db.session.rollback()
            flash(f'Error creating employee: {str(e)}', 'error')
    
    return render_template('employees/create.html')

@employees_bp.route('/<int:id>')
@login_required
def detail(id):
    employee = Employee.query.get_or_404(id)
    
    # Get recent attendance records
    attendance_records = AttendanceRecord.query.filter_by(employee_id=id)\
                                              .order_by(AttendanceRecord.date.desc())\
                                              .limit(30).all()
    
    # Get attendance statistics
    total_days = AttendanceRecord.query.filter_by(employee_id=id).count()
    present_days = AttendanceRecord.query.filter_by(employee_id=id, status='present').count()
    attendance_rate = (present_days / total_days * 100) if total_days > 0 else 0
    
    return render_template('employees/detail.html',
                         employee=employee,
                         attendance_records=attendance_records,
                         attendance_stats={
                             'total_days': total_days,
                             'present_days': present_days,
                             'attendance_rate': attendance_rate
                         })

@employees_bp.route('/<int:id>/edit', methods=['GET', 'POST'])
@login_required
def edit(id):
    employee = Employee.query.get_or_404(id)
    
    if request.method == 'POST':
        employee.name = request.form['name']
        employee.department = request.form['department']
        employee.email = request.form['email']
        employee.phone = request.form.get('phone', '')
        employee.position = request.form.get('position', '')
        employee.employee_id = request.form.get('employee_id', '')
        employee.hire_date = datetime.strptime(request.form['hire_date'], '%Y-%m-%d').date() if request.form.get('hire_date') else None
        employee.status = request.form.get('status', 'active')
        employee.updated_at = datetime.utcnow()
        
        try:
            db.session.commit()
            flash(f'Employee {employee.name} updated successfully!', 'success')
            return redirect(url_for('employees.detail', id=employee.id))
        except Exception as e:
            db.session.rollback()
            flash(f'Error updating employee: {str(e)}', 'error')
    
    return render_template('employees/edit.html', employee=employee)

@employees_bp.route('/<int:id>/delete', methods=['POST'])
@login_required
def delete(id):
    employee = Employee.query.get_or_404(id)
    
    try:
        db.session.delete(employee)
        db.session.commit()
        flash(f'Employee {employee.name} deleted successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting employee: {str(e)}', 'error')
    
    return redirect(url_for('employees.index'))

@employees_bp.route('/<int:id>/attendance')
@login_required
def attendance(id):
    employee = Employee.query.get_or_404(id)
    
    # Get date range from query parameters
    start_date = request.args.get('start_date', (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'))
    end_date = request.args.get('end_date', datetime.now().strftime('%Y-%m-%d'))
    
    # Parse dates
    start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
    
    # Get attendance records
    attendance_records = AttendanceRecord.query.filter(
        AttendanceRecord.employee_id == id,
        AttendanceRecord.date >= start_date,
        AttendanceRecord.date <= end_date
    ).order_by(AttendanceRecord.date.desc()).all()
    
    return render_template('employees/attendance.html',
                         employee=employee,
                         attendance_records=attendance_records,
                         start_date=start_date,
                         end_date=end_date)

@employees_bp.route('/api')
@login_required
def api_list():
    # API endpoint for employee data (used by other components)
    employees = Employee.query.filter_by(status='active').all()
    return jsonify([emp.to_dict() for emp in employees])