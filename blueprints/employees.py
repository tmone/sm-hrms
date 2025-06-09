from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app
from flask_login import login_required
from datetime import datetime, date, timedelta
from sqlalchemy import or_

employees_bp = Blueprint('employees', __name__)

@employees_bp.route('/')
@login_required
def index():
    # Get models from current app
    Employee = current_app.Employee
    db = current_app.db
    
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
        Employee = current_app.Employee
        db = current_app.db
        
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
    Employee = current_app.Employee
    AttendanceRecord = current_app.AttendanceRecord
    db = current_app.db
    
    employee = Employee.query.get_or_404(id)
    
    # Get recent attendance records
    recent_attendance = AttendanceRecord.query.filter_by(employee_id=id)\
                                              .order_by(AttendanceRecord.date.desc())\
                                              .limit(5).all()
    
    # Get assigned person codes
    assigned_person_codes = []
    try:
        # Query the mapping table
        result = db.session.execute(
            "SELECT person_code, is_primary, confidence, mapped_at FROM employee_person_mappings WHERE employee_id = :emp_id ORDER BY is_primary DESC, mapped_at DESC",
            {"emp_id": id}
        )
        assigned_person_codes = []
        for row in result:
            # Format the display of person code
            person_code = row[0]
            display_code = f"PERSON-{person_code}" if person_code.isdigit() else person_code
            assigned_person_codes.append({
                "person_code": person_code, 
                "display_code": display_code,
                "is_primary": row[1], 
                "confidence": row[2], 
                "mapped_at": row[3]
            })
    except Exception as e:
        # Table might not exist yet
        print(f"Could not get person mappings: {e}")
    
    # Get available unmapped person codes
    available_person_codes = []
    try:
        # Get all unique person codes from detections
        all_persons = db.session.execute(
            "SELECT DISTINCT person_id FROM detected_persons WHERE person_id IS NOT NULL ORDER BY person_id"
        ).fetchall()
        
        # Get ALL mapped person codes (from any employee)
        mapped_codes = db.session.execute(
            "SELECT DISTINCT person_code FROM employee_person_mappings"
        ).fetchall()
        mapped_codes = [row[0] for row in mapped_codes]
        
        # Only show person codes that haven't been mapped to ANY employee yet
        for row in all_persons:
            person_id = str(row[0])  # Convert to string to handle any type
            # Use the person_id as-is since it's already in the format used by the system
            if person_id not in mapped_codes:
                available_person_codes.append(person_id)
    except Exception as e:
        print(f"Could not get available person codes: {e}")
    
    return render_template('employees/detail.html',
                         employee=employee,
                         recent_attendance=recent_attendance,
                         assigned_person_codes=assigned_person_codes,
                         available_person_codes=available_person_codes)

@employees_bp.route('/<int:id>/edit', methods=['GET', 'POST'])
@login_required
def edit(id):
    Employee = current_app.Employee
    db = current_app.db
    
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
    Employee = current_app.Employee
    db = current_app.db
    
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
    Employee = current_app.Employee
    AttendanceRecord = current_app.AttendanceRecord
    
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
    
    # Calculate stats
    total_days = len(attendance_records)
    present_days = len([r for r in attendance_records if r.status == 'present'])
    absent_days = len([r for r in attendance_records if r.status == 'absent'])
    late_days = len([r for r in attendance_records if r.status == 'late'])
    attendance_rate = (present_days / total_days * 100) if total_days > 0 else 0
    
    stats = {
        'present_days': present_days,
        'absent_days': absent_days,
        'late_days': late_days,
        'attendance_rate': attendance_rate
    }
    
    return render_template('employees/attendance.html',
                         employee=employee,
                         attendance_records=attendance_records,
                         stats=stats,
                         start_date=start_date,
                         end_date=end_date)

@employees_bp.route('/<int:id>/map-person', methods=['POST'])
@login_required
def map_person(id):
    """Map a person code to an employee"""
    Employee = current_app.Employee
    db = current_app.db
    
    employee = Employee.query.get_or_404(id)
    person_code = request.form.get('person_code')
    is_primary = request.form.get('is_primary', 'true').lower() == 'true'
    notes = request.form.get('notes', '')
    
    if not person_code:
        flash('Please select a person code', 'error')
        return redirect(url_for('employees.detail', id=id))
    
    try:
        # Check if mapping already exists
        existing = db.session.execute(
            "SELECT 1 FROM employee_person_mappings WHERE employee_id = :emp_id AND person_code = :code",
            {"emp_id": id, "code": person_code}
        ).fetchone()
        
        if existing:
            flash(f'Person {person_code} is already mapped to this employee', 'warning')
            return redirect(url_for('employees.detail', id=id))
        
        # If marking as primary, unset other primary mappings
        if is_primary:
            db.session.execute(
                "UPDATE employee_person_mappings SET is_primary = 0 WHERE employee_id = :emp_id",
                {"emp_id": id}
            )
        
        # Create new mapping
        db.session.execute(
            """INSERT INTO employee_person_mappings 
               (employee_id, person_code, is_primary, confidence, mapped_by, notes, mapped_at) 
               VALUES (:emp_id, :code, :primary, 1.0, :user, :notes, CURRENT_TIMESTAMP)""",
            {
                "emp_id": id,
                "code": person_code,
                "primary": 1 if is_primary else 0,
                "user": "admin",  # TODO: Get actual user
                "notes": notes
            }
        )
        
        # Update employee's assigned_person_codes field
        mappings = db.session.execute(
            "SELECT person_code FROM employee_person_mappings WHERE employee_id = :emp_id ORDER BY is_primary DESC",
            {"emp_id": id}
        ).fetchall()
        codes = [row[0] for row in mappings]
        
        db.session.execute(
            "UPDATE employees SET assigned_person_codes = :codes WHERE id = :emp_id",
            {"codes": ",".join(codes), "emp_id": id}
        )
        
        db.session.commit()
        flash(f'Successfully mapped {person_code} to {employee.name}', 'success')
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error mapping person code: {str(e)}', 'error')
    
    return redirect(url_for('employees.detail', id=id))

@employees_bp.route('/<int:id>/unmap-person', methods=['POST'])
@login_required
def unmap_person(id):
    """Remove a person code mapping from an employee"""
    Employee = current_app.Employee
    db = current_app.db
    
    employee = Employee.query.get_or_404(id)
    person_code = request.form.get('person_code')
    
    if not person_code:
        flash('Invalid person code', 'error')
        return redirect(url_for('employees.detail', id=id))
    
    try:
        # Delete mapping
        result = db.session.execute(
            "DELETE FROM employee_person_mappings WHERE employee_id = :emp_id AND person_code = :code",
            {"emp_id": id, "code": person_code}
        )
        
        if result.rowcount == 0:
            flash('Mapping not found', 'warning')
            return redirect(url_for('employees.detail', id=id))
        
        # Update employee's assigned_person_codes field
        mappings = db.session.execute(
            "SELECT person_code FROM employee_person_mappings WHERE employee_id = :emp_id ORDER BY is_primary DESC",
            {"emp_id": id}
        ).fetchall()
        codes = [row[0] for row in mappings]
        
        db.session.execute(
            "UPDATE employees SET assigned_person_codes = :codes WHERE id = :emp_id",
            {"codes": ",".join(codes) if codes else None, "emp_id": id}
        )
        
        db.session.commit()
        flash(f'Successfully unmapped {person_code} from {employee.name}', 'success')
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error unmapping person code: {str(e)}', 'error')
    
    return redirect(url_for('employees.detail', id=id))

@employees_bp.route('/api')
@login_required
def api_list():
    # API endpoint for employee data (used by other components)
    Employee = current_app.Employee
    employees = Employee.query.filter_by(status='active').all()
    return jsonify([emp.to_dict() for emp in employees])