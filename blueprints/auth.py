from flask import Blueprint, render_template, request, redirect, url_for, flash, session, current_app
from flask_login import login_user, logout_user, login_required, current_user

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.index'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        # For demo purposes, we'll create a simple login system
        # In production, you should implement proper password hashing
        
        Employee = current_app.Employee
        employee = Employee.query.filter_by(email=email).first()
        if employee:
            login_user(employee, remember=True)
            next_page = request.args.get('next')
            flash(f'Welcome back, {employee.name}!', 'success')
            return redirect(next_page) if next_page else redirect(url_for('dashboard.index'))
        else:
            flash('Invalid email address.', 'error')
    
    return render_template('auth/login.html')

@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('auth.login'))

@auth_bp.route('/demo-login')
def demo_login():
    # Get models from current app
    Employee = current_app.Employee
    db = current_app.extensions['sqlalchemy']
    
    # Create a demo admin user if it doesn't exist
    admin = Employee.query.filter_by(email='admin@stepmedia.com').first()
    if not admin:
        admin = Employee(
            name='Admin User',
            email='admin@stepmedia.com',
            department='IT',
            position='System Administrator',
            employee_id='EMP001',
            status='active'
        )
        db.session.add(admin)
        db.session.commit()
    
    login_user(admin, remember=True)
    flash(f'Demo login successful! Welcome, {admin.name}', 'success')
    return redirect(url_for('dashboard.index'))