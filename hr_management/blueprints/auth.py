from flask import Blueprint, render_template, request, redirect, url_for, flash, session, current_app
from flask_login import login_user, logout_user, login_required, current_user

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/test-auth')
def test_auth():
    return "<h1>[OK] Auth Blueprint Working!</h1><p><a href='/auth/login'>Go to Login</a></p>"

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.index'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        # For demo purposes, we'll create a simple login system
        # In production, you should implement proper password hashing
        
        try:
            Employee = current_app.Employee
            employee = Employee.query.filter_by(email=email).first()
            if employee:
                login_user(employee, remember=True)
                next_page = request.args.get('next')
                flash(f'Welcome back, {employee.name}!', 'success')
                return redirect(next_page) if next_page else redirect(url_for('dashboard.index'))
            else:
                flash('Invalid email address.', 'error')
        except Exception as e:
            return f"<h1>Login Error</h1><p>{str(e)}</p>"
    
    # Simple HTML login form instead of template for now
    return '''
    <!DOCTYPE html>
    <html>
    <head><title>StepMedia HRM - Login</title></head>
    <body style="font-family: Arial; max-width: 400px; margin: 100px auto; padding: 20px;">
        <h2>StepMedia HRM Login</h2>
        <form method="POST">
            <div style="margin-bottom: 15px;">
                <label>Email:</label><br>
                <input type="email" name="email" required style="width: 100%; padding: 8px; margin-top: 5px;">
            </div>
            <button type="submit" style="background: #007cba; color: white; padding: 10px 20px; border: none; cursor: pointer;">Login</button>
        </form>
        <br>
        <a href="/auth/demo-login" style="background: #28a745; color: white; padding: 10px 20px; text-decoration: none; display: inline-block;">Demo Login</a>
        <br><br>
        <a href="/test">Back to Test</a>
    </body>
    </html>
    '''

@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('auth.login'))

@auth_bp.route('/demo-login')
def demo_login():
    try:
        # Get models from current app
        Employee = current_app.Employee
        
        # Get the database session properly
        from flask import g
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
        
    except Exception as e:
        # Debug error
        return f"<h1>Demo Login Error</h1><p>{str(e)}</p><p><a href='/test'>Back to Test</a></p>"