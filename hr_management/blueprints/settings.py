"""
Settings Management Blueprint
"""

from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for, current_app
from flask_login import login_required
import json

settings_bp = Blueprint('settings', __name__, url_prefix='/settings')

@settings_bp.route('/')
@login_required
def index():
    """Display settings page"""
    # Import SystemSettings from app context
    SystemSettings = current_app.SystemSettings if hasattr(current_app, 'SystemSettings') else None
    
    if not SystemSettings:
        flash('Settings system not initialized', 'error')
        return redirect(url_for('dashboard.index'))
    
    # Get all settings grouped by category
    settings_groups = SystemSettings.get_all_by_category()
    
    # Define category display names and icons
    category_info = {
        'general': {'name': 'General Settings', 'icon': 'cog'},
        'date_time': {'name': 'Date & Time Formats', 'icon': 'calendar'},
        'video_processing': {'name': 'Video Processing', 'icon': 'video'},
        'attendance': {'name': 'Attendance Rules', 'icon': 'clock'}
    }
    
    return render_template('settings/index.html', 
                         settings_groups=settings_groups,
                         category_info=category_info)

@settings_bp.route('/update', methods=['POST'])
@login_required
def update():
    """Update settings via AJAX"""
    SystemSettings = current_app.SystemSettings
    db = current_app.db
    
    try:
        data = request.get_json()
        key = data.get('key')
        value = data.get('value')
        
        if not key:
            return jsonify({'success': False, 'error': 'Setting key is required'}), 400
        
        # Get the setting
        setting = SystemSettings.query.filter_by(key=key).first()
        if not setting:
            return jsonify({'success': False, 'error': 'Setting not found'}), 404
        
        # Validate value based on type
        if setting.value_type == 'integer':
            try:
                value = str(int(value))
            except ValueError:
                return jsonify({'success': False, 'error': 'Invalid integer value'}), 400
        elif setting.value_type == 'boolean':
            value = 'true' if value in ['true', True, 1, '1'] else 'false'
        else:
            value = str(value)
        
        # Update the setting
        setting.value = value
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Setting "{key}" updated successfully',
            'value': setting.get_typed_value()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@settings_bp.route('/reset', methods=['POST'])
@login_required
def reset():
    """Reset a setting to default value"""
    SystemSettings = current_app.SystemSettings
    db = current_app.db
    
    try:
        data = request.get_json()
        key = data.get('key')
        
        if not key:
            return jsonify({'success': False, 'error': 'Setting key is required'}), 400
        
        # Find default value
        default_value = None
        for category, settings in SystemSettings.DEFAULT_SETTINGS.items():
            if key in settings:
                default_value = settings[key]['value']
                break
        
        if default_value is None:
            return jsonify({'success': False, 'error': 'No default value found'}), 404
        
        # Update the setting
        setting = SystemSettings.query.filter_by(key=key).first()
        if setting:
            setting.value = default_value
            db.session.commit()
            
            return jsonify({
                'success': True,
                'message': f'Setting "{key}" reset to default',
                'value': setting.get_typed_value()
            })
        else:
            return jsonify({'success': False, 'error': 'Setting not found'}), 404
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@settings_bp.route('/export')
@login_required
def export_settings():
    """Export all settings as JSON"""
    SystemSettings = current_app.SystemSettings
    
    settings = SystemSettings.query.all()
    export_data = {
        'exported_at': datetime.utcnow().isoformat(),
        'settings': {}
    }
    
    for setting in settings:
        if not setting.is_sensitive:  # Don't export sensitive settings
            export_data['settings'][setting.key] = {
                'value': setting.value,
                'type': setting.value_type,
                'category': setting.category,
                'description': setting.description
            }
    
    return jsonify(export_data)

@settings_bp.route('/import', methods=['POST'])
@login_required
def import_settings():
    """Import settings from JSON"""
    SystemSettings = current_app.SystemSettings
    db = current_app.db
    
    try:
        file = request.files.get('settings_file')
        if not file:
            flash('No file uploaded', 'error')
            return redirect(url_for('settings.index'))
        
        # Read and parse JSON
        import_data = json.load(file)
        settings_data = import_data.get('settings', {})
        
        imported_count = 0
        for key, data in settings_data.items():
            setting = SystemSettings.query.filter_by(key=key).first()
            if setting and not setting.is_sensitive:
                setting.value = data['value']
                imported_count += 1
        
        db.session.commit()
        flash(f'Successfully imported {imported_count} settings', 'success')
        
    except Exception as e:
        flash(f'Import failed: {str(e)}', 'error')
    
    return redirect(url_for('settings.index'))

# Helper function to get date format
def get_date_format():
    """Get the configured date format"""
    SystemSettings = current_app.SystemSettings if hasattr(current_app, 'SystemSettings') else None
    if SystemSettings:
        return SystemSettings.get_setting('date_format', 'DD-MM-YYYY')
    return 'DD-MM-YYYY'

# Helper function to format date according to settings
def format_date(date_obj):
    """Format a date object according to system settings"""
    if not date_obj:
        return ''
    
    format_string = get_date_format()
    
    # Convert format string to Python format
    format_map = {
        'DD-MM-YYYY': '%d-%m-%Y',
        'MM-DD-YYYY': '%m-%d-%Y',
        'YYYY-MM-DD': '%Y-%m-%d',
        'DD/MM/YYYY': '%d/%m/%Y',
        'MM/DD/YYYY': '%m/%d/%Y',
        'YYYY/MM/DD': '%Y/%m/%d'
    }
    
    py_format = format_map.get(format_string, '%d-%m-%Y')
    return date_obj.strftime(py_format)

from datetime import datetime