"""
Date utilities using system settings
"""

from datetime import datetime, date
from flask import current_app
import re

def get_date_format_patterns():
    """Get date format patterns based on system settings"""
    # Try to get from settings
    try:
        SystemSettings = current_app.SystemSettings
        ocr_format = SystemSettings.get_setting('ocr_date_format', 'DD-MM-YYYY')
    except:
        # Default if no app context
        ocr_format = 'DD-MM-YYYY'
    
    # Map settings format to regex patterns and strptime formats
    format_map = {
        'DD-MM-YYYY': {
            'regex': r'(\d{1,2})[-/](\d{1,2})[-/](\d{4})',
            'strptime': '%d-%m-%Y',
            'groups': ['day', 'month', 'year']
        },
        'MM-DD-YYYY': {
            'regex': r'(\d{1,2})[-/](\d{1,2})[-/](\d{4})',
            'strptime': '%m-%d-%Y',
            'groups': ['month', 'day', 'year']
        },
        'YYYY-MM-DD': {
            'regex': r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})',
            'strptime': '%Y-%m-%d',
            'groups': ['year', 'month', 'day']
        },
        'DD/MM/YYYY': {
            'regex': r'(\d{1,2})[-/](\d{1,2})[-/](\d{4})',
            'strptime': '%d/%m/%Y',
            'groups': ['day', 'month', 'year']
        },
        'MM/DD/YYYY': {
            'regex': r'(\d{1,2})[-/](\d{1,2})[-/](\d{4})',
            'strptime': '%m/%d/%Y',
            'groups': ['month', 'day', 'year']
        }
    }
    
    return format_map.get(ocr_format, format_map['DD-MM-YYYY'])

def parse_ocr_date(date_text):
    """Parse date from OCR text using configured format"""
    if not date_text:
        return None
    
    # Clean the text
    date_text = date_text.strip()
    
    # Get configured format patterns
    format_info = get_date_format_patterns()
    
    # Try to match the pattern
    match = re.search(format_info['regex'], date_text)
    if match:
        try:
            # Extract date parts
            date_str = match.group()
            # Normalize separators
            date_str = date_str.replace('/', '-')
            
            # Parse using strptime
            return datetime.strptime(date_str, format_info['strptime']).date()
        except ValueError:
            pass
    
    # Try alternative patterns if primary fails
    alternative_patterns = [
        (r'(\d{1,2})[-/](\d{1,2})[-/](\d{4})', '%d-%m-%Y'),
        (r'(\d{1,2})[-/](\d{1,2})[-/](\d{4})', '%m-%d-%Y'),
        (r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})', '%Y-%m-%d'),
    ]
    
    for pattern, strptime_fmt in alternative_patterns:
        match = re.search(pattern, date_text)
        if match:
            try:
                date_str = match.group().replace('/', '-')
                return datetime.strptime(date_str, strptime_fmt).date()
            except ValueError:
                continue
    
    return None

def format_date_for_display(date_obj):
    """Format date according to system settings"""
    if not date_obj:
        return ''
    
    try:
        SystemSettings = current_app.SystemSettings
        display_format = SystemSettings.get_setting('date_format', 'DD-MM-YYYY')
    except:
        display_format = 'DD-MM-YYYY'
    
    # Convert to Python strftime format
    format_map = {
        'DD-MM-YYYY': '%d-%m-%Y',
        'MM-DD-YYYY': '%m-%d-%Y',
        'YYYY-MM-DD': '%Y-%m-%d',
        'DD/MM/YYYY': '%d/%m/%Y',
        'MM/DD/YYYY': '%m/%d/%Y',
        'YYYY/MM/DD': '%Y/%m/%d'
    }
    
    py_format = format_map.get(display_format, '%d-%m-%Y')
    
    if isinstance(date_obj, str):
        # Parse string first
        try:
            date_obj = datetime.strptime(date_obj, '%Y-%m-%d').date()
        except:
            return date_obj
    
    return date_obj.strftime(py_format)

def get_time_format():
    """Get time format setting (12h or 24h)"""
    try:
        SystemSettings = current_app.SystemSettings
        return SystemSettings.get_setting('time_format', '24h')
    except:
        return '24h'

def format_time_for_display(time_obj):
    """Format time according to system settings"""
    if not time_obj:
        return ''
    
    time_format = get_time_format()
    
    if isinstance(time_obj, str):
        # If it's already a string, return as-is
        return time_obj
    
    if time_format == '12h':
        return time_obj.strftime('%I:%M:%S %p')
    else:
        return time_obj.strftime('%H:%M:%S')