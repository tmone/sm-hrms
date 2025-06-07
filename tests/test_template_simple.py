#!/usr/bin/env python3
"""Simple template syntax test"""

from jinja2 import Environment, FileSystemLoader, TemplateSyntaxError

try:
    # Create Jinja2 environment
    env = Environment(loader=FileSystemLoader('templates'))
    
    # Try to load the template - this will fail if syntax is invalid
    template = env.get_template('videos/detail.html')
    
    print("[OK] Template syntax is valid - no syntax errors found!")
    print("[OK] The template should load correctly in Flask")
    
except TemplateSyntaxError as e:
    print(f"[ERROR] Template syntax error found:")
    print(f"   Line {e.lineno}: {e.message}")
    print(f"   Near: {e.source[:100]}...")
    
except Exception as e:
    print(f"[ERROR] Unexpected error: {e}")
    import traceback
    traceback.print_exc()