"""
Fix Unicode output issues on Windows by setting UTF-8 encoding
"""
import sys
import io
import os


def fix_unicode_output():
    """
    Fix Unicode output issues on Windows by wrapping stdout/stderr with UTF-8 encoding
    """
    if sys.platform == 'win32':
        # Set console code page to UTF-8
        os.system('chcp 65001 > nul')
        
        # Wrap stdout and stderr with UTF-8 encoding
        if sys.stdout and hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        if sys.stderr and hasattr(sys.stderr, 'buffer'):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        
        # Set environment variable for Python
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        
        print("[System] Unicode output fixed for Windows")
    else:
        print("[System] Unicode output already supported on this platform")


# Call this function at the start of your application
if __name__ == '__main__':
    fix_unicode_output()
    # Test with emojis
    print("[OK] Unicode test successful! [TARGET] [INFO] [PROCESSING]")