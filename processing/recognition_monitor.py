"""Simple recognition monitoring for debugging"""

import logging
from datetime import datetime
from pathlib import Path

# Create a special logger for recognition
recognition_logger = logging.getLogger('recognition_monitor')
recognition_logger.setLevel(logging.INFO)

# Create file handler
log_dir = Path('processing/recognition_logs')
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f'recognition_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

fh = logging.FileHandler(log_file)
fh.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(message)s')
fh.setFormatter(formatter)

# Add handler
recognition_logger.addHandler(fh)

# Also log to console
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
recognition_logger.addHandler(ch)

def log_recognition_event(event_type, data):
    """Log a recognition event"""
    recognition_logger.info(f"[{event_type}] {data}")

# Make it globally available
print(f"Recognition logging to: {log_file}")
