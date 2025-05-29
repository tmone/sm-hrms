#!/usr/bin/env python3
"""
Quick script to run automatic model refinement
"""

import subprocess
import sys
import os

if __name__ == '__main__':
    print("Starting automatic model refinement...")
    print("This will try multiple methods to find the best model.")
    print("Expected time: 5-15 minutes\n")
    
    # Run the auto refinement script
    script_path = os.path.join(os.path.dirname(__file__), 'auto_refine_best_model.py')
    
    try:
        subprocess.run([sys.executable, script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running auto refinement: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nRefinement cancelled by user")
        sys.exit(0)