"""
Install face_recognition library with dependencies
"""

import subprocess
import sys
import platform

def install_face_recognition():
    """Install face_recognition library"""
    print("Installing face_recognition library...")
    
    # Check platform
    system = platform.system()
    
    if system == "Windows":
        print("[WARNING]  Windows detected. Installing face_recognition on Windows can be complex.")
        print("   You may need to install Visual Studio Build Tools first.")
        print("   Alternatively, you can install via conda:")
        print("   conda install -c conda-forge dlib")
        print("   conda install -c conda-forge face_recognition")
        print("")
        print("   Attempting pip install...")
    
    try:
        # Install dlib first (face_recognition dependency)
        print("Installing dlib...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "dlib"])
        
        # Install face_recognition
        print("Installing face_recognition...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "face_recognition"])
        
        # Install additional dependencies
        print("Installing additional dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn", "joblib"])
        
        print("[OK] face_recognition installed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Installation failed: {e}")
        print("\nAlternative installation methods:")
        print("1. Using conda: conda install -c conda-forge face_recognition")
        print("2. Using pre-built wheels: pip install face_recognition --find-links https://github.com/jloh02/dlib/releases/")
        print("3. Manual installation: See https://github.com/ageitgey/face_recognition#installation")
        
if __name__ == "__main__":
    install_face_recognition()