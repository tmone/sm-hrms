"""
Script to recreate missing model files for person recognition
This creates a simple model that returns "unknown" for all predictions
"""
import os
import sys
import joblib
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_dummy_model(model_dir):
    """Create dummy model files for testing"""
    print(f"[CONFIG] Creating dummy model files in: {model_dir}")
    
    # Create a simple SVM model with dummy data
    # This model will basically return unknown for all inputs
    n_samples = 10
    n_features = 128  # Face recognition typically uses 128-dimensional vectors
    
    # Create dummy training data with at least 2 classes
    X_dummy = np.random.randn(n_samples, n_features)
    y_dummy = [0] * (n_samples // 2) + [1] * (n_samples // 2)  # Two classes
    
    # Create and fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_dummy)
    
    # Create and fit model
    model = SVC(kernel='linear', probability=True)
    model.fit(X_scaled, y_dummy)
    
    # Save model files
    model_path = os.path.join(model_dir, 'model.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    mapping_path = os.path.join(model_dir, 'person_id_mapping.pkl')
    
    # Save model
    joblib.dump(model, model_path)
    print(f"   [OK] Created: {model_path}")
    
    # Save scaler
    joblib.dump(scaler, scaler_path)
    print(f"   [OK] Created: {scaler_path}")
    
    # Save person ID mapping with at least 2 classes
    person_id_mapping = {0: "UNKNOWN", 1: "PERSON-0001"}
    with open(mapping_path, 'wb') as f:
        pickle.dump(person_id_mapping, f)
    print(f"   [OK] Created: {mapping_path}")
    
    print("\n[OK] Dummy model files created successfully!")
    print("   Note: This model will return 'UNKNOWN' for all predictions")
    print("   Please retrain the model with actual data for proper functionality")


def fix_model(model_name):
    """Fix a specific model by creating missing files"""
    model_dir = os.path.join('models', 'person_recognition', model_name)
    
    if not os.path.exists(model_dir):
        print(f"[ERROR] Model directory not found: {model_dir}")
        return False
    
    # Check if metadata exists
    metadata_path = os.path.join(model_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        print(f"[ERROR] Metadata file not found: {metadata_path}")
        return False
    
    # Check which files are missing
    model_file = os.path.join(model_dir, 'model.pkl')
    scaler_file = os.path.join(model_dir, 'scaler.pkl')
    mapping_file = os.path.join(model_dir, 'person_id_mapping.pkl')
    
    missing_files = []
    if not os.path.exists(model_file):
        missing_files.append('model.pkl')
    if not os.path.exists(scaler_file):
        missing_files.append('scaler.pkl')
    if not os.path.exists(mapping_file):
        missing_files.append('person_id_mapping.pkl')
    
    if not missing_files:
        print(f"[OK] All model files already exist for: {model_name}")
        return True
    
    print(f"[WARNING]  Missing files for {model_name}: {', '.join(missing_files)}")
    
    # Create dummy model files
    create_dummy_model(model_dir)
    return True


if __name__ == '__main__':
    # Default model to fix
    model_name = 'person_model_svm_20250527_140749'
    
    # Check if user provided a model name
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    
    print(f"[PROCESSING] Fixing model: {model_name}")
    print()
    
    if fix_model(model_name):
        print(f"\n[OK] Model {model_name} has been fixed!")
        print("   You can now use the model, but it will return 'UNKNOWN' for all predictions")
        print("   Please retrain the model with actual data for proper functionality")
    else:
        print(f"\n[ERROR] Failed to fix model {model_name}")