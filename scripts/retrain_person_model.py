"""
Script to retrain the person recognition model when model files are missing
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hr_management.processing.person_recognition_trainer import PersonRecognitionTrainer
from datetime import datetime


def retrain_model():
    """Retrain the person recognition model"""
    print("[PROCESSING] Starting person recognition model retraining...")
    
    trainer = PersonRecognitionTrainer()
    
    # Generate new model name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"person_model_svm_{timestamp}"
    
    print(f"[INFO] Creating new model: {model_name}")
    
    try:
        # Train the model
        metadata = trainer.train_model(
            model_name=model_name,
            model_type='svm'
        )
        
        print("\n[OK] Model training completed successfully!")
        print(f"   Model saved as: {model_name}")
        print(f"   Number of persons: {metadata['num_persons']}")
        print(f"   Training samples: {metadata['num_train_samples']}")
        print(f"   Test accuracy: {metadata['test_score']:.3f}")
        
        # Update the person recognition config to use the new model
        config_path = 'models/person_recognition/config.json'
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            config['default_model'] = model_name
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"\n[OK] Updated default model to: {model_name}")
        
        return model_name
        
    except Exception as e:
        print(f"\n[ERROR] Error training model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    # Check if dataset exists
    dataset_path = 'datasets/person_features.pkl'
    if not os.path.exists(dataset_path):
        print("[ERROR] No person dataset found at datasets/person_features.pkl")
        print("   Please create a dataset first using the person recognition feature")
        sys.exit(1)
    
    # Ask for confirmation
    print("[WARNING]  This will create a new person recognition model")
    print("   The process may take a few minutes depending on dataset size")
    print()
    
    confirmation = input("Do you want to continue? Type 'YES' to confirm: ")
    
    if confirmation == 'YES':
        model_name = retrain_model()
        if model_name:
            print(f"\n🎉 You can now use the model: {model_name}")
            print("   Update your application to use this model name")
    else:
        print("[ERROR] Training cancelled")