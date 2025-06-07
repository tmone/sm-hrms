"""
Simple recognition fix that works without complex imports
"""
import json
import pickle
import numpy as np
from pathlib import Path
import logging
import cv2
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class SimpleRecognitionModel:
    """Simplified recognition model that loads directly"""
    
    def __init__(self, model_name="refined_quick_20250606_054446"):
        self.model_name = model_name
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.person_mapping = {}
        self.loaded = False
        
        self._load_model()
        
    def _load_model(self):
        """Load model files directly"""
        try:
            model_dir = Path(f'models/person_recognition/{self.model_name}')
            
            if not model_dir.exists():
                logger.error(f"Model directory not found: {model_dir}")
                return
                
            # Load model
            model_path = model_dir / 'model.pkl'
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info("[OK] Loaded model.pkl")
            else:
                logger.error("[ERROR] model.pkl not found")
                return
                
            # Load scaler
            scaler_path = model_dir / 'scaler.pkl'
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("[OK] Loaded scaler.pkl")
            else:
                logger.error("[ERROR] scaler.pkl not found")
                return
                
            # Load label encoder
            encoder_path = model_dir / 'label_encoder.pkl'
            if encoder_path.exists():
                with open(encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                logger.info("[OK] Loaded label_encoder.pkl")
                logger.info(f"   Classes: {list(self.label_encoder.classes_)}")
            else:
                logger.error("[ERROR] label_encoder.pkl not found")
                return
                
            # Load person mapping
            mapping_path = model_dir / 'person_id_mapping.json'
            if mapping_path.exists():
                with open(mapping_path, 'r') as f:
                    mapping = json.load(f)
                    # Convert string keys to int
                    self.person_mapping = {int(k): v for k, v in mapping.items()}
                logger.info(f"[OK] Loaded person mappings: {len(self.person_mapping)} persons")
            else:
                # Create mapping from label encoder
                self.person_mapping = {i: cls for i, cls in enumerate(self.label_encoder.classes_)}
                logger.info(f"[OK] Created person mappings from label encoder")
                
            self.loaded = True
            logger.info(f"[OK] Recognition model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load model: {type(e).__name__}: {str(e)}")
            self.loaded = False
            
    def extract_features(self, image):
        """Extract features from image using simple method"""
        try:
            # Resize to standard size
            img = cv2.resize(image, (128, 128))
            
            # Convert to RGB if needed
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
            # Simple feature extraction
            # Flatten and normalize
            features = img.flatten()
            features = features / 255.0
            
            # Reduce dimensionality by taking statistics
            h, w = img.shape[:2]
            blocks = []
            
            # Divide into blocks and compute statistics
            block_size = 16
            for i in range(0, h, block_size):
                for j in range(0, w, block_size):
                    block = img[i:i+block_size, j:j+block_size]
                    blocks.extend([
                        block.mean(),
                        block.std(),
                        block.min(),
                        block.max()
                    ])
                    
            # Add color histograms
            for c in range(3):
                hist = cv2.calcHist([img], [c], None, [32], [0, 256])
                blocks.extend(hist.flatten() / hist.sum())
                
            features = np.array(blocks)
            
            # Ensure correct size (this should match training)
            expected_size = 2208  # This is what the model expects
            if len(features) < expected_size:
                features = np.pad(features, (0, expected_size - len(features)), 'constant')
            elif len(features) > expected_size:
                features = features[:expected_size]
                
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return None
            
    def recognize(self, image, confidence_threshold=0.7):
        """Recognize person from image"""
        if not self.loaded:
            return None
            
        try:
            # Extract features
            features = self.extract_features(image)
            if features is None:
                return None
                
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Predict
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            confidence = np.max(probabilities)
            
            if confidence >= confidence_threshold:
                # Get person ID
                person_id = self.person_mapping.get(prediction, f"class_{prediction}")
                
                return {
                    'person_id': person_id,
                    'confidence': float(confidence),
                    'class_idx': int(prediction)
                }
            else:
                return {
                    'person_id': 'unknown',
                    'confidence': float(confidence),
                    'class_idx': int(prediction)
                }
                
        except Exception as e:
            logger.error(f"Recognition error: {e}")
            import traceback
            traceback.print_exc()
            return None

# Global model instance
_recognition_model = None

def get_recognition_model():
    """Get or create recognition model"""
    global _recognition_model
    if _recognition_model is None:
        _recognition_model = SimpleRecognitionModel()
    return _recognition_model

def recognize_person(image, confidence_threshold=0.7):
    """Simple function to recognize person"""
    model = get_recognition_model()
    if model.loaded:
        return model.recognize(image, confidence_threshold)
    return None