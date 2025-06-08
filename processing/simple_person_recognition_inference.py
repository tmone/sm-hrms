"""
Simple Person Recognition Inference Wrapper for chunked processing
"""
import logging
from pathlib import Path
import json
import numpy as np
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class SimplePersonRecognitionInference:
    """Wrapper for PersonRecognitionInferenceSimple for chunked processing"""
    
    def __init__(self, model_name: str = None, confidence_threshold: float = 0.85):
        """
        Initialize recognition model
        
        Args:
            model_name: Name of the model to load (None for default)
            confidence_threshold: Minimum confidence for recognition
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.inference = None
        
        # Try to load the actual inference class
        try:
            from hr_management.processing.person_recognition_inference_simple import PersonRecognitionInferenceSimple
            
            # Get default model if not specified
            if not model_name:
                config_path = Path('models/person_recognition/config.json')
                if config_path.exists():
                    with open(config_path) as f:
                        config = json.load(f)
                        model_name = config.get('default_model')
                        
            if model_name:
                self.inference = PersonRecognitionInferenceSimple(
                    model_name, 
                    confidence_threshold=confidence_threshold
                )
                logger.info(f"Loaded recognition model: {model_name}")
            else:
                logger.warning("No default model found in config")
                
        except Exception as e:
            logger.error(f"Failed to load recognition model: {e}")
            self.inference = None
            
    def predict_single(self, image: np.ndarray) -> Dict:
        """
        Predict person ID from a single image/crop
        
        Args:
            image: Person crop image (BGR format)
            
        Returns:
            Dictionary with prediction results
        """
        if self.inference is None:
            return None
            
        try:
            # Extract features
            features = self.inference.feature_extractor.extract_features_from_image(image)
            
            if features is None:
                return None
                
            # Predict
            features_scaled = self.inference.scaler.transform([features])
            predictions = self.inference.model.predict(features_scaled)
            probabilities = self.inference.model.predict_proba(features_scaled)
            
            predicted_idx = predictions[0]
            confidence = np.max(probabilities[0])
            
            if confidence >= self.confidence_threshold:
                # Get person ID from mapping
                person_id = self.inference.person_id_mapping.get(predicted_idx, 'unknown')
                
                return {
                    'person_id': person_id,
                    'confidence': float(confidence),
                    'all_probabilities': probabilities[0].tolist()
                }
            else:
                return {
                    'person_id': 'unknown',
                    'confidence': float(confidence),
                    'all_probabilities': probabilities[0].tolist()
                }
                
        except Exception as e:
            logger.error(f"Recognition error: {e}")
            return None
            
    def process_batch(self, images: List[np.ndarray]) -> List[Dict]:
        """
        Process multiple images in batch
        
        Args:
            images: List of person crop images
            
        Returns:
            List of prediction results
        """
        results = []
        for image in images:
            result = self.predict_single(image)
            results.append(result)
        return results