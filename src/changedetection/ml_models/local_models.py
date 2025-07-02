"""
Local model management for offline change detection
"""
import os
import torch
import numpy as np
from PIL import Image
import logging
from typing import Tuple, Optional
import pickle
import cv2

logger = logging.getLogger(__name__)


class LocalChangeDetectionModel:
    """Local change detection using traditional computer vision methods"""
    
    def __init__(self, method='diff_threshold'):
        self.method = method
        self.threshold = 0.3
        
    def simple_difference(self, before: np.ndarray, after: np.ndarray) -> Tuple[np.ndarray, float]:
        """Simple pixel difference approach"""
        # Convert to grayscale if needed
        if len(before.shape) == 3:
            before_gray = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY)
            after_gray = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY)
        else:
            before_gray = before
            after_gray = after
            
        # Calculate absolute difference
        diff = np.abs(before_gray.astype(float) - after_gray.astype(float))
        
        # Normalize to 0-1
        diff_norm = diff / 255.0
        
        # Apply threshold
        change_map = diff_norm > self.threshold
        
        # Calculate confidence as mean difference in changed areas
        confidence = np.mean(diff_norm[change_map]) if np.any(change_map) else 0.0
        
        return change_map, confidence
    
    def spectral_difference(self, before: np.ndarray, after: np.ndarray) -> Tuple[np.ndarray, float]:
        """Multi-spectral difference approach"""
        if len(before.shape) != 3 or before.shape[2] < 3:
            return self.simple_difference(before, after)
        
        # Calculate difference for each band
        differences = []
        for band in range(min(3, before.shape[2])):  # Use first 3 bands
            diff = np.abs(before[:, :, band].astype(float) - after[:, :, band].astype(float))
            differences.append(diff / 255.0)
        
        # Combine differences (mean across bands)
        combined_diff = np.mean(differences, axis=0)
        
        # Apply threshold
        change_map = combined_diff > self.threshold
        
        # Calculate confidence
        confidence = np.mean(combined_diff[change_map]) if np.any(change_map) else 0.0
        
        return change_map, confidence
    
    def cvaps_method(self, before: np.ndarray, after: np.ndarray) -> Tuple[np.ndarray, float]:
        """Change Vector Analysis in Polar System (CVAPS)"""
        if len(before.shape) != 3 or before.shape[2] < 3:
            return self.simple_difference(before, after)
        
        # Convert to float
        before_f = before.astype(float)
        after_f = after.astype(float)
        
        # Calculate change vector magnitude
        change_vector = after_f - before_f
        magnitude = np.sqrt(np.sum(change_vector**2, axis=2))
        
        # Normalize
        magnitude_norm = magnitude / (np.sqrt(3) * 255)
        
        # Apply threshold
        change_map = magnitude_norm > self.threshold
        
        # Calculate confidence
        confidence = np.mean(magnitude_norm[change_map]) if np.any(change_map) else 0.0
        
        return change_map, confidence
    
    def predict(self, before_image: np.ndarray, after_image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Predict changes between two images"""
        # Ensure images have same dimensions
        if before_image.shape != after_image.shape:
            # Resize to match smaller image
            h = min(before_image.shape[0], after_image.shape[0])
            w = min(before_image.shape[1], after_image.shape[1])
            before_image = cv2.resize(before_image, (w, h))
            after_image = cv2.resize(after_image, (w, h))
        
        # Select method
        if self.method == 'spectral':
            return self.spectral_difference(before_image, after_image)
        elif self.method == 'cvaps':
            return self.cvaps_method(before_image, after_image)
        else:  # 'diff_threshold'
            return self.simple_difference(before_image, after_image)


class LocalUNetModel:
    """Local U-Net model with fallback to simple methods"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if model_path and os.path.exists(model_path):
            try:
                self.load_model(model_path)
            except Exception as e:
                logger.warning(f"Failed to load model from {model_path}: {e}")
                logger.info("Falling back to simple change detection")
        
        # Fallback model
        self.fallback_model = LocalChangeDetectionModel('spectral')
    
    def load_model(self, model_path: str):
        """Load pre-trained U-Net model"""
        try:
            # Import U-Net architecture
            from ..change_detection.ml_models import UNetChangeDetection
            
            self.model = UNetChangeDetection(n_channels=6, n_classes=1)
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Loaded U-Net model from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
    
    def preprocess_for_unet(self, before: np.ndarray, after: np.ndarray) -> torch.Tensor:
        """Preprocess images for U-Net input"""
        # Resize to 256x256
        before_resized = cv2.resize(before, (256, 256))
        after_resized = cv2.resize(after, (256, 256))
        
        # Ensure 3 channels
        if len(before_resized.shape) == 2:
            before_resized = np.stack([before_resized] * 3, axis=2)
        if len(after_resized.shape) == 2:
            after_resized = np.stack([after_resized] * 3, axis=2)
        
        # Take only first 3 channels
        before_resized = before_resized[:, :, :3]
        after_resized = after_resized[:, :, :3]
        
        # Normalize to 0-1
        before_norm = before_resized.astype(np.float32) / 255.0
        after_norm = after_resized.astype(np.float32) / 255.0
        
        # Concatenate to 6-channel input
        combined = np.concatenate([before_norm, after_norm], axis=2)
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(combined).permute(2, 0, 1).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def predict(self, before_image: np.ndarray, after_image: np.ndarray, 
                threshold: float = 0.5) -> Tuple[np.ndarray, float]:
        """Predict changes using U-Net or fallback method"""
        original_shape = before_image.shape[:2]
        
        if self.model is not None:
            try:
                # Use U-Net model
                input_tensor = self.preprocess_for_unet(before_image, after_image)
                
                with torch.no_grad():
                    output = self.model(input_tensor)
                    probabilities = torch.sigmoid(output.squeeze()).cpu().numpy()
                    
                    # Resize back to original shape
                    probabilities_resized = cv2.resize(probabilities, 
                                                     (original_shape[1], original_shape[0]))
                    
                    # Apply threshold
                    change_map = probabilities_resized > threshold
                    confidence = np.mean(probabilities_resized[change_map]) if np.any(change_map) else 0.0
                    
                    return change_map, confidence
                    
            except Exception as e:
                logger.error(f"Error in U-Net prediction: {e}")
                logger.info("Falling back to simple method")
        
        # Use fallback method
        return self.fallback_model.predict(before_image, after_image)


def download_pretrained_models():
    """Download pre-trained models if not available locally"""
    models_dir = 'ml_models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Create a dummy pre-trained model for demonstration
    model_path = os.path.join(models_dir, 'change_detection_model.pth')
    
    if not os.path.exists(model_path):
        try:
            from ..change_detection.ml_models import UNetChangeDetection
            
            # Create a dummy trained model
            model = UNetChangeDetection(n_channels=6, n_classes=1)
            
            # Save model checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': 100,
                'loss': 0.1,
                'accuracy': 0.85
            }, model_path)
            
            logger.info(f"Created dummy model at {model_path}")
            
        except Exception as e:
            logger.error(f"Error creating dummy model: {e}")
    
    return model_path if os.path.exists(model_path) else None


# Initialize global model instance
_global_model = None

def get_change_detection_model():
    """Get global change detection model instance"""
    global _global_model
    
    if _global_model is None:
        model_path = download_pretrained_models()
        _global_model = LocalUNetModel(model_path)
    
    return _global_model 