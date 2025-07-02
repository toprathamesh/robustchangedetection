"""
Image preprocessing for satellite data
"""
import numpy as np
import cv2
from PIL import Image
import os
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class CloudShadowMask:
    """Cloud and shadow detection and masking"""
    
    def __init__(self):
        self.cloud_threshold = 0.8
        self.shadow_threshold = 0.3
    
    def detect_clouds(self, image: np.ndarray) -> np.ndarray:
        """Detect clouds using brightness threshold"""
        # Convert to grayscale if RGB
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Normalize to 0-1 range
        normalized = gray.astype(np.float32) / 255.0
        
        # Apply threshold
        cloud_mask = normalized > self.cloud_threshold
        
        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cloud_mask = cv2.morphologyEx(cloud_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        return cloud_mask.astype(bool)
    
    def detect_shadows(self, image: np.ndarray) -> np.ndarray:
        """Detect shadows using darkness threshold"""
        # Convert to grayscale if RGB
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Normalize to 0-1 range
        normalized = gray.astype(np.float32) / 255.0
        
        # Apply threshold
        shadow_mask = normalized < self.shadow_threshold
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        shadow_mask = cv2.morphologyEx(shadow_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        return shadow_mask.astype(bool)
    
    def process(self, image_path: str) -> str:
        """Process image to mask clouds and shadows"""
        try:
            # Load image
            image = np.array(Image.open(image_path))
            
            # Detect clouds and shadows
            cloud_mask = self.detect_clouds(image)
            shadow_mask = self.detect_shadows(image)
            
            # Combine masks
            combined_mask = cloud_mask | shadow_mask
            
            # Apply mask to image
            masked_image = image.copy()
            if len(image.shape) == 3:
                masked_image[combined_mask] = [0, 0, 0]  # Set masked pixels to black
            else:
                masked_image[combined_mask] = 0
            
            # Save masked image
            base_name = os.path.splitext(image_path)[0]
            output_path = f"{base_name}_masked.tiff"
            Image.fromarray(masked_image).save(output_path)
            
            logger.info(f"Cloud/shadow masking completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error in cloud/shadow masking: {str(e)}")
            raise e


class RadiometricNormalization:
    """Radiometric normalization for satellite images"""
    
    def __init__(self):
        self.method = 'histogram_matching'
    
    def histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """Apply histogram equalization"""
        if len(image.shape) == 3:
            # Apply to each channel
            result = np.zeros_like(image)
            for i in range(image.shape[2]):
                result[:, :, i] = cv2.equalizeHist(image[:, :, i])
            return result
        else:
            return cv2.equalizeHist(image)
    
    def percentile_stretch(self, image: np.ndarray, lower_percentile: float = 2, 
                          upper_percentile: float = 98) -> np.ndarray:
        """Apply percentile stretching"""
        result = image.copy().astype(np.float32)
        
        if len(image.shape) == 3:
            for i in range(image.shape[2]):
                band = result[:, :, i]
                p_low = np.percentile(band[band > 0], lower_percentile)
                p_high = np.percentile(band[band > 0], upper_percentile)
                
                # Stretch
                band = (band - p_low) / (p_high - p_low) * 255
                band = np.clip(band, 0, 255)
                result[:, :, i] = band
        else:
            p_low = np.percentile(result[result > 0], lower_percentile)
            p_high = np.percentile(result[result > 0], upper_percentile)
            result = (result - p_low) / (p_high - p_low) * 255
            result = np.clip(result, 0, 255)
        
        return result.astype(np.uint8)
    
    def process(self, image_path: str) -> str:
        """Apply radiometric normalization"""
        try:
            # Load image
            image = np.array(Image.open(image_path))
            
            # Apply normalization
            if self.method == 'histogram_equalization':
                normalized_image = self.histogram_equalization(image)
            else:  # percentile_stretch
                normalized_image = self.percentile_stretch(image)
            
            # Save normalized image
            base_name = os.path.splitext(image_path)[0]
            output_path = f"{base_name}_normalized.tiff"
            Image.fromarray(normalized_image).save(output_path)
            
            logger.info(f"Radiometric normalization completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error in radiometric normalization: {str(e)}")
            raise e


class SpectralIndices:
    """Calculate spectral indices for change detection"""
    
    @staticmethod
    def ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """Calculate NDVI"""
        with np.errstate(divide='ignore', invalid='ignore'):
            ndvi = (nir - red) / (nir + red)
            ndvi[np.isnan(ndvi)] = 0
            return ndvi
    
    @staticmethod
    def ndbi(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """Calculate NDBI (Normalized Difference Built-up Index)"""
        with np.errstate(divide='ignore', invalid='ignore'):
            ndbi = (swir - nir) / (swir + nir)
            ndbi[np.isnan(ndbi)] = 0
            return ndbi
    
    @staticmethod
    def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """Calculate NDWI (Normalized Difference Water Index)"""
        with np.errstate(divide='ignore', invalid='ignore'):
            ndwi = (green - nir) / (green + nir)
            ndwi[np.isnan(ndwi)] = 0
            return ndwi 