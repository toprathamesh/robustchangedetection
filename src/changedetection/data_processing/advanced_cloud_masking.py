"""
Advanced Cloud and Shadow Masking System
========================================
State-of-the-art cloud and shadow detection using machine learning,
atmospheric correction, and multi-temporal consistency analysis.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from pathlib import Path
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from scipy import ndimage
from skimage.morphology import binary_opening, binary_closing, disk
from skimage.feature import local_binary_pattern
from skimage.filters import gaussian

logger = logging.getLogger(__name__)


@dataclass
class CloudMaskResult:
    """Result from cloud masking operation"""
    cloud_mask: np.ndarray
    shadow_mask: np.ndarray
    combined_mask: np.ndarray
    confidence_map: np.ndarray
    quality_score: float
    metadata: Dict


class CloudNetModel(nn.Module):
    """
    Deep learning model for cloud detection
    Based on U-Net architecture with attention mechanisms
    """
    
    def __init__(self, input_channels=13, output_classes=3):
        """
        Args:
            input_channels: Number of input channels (all Sentinel-2 bands)
            output_classes: 3 classes (clear, cloud, shadow)
        """
        super(CloudNetModel, self).__init__()
        
        # Encoder
        self.enc1 = self._conv_block(input_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # Bottleneck with attention
        self.bottleneck = self._conv_block(512, 1024)
        self.attention = SpatialAttentionBlock(1024)
        
        # Decoder
        self.dec4 = self._upconv_block(1024, 512)
        self.dec3 = self._upconv_block(512, 256)
        self.dec2 = self._upconv_block(256, 128)
        self.dec1 = self._upconv_block(128, 64)
        
        # Final classifier
        self.final_conv = nn.Conv2d(64, output_classes, kernel_size=1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck with attention
        b = self.bottleneck(self.pool(e4))
        b = self.attention(b)
        
        # Decoder path with skip connections
        d4 = self.dec4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self._conv_block(d4.shape[1], 512)(d4)
        
        d3 = self.dec3(self.upsample(d4))
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self._conv_block(d3.shape[1], 256)(d3)
        
        d2 = self.dec2(self.upsample(d3))
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self._conv_block(d2.shape[1], 128)(d2)
        
        d1 = self.dec1(self.upsample(d2))
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self._conv_block(d1.shape[1], 64)(d1)
        
        # Final output
        output = self.final_conv(d1)
        return output


class SpatialAttentionBlock(nn.Module):
    """Spatial attention mechanism for cloud detection"""
    
    def __init__(self, channels):
        super(SpatialAttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels // 8, 1)
        self.conv2 = nn.Conv2d(channels, channels // 8, 1)
        self.conv3 = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Calculate attention maps
        proj_query = self.conv1(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.conv2(x).view(batch_size, -1, width * height)
        proj_value = self.conv3(x).view(batch_size, -1, width * height)
        
        attention = torch.bmm(proj_query, proj_key)
        attention = F.softmax(attention, dim=-1)
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        return self.gamma * out + x


class AdvancedCloudDetector:
    """
    Advanced cloud detection system using multiple approaches
    """
    
    def __init__(self):
        self.ml_model = None
        self.deep_model = None
        self.feature_scaler = StandardScaler()
        self.is_trained = False
        
        # Cloud detection thresholds and parameters
        self.cloud_threshold = 0.5
        self.shadow_threshold = 0.3
        self.confidence_threshold = 0.7
        
        # Atmospheric correction parameters
        self.atmospheric_params = {
            'water_vapor': 2.5,  # cm
            'ozone': 0.3,        # atm-cm
            'aerosol_optical_depth': 0.1
        }
        
    def extract_features(self, image: np.ndarray, bands_info: Dict) -> np.ndarray:
        """
        Extract comprehensive features for cloud detection
        
        Args:
            image: Multi-spectral satellite image
            bands_info: Information about available bands
            
        Returns:
            Feature array for classification
        """
        features = []
        height, width = image.shape[:2]
        
        # Spectral features
        spectral_features = self._extract_spectral_features(image, bands_info)
        features.extend(spectral_features)
        
        # Textural features
        textural_features = self._extract_textural_features(image)
        features.extend(textural_features)
        
        # Contextual features
        contextual_features = self._extract_contextual_features(image)
        features.extend(contextual_features)
        
        # Geometric features
        geometric_features = self._extract_geometric_features(image)
        features.extend(geometric_features)
        
        return np.array(features).reshape(height * width, -1)
    
    def _extract_spectral_features(self, image: np.ndarray, bands_info: Dict) -> List[np.ndarray]:
        """Extract spectral-based features"""
        features = []
        bands = bands_info.get('bands', {})
        
        # Individual band statistics
        for i in range(min(image.shape[2], 13)):  # Up to 13 Sentinel-2 bands
            band = image[:, :, i]
            features.append(band.flatten())
        
        # Spectral indices for cloud detection
        if 'blue' in bands and 'green' in bands and 'red' in bands and 'nir' in bands:
            blue_idx = bands['blue']
            green_idx = bands['green']
            red_idx = bands['red']
            nir_idx = bands['nir']
            
            if all(idx < image.shape[2] for idx in [blue_idx, green_idx, red_idx, nir_idx]):
                blue = image[:, :, blue_idx]
                green = image[:, :, green_idx]
                red = image[:, :, red_idx]
                nir = image[:, :, nir_idx]
                
                # Cloud indices
                # Whiteness index
                whiteness = np.mean([blue, green, red], axis=0)
                features.append(whiteness.flatten())
                
                # Haze-optimized transformation (HOT)
                hot = blue - 0.5 * red - 0.08
                features.append(hot.flatten())
                
                # Normalized Difference Snow Index (for bright targets)
                ndsi = self._safe_divide(green - nir, green + nir)
                features.append(ndsi.flatten())
        
        # SWIR-based features for thin clouds
        if 'swir1' in bands and 'swir2' in bands:
            swir1_idx = bands['swir1']
            swir2_idx = bands['swir2']
            
            if swir1_idx < image.shape[2] and swir2_idx < image.shape[2]:
                swir1 = image[:, :, swir1_idx]
                swir2 = image[:, :, swir2_idx]
                
                # Cirrus detection (if available)
                if 'cirrus' in bands and bands['cirrus'] < image.shape[2]:
                    cirrus = image[:, :, bands['cirrus']]
                    features.append(cirrus.flatten())
                
                # SWIR ratio for cloud detection
                swir_ratio = self._safe_divide(swir1, swir2)
                features.append(swir_ratio.flatten())
        
        return features
    
    def _extract_textural_features(self, image: np.ndarray) -> List[np.ndarray]:
        """Extract texture-based features"""
        features = []
        
        # Convert to grayscale for texture analysis
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Local Binary Pattern
        lbp = local_binary_pattern(gray, 8, 1, method='uniform')
        features.append(lbp.flatten())
        
        # Gabor filter responses (simplified)
        for theta in [0, 45, 90, 135]:
            kernel = cv2.getGaborKernel((21, 21), 5, np.radians(theta), 2*np.pi*0.25, 0.5, 0, ktype=cv2.CV_32F)
            gabor_response = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
            features.append(gabor_response.flatten())
        
        # Standard deviation in local windows
        kernel = np.ones((9, 9)) / 81
        local_mean = cv2.filter2D(gray, -1, kernel)
        local_std = cv2.filter2D((gray - local_mean)**2, -1, kernel)
        features.append(local_std.flatten())
        
        return features
    
    def _extract_contextual_features(self, image: np.ndarray) -> List[np.ndarray]:
        """Extract contextual features"""
        features = []
        
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Gradient magnitude and direction
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_direction = np.arctan2(grad_y, grad_x)
        
        features.append(gradient_magnitude.flatten())
        features.append(gradient_direction.flatten())
        
        # Distance from edges
        edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
        distance_from_edges = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
        features.append(distance_from_edges.flatten())
        
        return features
    
    def _extract_geometric_features(self, image: np.ndarray) -> List[np.ndarray]:
        """Extract geometric features"""
        features = []
        height, width = image.shape[:2]
        
        # Spatial coordinates (normalized)
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        y_coords_norm = y_coords / height
        x_coords_norm = x_coords / width
        
        features.append(y_coords_norm.flatten())
        features.append(x_coords_norm.flatten())
        
        # Distance from center
        center_y, center_x = height // 2, width // 2
        dist_from_center = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
        dist_from_center_norm = dist_from_center / np.max(dist_from_center)
        features.append(dist_from_center_norm.flatten())
        
        return features
    
    def _safe_divide(self, numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        """Safe division avoiding division by zero"""
        return np.divide(numerator, denominator, 
                        out=np.zeros_like(numerator), 
                        where=denominator!=0)
    
    def train_ml_model(self, 
                      training_images: List[np.ndarray],
                      training_masks: List[np.ndarray],
                      bands_info: Dict,
                      model_type: str = 'random_forest'):
        """
        Train machine learning model for cloud detection
        
        Args:
            training_images: List of training images
            training_masks: List of corresponding cloud masks (0=clear, 1=cloud, 2=shadow)
            bands_info: Information about available bands
            model_type: Type of ML model to train
        """
        logger.info("Extracting features from training data...")
        
        all_features = []
        all_labels = []
        
        for img, mask in zip(training_images, training_masks):
            # Extract features
            features = self.extract_features(img, bands_info)
            labels = mask.flatten()
            
            # Remove invalid pixels
            valid_mask = ~np.isnan(features).any(axis=1) & ~np.isnan(labels)
            
            all_features.append(features[valid_mask])
            all_labels.append(labels[valid_mask])
        
        # Combine all training data
        X = np.vstack(all_features)
        y = np.hstack(all_labels)
        
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        logger.info(f"Training {model_type} model...")
        
        if model_type == 'random_forest':
            self.ml_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=10,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=42
            )
        elif model_type == 'gradient_boosting':
            self.ml_model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Fit model
        self.ml_model.fit(X_train, y_train)
        
        # Validate
        y_pred = self.ml_model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        
        logger.info(f"Validation accuracy: {accuracy:.3f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_val, y_pred, 
                                        target_names=['Clear', 'Cloud', 'Shadow']))
        
        self.is_trained = True
        
        # Save feature importance
        if hasattr(self.ml_model, 'feature_importances_'):
            self.feature_importance = self.ml_model.feature_importances_
    
    def detect_clouds_ml(self, 
                        image: np.ndarray, 
                        bands_info: Dict,
                        return_confidence: bool = True) -> CloudMaskResult:
        """
        Detect clouds using trained ML model
        
        Args:
            image: Input satellite image
            bands_info: Information about available bands
            return_confidence: Whether to return confidence maps
            
        Returns:
            Cloud mask result with confidence information
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_ml_model() first.")
        
        height, width = image.shape[:2]
        
        # Extract features
        features = self.extract_features(image, bands_info)
        
        # Handle invalid features
        valid_mask = ~np.isnan(features).any(axis=1)
        
        # Scale features
        features_scaled = np.zeros_like(features)
        features_scaled[valid_mask] = self.feature_scaler.transform(features[valid_mask])
        
        # Predict
        predictions = np.zeros(features.shape[0])
        confidence_scores = np.zeros(features.shape[0])
        
        if np.any(valid_mask):
            pred_valid = self.ml_model.predict(features_scaled[valid_mask])
            predictions[valid_mask] = pred_valid
            
            if return_confidence and hasattr(self.ml_model, 'predict_proba'):
                proba_valid = self.ml_model.predict_proba(features_scaled[valid_mask])
                confidence_scores[valid_mask] = np.max(proba_valid, axis=1)
        
        # Reshape to image dimensions
        prediction_map = predictions.reshape(height, width)
        confidence_map = confidence_scores.reshape(height, width)
        
        # Create individual masks
        cloud_mask = (prediction_map == 1).astype(np.uint8)
        shadow_mask = (prediction_map == 2).astype(np.uint8)
        combined_mask = ((prediction_map == 1) | (prediction_map == 2)).astype(np.uint8)
        
        # Post-processing
        cloud_mask = self._post_process_mask(cloud_mask, 'cloud')
        shadow_mask = self._post_process_mask(shadow_mask, 'shadow')
        combined_mask = cloud_mask | shadow_mask
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(
            image, cloud_mask, shadow_mask, confidence_map
        )
        
        return CloudMaskResult(
            cloud_mask=cloud_mask.astype(bool),
            shadow_mask=shadow_mask.astype(bool),
            combined_mask=combined_mask.astype(bool),
            confidence_map=confidence_map,
            quality_score=quality_score,
            metadata={
                'method': 'ml_classification',
                'model_type': type(self.ml_model).__name__,
                'cloud_percentage': np.sum(cloud_mask) / cloud_mask.size * 100,
                'shadow_percentage': np.sum(shadow_mask) / shadow_mask.size * 100,
                'mean_confidence': np.mean(confidence_map[valid_mask.reshape(height, width)])
            }
        )
    
    def detect_clouds_deep(self, 
                          image: np.ndarray,
                          model_path: Optional[str] = None) -> CloudMaskResult:
        """
        Detect clouds using deep learning model
        
        Args:
            image: Input satellite image
            model_path: Path to pre-trained deep model
            
        Returns:
            Cloud mask result
        """
        if self.deep_model is None:
            if model_path and Path(model_path).exists():
                self.load_deep_model(model_path)
            else:
                logger.warning("No deep model available, using fallback method")
                return self._detect_clouds_fallback(image)
        
        # Preprocess for deep model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Normalize and reshape
        image_tensor = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(device)
        
        # Predict
        self.deep_model.eval()
        with torch.no_grad():
            outputs = self.deep_model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        
        # Convert back to numpy
        prediction_map = predictions.cpu().numpy().squeeze()
        confidence_map = torch.max(probabilities, dim=1)[0].cpu().numpy().squeeze()
        
        # Create masks
        cloud_mask = (prediction_map == 1)
        shadow_mask = (prediction_map == 2)
        combined_mask = cloud_mask | shadow_mask
        
        # Post-processing
        cloud_mask = self._post_process_mask(cloud_mask.astype(np.uint8), 'cloud')
        shadow_mask = self._post_process_mask(shadow_mask.astype(np.uint8), 'shadow')
        combined_mask = cloud_mask | shadow_mask
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(
            image, cloud_mask, shadow_mask, confidence_map
        )
        
        return CloudMaskResult(
            cloud_mask=cloud_mask.astype(bool),
            shadow_mask=shadow_mask.astype(bool),
            combined_mask=combined_mask.astype(bool),
            confidence_map=confidence_map,
            quality_score=quality_score,
            metadata={
                'method': 'deep_learning',
                'model_type': 'CloudNetModel',
                'cloud_percentage': np.sum(cloud_mask) / cloud_mask.size * 100,
                'shadow_percentage': np.sum(shadow_mask) / shadow_mask.size * 100,
                'mean_confidence': np.mean(confidence_map)
            }
        )
    
    def _detect_clouds_fallback(self, image: np.ndarray) -> CloudMaskResult:
        """Fallback cloud detection using traditional methods"""
        logger.info("Using fallback cloud detection method")
        
        # Simple threshold-based detection
        if len(image.shape) == 3:
            brightness = np.mean(image, axis=2)
        else:
            brightness = image
        
        # Normalize brightness
        brightness_norm = brightness / np.max(brightness)
        
        # Cloud detection (bright areas)
        cloud_mask = brightness_norm > 0.6
        
        # Shadow detection (dark areas)
        shadow_mask = brightness_norm < 0.3
        
        # Combined mask
        combined_mask = cloud_mask | shadow_mask
        
        # Basic post-processing
        cloud_mask = binary_opening(cloud_mask, disk(3))
        shadow_mask = binary_opening(shadow_mask, disk(2))
        combined_mask = cloud_mask | shadow_mask
        
        # Confidence map (distance from thresholds)
        confidence_map = np.ones_like(brightness_norm) * 0.5
        
        return CloudMaskResult(
            cloud_mask=cloud_mask,
            shadow_mask=shadow_mask,
            combined_mask=combined_mask,
            confidence_map=confidence_map,
            quality_score=0.5,
            metadata={
                'method': 'fallback_threshold',
                'cloud_percentage': np.sum(cloud_mask) / cloud_mask.size * 100,
                'shadow_percentage': np.sum(shadow_mask) / shadow_mask.size * 100
            }
        )
    
    def _post_process_mask(self, mask: np.ndarray, mask_type: str) -> np.ndarray:
        """Post-process masks to remove noise and fill gaps"""
        
        if mask_type == 'cloud':
            # Remove small isolated cloud pixels
            mask = binary_opening(mask, disk(2))
            # Fill small gaps in clouds
            mask = binary_closing(mask, disk(3))
        elif mask_type == 'shadow':
            # Remove small shadow pixels
            mask = binary_opening(mask, disk(1))
            # Fill small gaps in shadows
            mask = binary_closing(mask, disk(2))
        
        # Remove very small connected components
        num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
        min_size = 100 if mask_type == 'cloud' else 50
        
        cleaned_mask = np.zeros_like(mask)
        for i in range(1, num_labels):
            component = labels == i
            if np.sum(component) >= min_size:
                cleaned_mask[component] = 1
        
        return cleaned_mask
    
    def _calculate_quality_score(self, 
                                image: np.ndarray,
                                cloud_mask: np.ndarray,
                                shadow_mask: np.ndarray,
                                confidence_map: np.ndarray) -> float:
        """Calculate overall quality score for cloud masking"""
        
        quality_factors = []
        
        # 1. Confidence score
        mean_confidence = np.mean(confidence_map)
        quality_factors.append(mean_confidence)
        
        # 2. Spatial consistency
        cloud_smoothness = self._calculate_smoothness(cloud_mask)
        shadow_smoothness = self._calculate_smoothness(shadow_mask)
        avg_smoothness = (cloud_smoothness + shadow_smoothness) / 2
        quality_factors.append(avg_smoothness)
        
        # 3. Coverage reasonableness (not too much or too little)
        total_coverage = (np.sum(cloud_mask) + np.sum(shadow_mask)) / cloud_mask.size
        coverage_score = 1.0 - abs(total_coverage - 0.2)  # Penalty for extreme coverage
        coverage_score = max(0, min(1, coverage_score))
        quality_factors.append(coverage_score)
        
        # 4. Edge coherence
        edge_score = self._calculate_edge_coherence(image, cloud_mask | shadow_mask)
        quality_factors.append(edge_score)
        
        return np.mean(quality_factors)
    
    def _calculate_smoothness(self, mask: np.ndarray) -> float:
        """Calculate spatial smoothness of mask"""
        # Calculate gradient magnitude
        grad_x = np.abs(np.diff(mask.astype(float), axis=1))
        grad_y = np.abs(np.diff(mask.astype(float), axis=0))
        
        # Average gradient (lower is smoother)
        avg_gradient = (np.mean(grad_x) + np.mean(grad_y)) / 2
        
        # Convert to smoothness score (0-1, higher is better)
        smoothness = 1.0 / (1.0 + avg_gradient * 10)
        
        return smoothness
    
    def _calculate_edge_coherence(self, image: np.ndarray, mask: np.ndarray) -> float:
        """Calculate how well mask edges align with image edges"""
        
        # Calculate image edges
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        image_edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
        
        # Calculate mask edges
        mask_edges = cv2.Canny(mask.astype(np.uint8) * 255, 50, 150)
        
        # Calculate overlap
        overlap = np.sum((image_edges > 0) & (mask_edges > 0))
        total_mask_edges = np.sum(mask_edges > 0)
        
        if total_mask_edges == 0:
            return 1.0
        
        coherence = overlap / total_mask_edges
        return min(1.0, coherence)
    
    def apply_atmospheric_correction(self, 
                                   image: np.ndarray,
                                   bands_info: Dict,
                                   metadata: Dict) -> np.ndarray:
        """
        Apply simplified atmospheric correction
        
        Args:
            image: Input satellite image
            bands_info: Information about available bands
            metadata: Image metadata (sun angle, acquisition date, etc.)
            
        Returns:
            Atmospherically corrected image
        """
        
        # Get solar zenith angle
        solar_zenith = metadata.get('solar_zenith_angle', 30.0)  # degrees
        solar_zenith_rad = np.radians(solar_zenith)
        
        # Rayleigh scattering correction (simplified)
        corrected_image = image.copy().astype(np.float32)
        
        for i in range(image.shape[2]):
            band = corrected_image[:, :, i]
            
            # Simple path radiance removal
            path_radiance = 0.01 * band  # Simplified model
            band_corrected = band - path_radiance
            
            # Solar angle correction
            band_corrected = band_corrected / np.cos(solar_zenith_rad)
            
            corrected_image[:, :, i] = np.clip(band_corrected, 0, 1)
        
        return corrected_image
    
    def multi_temporal_consistency_check(self, 
                                       current_mask: CloudMaskResult,
                                       previous_masks: List[CloudMaskResult],
                                       temporal_weights: Optional[List[float]] = None) -> CloudMaskResult:
        """
        Improve cloud mask using multi-temporal consistency
        
        Args:
            current_mask: Current cloud mask result
            previous_masks: List of previous cloud mask results
            temporal_weights: Weights for temporal integration
            
        Returns:
            Improved cloud mask with temporal consistency
        """
        
        if not previous_masks:
            return current_mask
        
        if temporal_weights is None:
            # Exponentially decreasing weights for older images
            temporal_weights = [0.5**i for i in range(len(previous_masks))]
        
        # Normalize weights
        total_weight = sum(temporal_weights) + 1.0  # +1 for current image
        temporal_weights = [w/total_weight for w in temporal_weights]
        current_weight = 1.0 / total_weight
        
        # Combine masks using weighted average
        combined_cloud_prob = current_mask.cloud_mask.astype(float) * current_weight
        combined_shadow_prob = current_mask.shadow_mask.astype(float) * current_weight
        
        for i, prev_mask in enumerate(previous_masks):
            weight = temporal_weights[i]
            combined_cloud_prob += prev_mask.cloud_mask.astype(float) * weight
            combined_shadow_prob += prev_mask.shadow_mask.astype(float) * weight
        
        # Create improved masks
        improved_cloud_mask = combined_cloud_prob > 0.5
        improved_shadow_mask = combined_shadow_prob > 0.5
        improved_combined_mask = improved_cloud_mask | improved_shadow_mask
        
        # Update confidence based on temporal consistency
        temporal_consistency = self._calculate_temporal_consistency(
            current_mask, previous_masks
        )
        
        improved_confidence = (current_mask.confidence_map + temporal_consistency) / 2
        
        # Calculate new quality score
        improved_quality = (current_mask.quality_score + temporal_consistency.mean()) / 2
        
        return CloudMaskResult(
            cloud_mask=improved_cloud_mask,
            shadow_mask=improved_shadow_mask,
            combined_mask=improved_combined_mask,
            confidence_map=improved_confidence,
            quality_score=improved_quality,
            metadata={
                **current_mask.metadata,
                'temporal_consistency_applied': True,
                'num_previous_masks': len(previous_masks),
                'temporal_consistency_score': temporal_consistency.mean()
            }
        )
    
    def _calculate_temporal_consistency(self, 
                                      current_mask: CloudMaskResult,
                                      previous_masks: List[CloudMaskResult]) -> np.ndarray:
        """Calculate pixel-wise temporal consistency"""
        
        height, width = current_mask.cloud_mask.shape
        consistency_map = np.ones((height, width))
        
        if not previous_masks:
            return consistency_map
        
        current_combined = current_mask.cloud_mask | current_mask.shadow_mask
        
        agreements = []
        for prev_mask in previous_masks:
            prev_combined = prev_mask.cloud_mask | prev_mask.shadow_mask
            
            # Calculate agreement
            agreement = (current_combined == prev_combined).astype(float)
            agreements.append(agreement)
        
        # Average agreement across all previous masks
        if agreements:
            consistency_map = np.mean(agreements, axis=0)
        
        return consistency_map
    
    def save_model(self, filepath: str):
        """Save trained models"""
        model_data = {
            'ml_model': self.ml_model,
            'feature_scaler': self.feature_scaler,
            'is_trained': self.is_trained,
            'atmospheric_params': self.atmospheric_params,
            'cloud_threshold': self.cloud_threshold,
            'shadow_threshold': self.shadow_threshold
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Cloud detection model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained models"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.ml_model = model_data['ml_model']
        self.feature_scaler = model_data['feature_scaler']
        self.is_trained = model_data['is_trained']
        self.atmospheric_params = model_data.get('atmospheric_params', self.atmospheric_params)
        self.cloud_threshold = model_data.get('cloud_threshold', self.cloud_threshold)
        self.shadow_threshold = model_data.get('shadow_threshold', self.shadow_threshold)
        
        logger.info(f"Cloud detection model loaded from {filepath}")
    
    def load_deep_model(self, model_path: str):
        """Load pre-trained deep learning model"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.deep_model = CloudNetModel()
        checkpoint = torch.load(model_path, map_location=device)
        self.deep_model.load_state_dict(checkpoint['model_state_dict'])
        self.deep_model.to(device)
        self.deep_model.eval()
        
        logger.info(f"Deep cloud detection model loaded from {model_path}")


class CloudMaskingPipeline:
    """
    Complete cloud masking pipeline integrating all components
    """
    
    def __init__(self):
        self.detector = AdvancedCloudDetector()
        self.mask_history = []
        self.max_history = 10
        
    def process_image(self, 
                     image: np.ndarray,
                     bands_info: Dict,
                     metadata: Dict,
                     method: str = 'ml',
                     apply_atmospheric_correction: bool = True,
                     use_temporal_consistency: bool = True) -> CloudMaskResult:
        """
        Complete cloud masking pipeline
        
        Args:
            image: Input satellite image
            bands_info: Information about available bands
            metadata: Image metadata
            method: Detection method ('ml', 'deep', 'fallback')
            apply_atmospheric_correction: Whether to apply atmospheric correction
            use_temporal_consistency: Whether to use temporal consistency
            
        Returns:
            Final cloud mask result
        """
        
        # 1. Atmospheric correction
        if apply_atmospheric_correction:
            corrected_image = self.detector.apply_atmospheric_correction(
                image, bands_info, metadata
            )
        else:
            corrected_image = image
        
        # 2. Cloud detection
        if method == 'ml' and self.detector.is_trained:
            mask_result = self.detector.detect_clouds_ml(corrected_image, bands_info)
        elif method == 'deep':
            mask_result = self.detector.detect_clouds_deep(corrected_image)
        else:
            mask_result = self.detector._detect_clouds_fallback(corrected_image)
        
        # 3. Temporal consistency (if enabled and history available)
        if use_temporal_consistency and self.mask_history:
            mask_result = self.detector.multi_temporal_consistency_check(
                mask_result, self.mask_history
            )
        
        # 4. Update history
        self.mask_history.append(mask_result)
        if len(self.mask_history) > self.max_history:
            self.mask_history.pop(0)
        
        return mask_result
    
    def train_pipeline(self, 
                      training_data: List[Tuple[np.ndarray, np.ndarray]],
                      bands_info: Dict):
        """Train the cloud detection pipeline"""
        
        images, masks = zip(*training_data)
        
        self.detector.train_ml_model(
            list(images), 
            list(masks), 
            bands_info,
            model_type='random_forest'
        )
        
        logger.info("Cloud masking pipeline training completed")
    
    def evaluate_performance(self, 
                           test_images: List[np.ndarray],
                           test_masks: List[np.ndarray],
                           bands_info: Dict) -> Dict:
        """Evaluate pipeline performance on test data"""
        
        metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'quality_scores': []
        }
        
        for img, true_mask in zip(test_images, test_masks):
            # Get prediction
            pred_result = self.detector.detect_clouds_ml(img, bands_info)
            pred_mask = pred_result.combined_mask
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support
            
            true_flat = true_mask.flatten()
            pred_flat = pred_mask.flatten()
            
            accuracy = accuracy_score(true_flat, pred_flat)
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_flat, pred_flat, average='binary'
            )
            
            metrics['accuracy'].append(accuracy)
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1_score'].append(f1)
            metrics['quality_scores'].append(pred_result.quality_score)
        
        # Calculate mean metrics
        return {metric: np.mean(values) for metric, values in metrics.items()} 