"""
Explainability and Confidence Framework for Change Detection
===========================================================
Advanced framework for providing confidence scores and explanations
for change detection results, enhancing transparency and trust.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
import shap
import lime
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
import cv2
from scipy import stats
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceMetrics:
    """Comprehensive confidence metrics for change detection"""
    overall_confidence: float
    spatial_confidence: np.ndarray
    temporal_confidence: float
    spectral_confidence: float
    model_uncertainty: float
    data_quality_score: float
    change_type_confidence: Dict[str, float]
    reliability_factors: Dict[str, float]


@dataclass
class ExplanationReport:
    """Detailed explanation report for change detection result"""
    detection_summary: Dict
    confidence_breakdown: ConfidenceMetrics
    feature_importance: Dict[str, float]
    spatial_explanations: Dict[str, np.ndarray]
    uncertainty_analysis: Dict
    quality_assessment: Dict
    recommendations: List[str]
    metadata: Dict


class ConfidenceEstimator:
    """
    Advanced confidence estimation for change detection results
    
    Combines multiple sources of uncertainty:
    - Model prediction uncertainty
    - Data quality assessment
    - Spatial consistency
    - Temporal coherence
    - Spectral reliability
    """
    
    def __init__(self):
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }
        
        self.quality_weights = {
            'model_uncertainty': 0.3,
            'data_quality': 0.25,
            'spatial_consistency': 0.2,
            'temporal_coherence': 0.15,
            'spectral_reliability': 0.1
        }
    
    def calculate_confidence(self, 
                           model_outputs: Dict,
                           input_data: Dict,
                           temporal_context: Optional[Dict] = None) -> ConfidenceMetrics:
        """
        Calculate comprehensive confidence metrics
        
        Args:
            model_outputs: Output from change detection model
            input_data: Input images and metadata
            temporal_context: Optional temporal analysis context
            
        Returns:
            Comprehensive confidence metrics
        """
        
        # 1. Model prediction uncertainty
        model_uncertainty = self._calculate_model_uncertainty(model_outputs)
        
        # 2. Data quality assessment
        data_quality = self._assess_data_quality(input_data)
        
        # 3. Spatial consistency
        spatial_consistency = self._calculate_spatial_consistency(model_outputs)
        
        # 4. Temporal coherence (if available)
        temporal_coherence = self._calculate_temporal_coherence(temporal_context) if temporal_context else 0.5
        
        # 5. Spectral reliability
        spectral_reliability = self._assess_spectral_reliability(input_data)
        
        # Calculate weighted overall confidence
        overall_confidence = (
            self.quality_weights['model_uncertainty'] * (1 - model_uncertainty) +
            self.quality_weights['data_quality'] * data_quality +
            self.quality_weights['spatial_consistency'] * spatial_consistency +
            self.quality_weights['temporal_coherence'] * temporal_coherence +
            self.quality_weights['spectral_reliability'] * spectral_reliability
        )
        
        # Spatial confidence map
        spatial_confidence = self._generate_spatial_confidence_map(
            model_outputs, input_data
        )
        
        # Change type specific confidence
        change_type_confidence = self._calculate_type_confidence(model_outputs)
        
        # Reliability factors
        reliability_factors = {
            'model_uncertainty': 1 - model_uncertainty,
            'data_quality': data_quality,
            'spatial_consistency': spatial_consistency,
            'temporal_coherence': temporal_coherence,
            'spectral_reliability': spectral_reliability
        }
        
        return ConfidenceMetrics(
            overall_confidence=overall_confidence,
            spatial_confidence=spatial_confidence,
            temporal_confidence=temporal_coherence,
            spectral_confidence=spectral_reliability,
            model_uncertainty=model_uncertainty,
            data_quality_score=data_quality,
            change_type_confidence=change_type_confidence,
            reliability_factors=reliability_factors
        )
    
    def _calculate_model_uncertainty(self, model_outputs: Dict) -> float:
        """Calculate model prediction uncertainty"""
        try:
            # Entropy-based uncertainty for classification
            if 'type_probabilities' in model_outputs:
                probs = model_outputs['type_probabilities']
                if isinstance(probs, np.ndarray):
                    # Calculate entropy
                    entropy = -np.sum(probs * np.log(probs + 1e-8), axis=-1)
                    max_entropy = np.log(probs.shape[-1])  # Maximum possible entropy
                    normalized_entropy = np.mean(entropy) / max_entropy
                    return normalized_entropy
            
            # Variance-based uncertainty for binary classification
            if 'change_probability' in model_outputs:
                change_prob = model_outputs['change_probability']
                if isinstance(change_prob, np.ndarray):
                    # Maximum uncertainty at 0.5, minimum at 0 or 1
                    uncertainty = 4 * change_prob * (1 - change_prob)
                    return np.mean(uncertainty)
            
            return 0.5  # Default moderate uncertainty
            
        except Exception as e:
            logger.warning(f"Error calculating model uncertainty: {e}")
            return 0.5
    
    def _assess_data_quality(self, input_data: Dict) -> float:
        """Assess quality of input data"""
        quality_scores = []
        
        # Image quality metrics
        for key in ['before_image', 'after_image']:
            if key in input_data:
                image = input_data[key]
                if isinstance(image, np.ndarray):
                    # Check for saturation
                    saturation_ratio = np.sum(image >= 255) / image.size
                    quality_scores.append(1.0 - min(saturation_ratio * 10, 1.0))
                    
                    # Check for low contrast
                    contrast = np.std(image)
                    contrast_score = min(contrast / 50.0, 1.0)  # Normalize to reasonable range
                    quality_scores.append(contrast_score)
                    
                    # Check for noise (using Laplacian variance)
                    if len(image.shape) >= 2:
                        gray = np.mean(image, axis=-1) if len(image.shape) == 3 else image
                        noise_variance = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var()
                        noise_score = min(noise_variance / 1000.0, 1.0)
                        quality_scores.append(noise_score)
        
        # Cloud coverage assessment
        if 'cloud_coverage' in input_data:
            cloud_coverage = input_data['cloud_coverage']
            cloud_score = max(0, 1.0 - cloud_coverage / 20.0)  # Penalize >20% cloud coverage
            quality_scores.append(cloud_score)
        
        # Temporal separation assessment
        if 'temporal_gap_days' in input_data:
            gap_days = input_data['temporal_gap_days']
            # Optimal gap is 30-365 days
            if 30 <= gap_days <= 365:
                temporal_score = 1.0
            elif gap_days < 30:
                temporal_score = gap_days / 30.0
            else:
                temporal_score = max(0.1, 1.0 - (gap_days - 365) / 1000.0)
            quality_scores.append(temporal_score)
        
        return np.mean(quality_scores) if quality_scores else 0.5
    
    def _calculate_spatial_consistency(self, model_outputs: Dict) -> float:
        """Calculate spatial consistency of predictions"""
        try:
            if 'change_probability' in model_outputs:
                change_map = model_outputs['change_probability']
                if isinstance(change_map, np.ndarray) and len(change_map.shape) >= 2:
                    # Calculate local variance to assess consistency
                    kernel = np.ones((5, 5)) / 25
                    local_mean = cv2.filter2D(change_map, -1, kernel)
                    local_variance = cv2.filter2D((change_map - local_mean)**2, -1, kernel)
                    
                    # Lower variance indicates higher consistency
                    consistency = 1.0 - np.mean(local_variance)
                    return max(0, min(1, consistency))
            
            return 0.5
            
        except Exception as e:
            logger.warning(f"Error calculating spatial consistency: {e}")
            return 0.5
    
    def _calculate_temporal_coherence(self, temporal_context: Dict) -> float:
        """Calculate temporal coherence of change detection"""
        if not temporal_context:
            return 0.5
        
        coherence_scores = []
        
        # Check consistency with seasonal patterns
        if 'seasonal_expected' in temporal_context:
            expected = temporal_context['seasonal_expected']
            observed = temporal_context.get('observed_change', 0)
            
            # High coherence if observed matches expected seasonal pattern
            if expected == 0:  # No seasonal change expected
                coherence = 1.0 - abs(observed)
            else:
                coherence = 1.0 - abs(observed - expected) / max(abs(expected), 1)
            coherence_scores.append(max(0, min(1, coherence)))
        
        # Check trend consistency
        if 'trend_consistency' in temporal_context:
            coherence_scores.append(temporal_context['trend_consistency'])
        
        # Check anomaly score
        if 'anomaly_score' in temporal_context:
            anomaly_score = temporal_context['anomaly_score']
            # Lower anomaly score indicates higher coherence with historical patterns
            coherence_scores.append(1.0 - min(abs(anomaly_score), 1.0))
        
        return np.mean(coherence_scores) if coherence_scores else 0.5
    
    def _assess_spectral_reliability(self, input_data: Dict) -> float:
        """Assess reliability based on spectral characteristics"""
        reliability_scores = []
        
        # Check spectral index quality
        if 'spectral_indices' in input_data:
            indices = input_data['spectral_indices']
            for name, values in indices.items():
                if isinstance(values, np.ndarray):
                    # Check for reasonable value ranges
                    valid_ratio = np.sum(np.isfinite(values)) / values.size
                    reliability_scores.append(valid_ratio)
                    
                    # Check for expected value ranges
                    if name == 'ndvi':
                        in_range = np.sum((values >= -1) & (values <= 1)) / values.size
                        reliability_scores.append(in_range)
        
        # Check band availability
        if 'available_bands' in input_data:
            bands = input_data['available_bands']
            essential_bands = ['red', 'nir', 'green']
            available_essential = sum(1 for band in essential_bands if band in bands)
            reliability_scores.append(available_essential / len(essential_bands))
        
        return np.mean(reliability_scores) if reliability_scores else 0.5
    
    def _generate_spatial_confidence_map(self, 
                                       model_outputs: Dict, 
                                       input_data: Dict) -> np.ndarray:
        """Generate pixel-wise confidence map"""
        try:
            # Start with prediction confidence
            if 'change_probability' in model_outputs:
                change_prob = model_outputs['change_probability']
                # Convert probability to confidence (higher at extremes)
                base_confidence = 1 - 4 * change_prob * (1 - change_prob)
            else:
                base_confidence = np.ones((256, 256)) * 0.5
            
            # Adjust based on local data quality
            if 'before_image' in input_data and 'after_image' in input_data:
                before_img = input_data['before_image']
                after_img = input_data['after_image']
                
                if isinstance(before_img, np.ndarray) and isinstance(after_img, np.ndarray):
                    # Local contrast adjustment
                    before_gray = np.mean(before_img, axis=-1) if len(before_img.shape) == 3 else before_img
                    after_gray = np.mean(after_img, axis=-1) if len(after_img.shape) == 3 else after_img
                    
                    # Calculate local standard deviation
                    kernel = np.ones((9, 9)) / 81
                    before_local_std = cv2.filter2D((before_gray - cv2.filter2D(before_gray, -1, kernel))**2, -1, kernel)
                    after_local_std = cv2.filter2D((after_gray - cv2.filter2D(after_gray, -1, kernel))**2, -1, kernel)
                    
                    # Higher standard deviation indicates more texture/detail
                    local_quality = (before_local_std + after_local_std) / 2
                    local_quality = np.clip(local_quality / np.percentile(local_quality, 95), 0, 1)
                    
                    # Combine with base confidence
                    if base_confidence.shape == local_quality.shape:
                        base_confidence = 0.7 * base_confidence + 0.3 * local_quality
            
            return np.clip(base_confidence, 0, 1)
            
        except Exception as e:
            logger.warning(f"Error generating spatial confidence map: {e}")
            return np.ones((256, 256)) * 0.5
    
    def _calculate_type_confidence(self, model_outputs: Dict) -> Dict[str, float]:
        """Calculate confidence for each change type"""
        if 'type_probabilities' in model_outputs:
            probs = model_outputs['type_probabilities']
            if isinstance(probs, np.ndarray):
                # Use the probabilities directly as confidence
                change_types = ['no_change', 'urban_development', 'deforestation', 
                              'mining', 'agriculture_expansion', 'infrastructure']
                
                if len(probs.shape) == 1:  # Single prediction
                    return {change_types[i]: float(probs[i]) for i in range(min(len(probs), len(change_types)))}
                else:  # Spatial predictions
                    return {change_types[i]: float(np.mean(probs[..., i])) 
                           for i in range(min(probs.shape[-1], len(change_types)))}
        
        return {'unknown': 0.5}
    
    def interpret_confidence_level(self, confidence: float) -> Tuple[str, str]:
        """Interpret confidence level with description"""
        if confidence >= self.confidence_thresholds['high']:
            return 'HIGH', 'Very reliable detection with strong evidence'
        elif confidence >= self.confidence_thresholds['medium']:
            return 'MEDIUM', 'Moderately reliable detection, consider additional verification'
        elif confidence >= self.confidence_thresholds['low']:
            return 'LOW', 'Low reliability detection, requires manual verification'
        else:
            return 'VERY_LOW', 'Unreliable detection, likely false positive'


class ExplainabilityEngine:
    """
    Advanced explainability engine for change detection models
    
    Provides multiple explanation methods:
    - LIME for local interpretability
    - SHAP for feature importance
    - Gradient-based attribution
    - Custom spatial explanations
    """
    
    def __init__(self):
        self.explanation_methods = ['lime', 'gradients', 'feature_importance', 'spatial_analysis']
        
    def generate_explanation(self, 
                           model,
                           before_image: np.ndarray,
                           after_image: np.ndarray,
                           model_outputs: Dict,
                           confidence_metrics: ConfidenceMetrics,
                           methods: Optional[List[str]] = None) -> ExplanationReport:
        """
        Generate comprehensive explanation for change detection result
        
        Args:
            model: Trained change detection model
            before_image: Before image
            after_image: After image
            model_outputs: Model prediction outputs
            confidence_metrics: Confidence assessment
            methods: Explanation methods to use
            
        Returns:
            Comprehensive explanation report
        """
        
        if methods is None:
            methods = self.explanation_methods
        
        explanations = {}
        
        # LIME explanation
        if 'lime' in methods:
            try:
                lime_explanation = self._generate_lime_explanation(
                    model, before_image, after_image
                )
                explanations['lime'] = lime_explanation
            except Exception as e:
                logger.warning(f"LIME explanation failed: {e}")
        
        # Gradient-based explanations
        if 'gradients' in methods:
            try:
                gradient_explanation = self._generate_gradient_explanation(
                    model, before_image, after_image, model_outputs
                )
                explanations['gradients'] = gradient_explanation
            except Exception as e:
                logger.warning(f"Gradient explanation failed: {e}")
        
        # Feature importance
        if 'feature_importance' in methods:
            feature_importance = self._calculate_feature_importance(model_outputs)
            explanations['feature_importance'] = feature_importance
        
        # Spatial analysis
        if 'spatial_analysis' in methods:
            spatial_analysis = self._generate_spatial_explanation(
                before_image, after_image, model_outputs
            )
            explanations['spatial_analysis'] = spatial_analysis
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            model_outputs, confidence_metrics, explanations
        )
        
        # Uncertainty analysis
        uncertainty_analysis = self._analyze_uncertainty(
            model_outputs, confidence_metrics
        )
        
        # Quality assessment
        quality_assessment = self._assess_explanation_quality(explanations)
        
        # Detection summary
        detection_summary = self._create_detection_summary(model_outputs, confidence_metrics)
        
        return ExplanationReport(
            detection_summary=detection_summary,
            confidence_breakdown=confidence_metrics,
            feature_importance=explanations.get('feature_importance', {}),
            spatial_explanations=explanations.get('spatial_analysis', {}),
            uncertainty_analysis=uncertainty_analysis,
            quality_assessment=quality_assessment,
            recommendations=recommendations,
            metadata={
                'timestamp': datetime.now().isoformat(),
                'methods_used': methods,
                'image_shapes': {
                    'before': before_image.shape,
                    'after': after_image.shape
                }
            }
        )
    
    def _generate_lime_explanation(self, model, before_image: np.ndarray, after_image: np.ndarray) -> Dict:
        """Generate LIME explanation for change detection"""
        try:
            # Combine images for LIME
            combined_image = np.concatenate([before_image, after_image], axis=-1)
            
            # Create LIME explainer
            explainer = lime_image.LimeImageExplainer()
            
            # Define prediction function for LIME
            def predict_fn(images):
                results = []
                for img in images:
                    # Split back into before/after
                    channels = img.shape[-1] // 2
                    before = img[..., :channels]
                    after = img[..., channels:]
                    
                    # Get model prediction (simplified)
                    if hasattr(model, 'predict_change'):
                        result = model.predict_change(before, after)
                        if hasattr(result, 'change_probability'):
                            prob = result.change_probability
                        else:
                            prob = 0.5
                    else:
                        prob = 0.5
                    
                    results.append([1-prob, prob])  # [no_change, change]
                
                return np.array(results)
            
            # Generate explanation
            explanation = explainer.explain_instance(
                combined_image, 
                predict_fn,
                top_labels=2,
                hide_color=0,
                num_samples=100
            )
            
            # Extract explanation data
            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0], 
                positive_only=True,
                num_features=10,
                hide_rest=False
            )
            
            return {
                'explanation_image': temp,
                'importance_mask': mask,
                'top_features': explanation.top_labels,
                'local_explanation': explanation.local_exp
            }
            
        except Exception as e:
            logger.error(f"LIME explanation error: {e}")
            return {'error': str(e)}
    
    def _generate_gradient_explanation(self, 
                                     model, 
                                     before_image: np.ndarray, 
                                     after_image: np.ndarray,
                                     model_outputs: Dict) -> Dict:
        """Generate gradient-based explanation"""
        try:
            import torch
            
            if not hasattr(model, 'parameters'):  # Not a PyTorch model
                return {'error': 'Gradient explanation requires PyTorch model'}
            
            # Convert to tensors
            before_tensor = torch.tensor(before_image, dtype=torch.float32, requires_grad=True)
            after_tensor = torch.tensor(after_image, dtype=torch.float32, requires_grad=True)
            
            # Ensure correct dimensions
            if len(before_tensor.shape) == 3:
                before_tensor = before_tensor.permute(2, 0, 1).unsqueeze(0)
                after_tensor = after_tensor.permute(2, 0, 1).unsqueeze(0)
            
            # Forward pass
            model.eval()
            outputs = model(before_tensor, after_tensor)
            
            # Calculate gradients
            loss = outputs['change_probability'].sum()
            loss.backward()
            
            # Extract gradients
            before_gradients = before_tensor.grad.detach().numpy()
            after_gradients = after_tensor.grad.detach().numpy()
            
            # Calculate saliency maps
            before_saliency = np.max(np.abs(before_gradients), axis=1).squeeze()
            after_saliency = np.max(np.abs(after_gradients), axis=1).squeeze()
            
            return {
                'before_saliency': before_saliency,
                'after_saliency': after_saliency,
                'gradient_magnitude': np.sqrt(before_saliency**2 + after_saliency**2),
                'gradient_direction': np.arctan2(after_saliency, before_saliency)
            }
            
        except Exception as e:
            logger.error(f"Gradient explanation error: {e}")
            return {'error': str(e)}
    
    def _calculate_feature_importance(self, model_outputs: Dict) -> Dict[str, float]:
        """Calculate feature importance from model outputs"""
        importance = {}
        
        # Use attention weights if available
        if 'before_attention' in model_outputs and 'after_attention' in model_outputs:
            before_attention = model_outputs['before_attention']
            after_attention = model_outputs['after_attention']
            
            # Channel attention importance
            if 'channel' in before_attention and 'channel' in after_attention:
                before_channel = before_attention['channel']
                after_channel = after_attention['channel']
                
                if isinstance(before_channel, np.ndarray) and isinstance(after_channel, np.ndarray):
                    # Average channel importance
                    avg_importance = (np.mean(before_channel) + np.mean(after_channel)) / 2
                    importance['channel_attention'] = float(avg_importance)
            
            # Spatial attention importance
            if 'spatial' in before_attention and 'spatial' in after_attention:
                before_spatial = before_attention['spatial']
                after_spatial = after_attention['spatial']
                
                if isinstance(before_spatial, np.ndarray) and isinstance(after_spatial, np.ndarray):
                    avg_spatial = (np.mean(before_spatial) + np.mean(after_spatial)) / 2
                    importance['spatial_attention'] = float(avg_spatial)
        
        # Use feature vector norms if available
        if 'features' in model_outputs:
            features = model_outputs['features']
            if isinstance(features, np.ndarray):
                # Group features and calculate importance
                feature_groups = {
                    'low_level_features': features[:len(features)//4],
                    'mid_level_features': features[len(features)//4:len(features)//2],
                    'high_level_features': features[len(features)//2:3*len(features)//4],
                    'semantic_features': features[3*len(features)//4:]
                }
                
                for group_name, group_features in feature_groups.items():
                    if len(group_features) > 0:
                        importance[group_name] = float(np.mean(np.abs(group_features)))
        
        # Normalize importance scores
        if importance:
            total = sum(importance.values())
            if total > 0:
                importance = {k: v/total for k, v in importance.items()}
        
        return importance
    
    def _generate_spatial_explanation(self, 
                                    before_image: np.ndarray, 
                                    after_image: np.ndarray, 
                                    model_outputs: Dict) -> Dict:
        """Generate spatial explanations for change detection"""
        explanations = {}
        
        # Calculate difference maps
        if len(before_image.shape) == 3 and len(after_image.shape) == 3:
            # Per-channel differences
            channel_diffs = []
            for i in range(min(before_image.shape[2], after_image.shape[2])):
                diff = after_image[:, :, i].astype(float) - before_image[:, :, i].astype(float)
                channel_diffs.append(diff)
            
            explanations['channel_differences'] = np.stack(channel_diffs, axis=-1)
            
            # Magnitude and direction of change
            magnitude = np.sqrt(np.sum(np.array(channel_diffs)**2, axis=0))
            explanations['change_magnitude'] = magnitude
        
        # Edge analysis
        if len(before_image.shape) >= 2:
            # Convert to grayscale for edge detection
            before_gray = np.mean(before_image, axis=-1) if len(before_image.shape) == 3 else before_image
            after_gray = np.mean(after_image, axis=-1) if len(after_image.shape) == 3 else after_image
            
            # Sobel edge detection
            before_edges = cv2.Sobel(before_gray.astype(np.uint8), cv2.CV_64F, 1, 1, ksize=3)
            after_edges = cv2.Sobel(after_gray.astype(np.uint8), cv2.CV_64F, 1, 1, ksize=3)
            
            edge_change = np.abs(after_edges - before_edges)
            explanations['edge_changes'] = edge_change
        
        # Texture analysis using Local Binary Patterns
        try:
            from skimage.feature import local_binary_pattern
            
            if len(before_image.shape) >= 2:
                before_gray = np.mean(before_image, axis=-1) if len(before_image.shape) == 3 else before_image
                after_gray = np.mean(after_image, axis=-1) if len(after_image.shape) == 3 else after_image
                
                before_lbp = local_binary_pattern(before_gray, 8, 1, method='uniform')
                after_lbp = local_binary_pattern(after_gray, 8, 1, method='uniform')
                
                texture_change = np.abs(after_lbp - before_lbp)
                explanations['texture_changes'] = texture_change
        except ImportError:
            logger.warning("scikit-image not available for texture analysis")
        
        return explanations
    
    def _generate_recommendations(self, 
                                model_outputs: Dict,
                                confidence_metrics: ConfidenceMetrics,
                                explanations: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Confidence-based recommendations
        if confidence_metrics.overall_confidence < 0.4:
            recommendations.append("Low confidence detection - recommend manual verification")
            recommendations.append("Consider acquiring additional imagery with better conditions")
        
        if confidence_metrics.data_quality_score < 0.5:
            recommendations.append("Poor data quality detected - check for cloud cover, shadows, or sensor issues")
        
        if confidence_metrics.model_uncertainty > 0.7:
            recommendations.append("High model uncertainty - consider ensemble prediction or additional features")
        
        # Spatial consistency recommendations
        spatial_var = np.var(confidence_metrics.spatial_confidence)
        if spatial_var > 0.1:
            recommendations.append("High spatial variability in confidence - focus on high-confidence regions")
        
        # Change type recommendations
        max_type_conf = max(confidence_metrics.change_type_confidence.values())
        if max_type_conf < 0.6:
            recommendations.append("Uncertain change type classification - consider additional spectral analysis")
        
        # Temporal recommendations
        if confidence_metrics.temporal_confidence < 0.5:
            recommendations.append("Change inconsistent with temporal patterns - verify against seasonal baselines")
        
        return recommendations
    
    def _analyze_uncertainty(self, 
                           model_outputs: Dict, 
                           confidence_metrics: ConfidenceMetrics) -> Dict:
        """Analyze sources of uncertainty"""
        uncertainty_sources = {}
        
        # Epistemic uncertainty (model uncertainty)
        uncertainty_sources['epistemic'] = confidence_metrics.model_uncertainty
        
        # Aleatoric uncertainty (data uncertainty)
        uncertainty_sources['aleatoric'] = 1.0 - confidence_metrics.data_quality_score
        
        # Spatial uncertainty
        if hasattr(confidence_metrics.spatial_confidence, 'std'):
            spatial_uncertainty = np.std(confidence_metrics.spatial_confidence)
        else:
            spatial_uncertainty = 0.1
        uncertainty_sources['spatial'] = spatial_uncertainty
        
        # Total uncertainty
        total_uncertainty = np.sqrt(
            uncertainty_sources['epistemic']**2 + 
            uncertainty_sources['aleatoric']**2 + 
            uncertainty_sources['spatial']**2
        )
        uncertainty_sources['total'] = total_uncertainty
        
        # Uncertainty interpretation
        if total_uncertainty < 0.2:
            interpretation = "Low uncertainty - high confidence in result"
        elif total_uncertainty < 0.5:
            interpretation = "Moderate uncertainty - result likely correct"
        elif total_uncertainty < 0.8:
            interpretation = "High uncertainty - result requires verification"
        else:
            interpretation = "Very high uncertainty - result unreliable"
        
        uncertainty_sources['interpretation'] = interpretation
        
        return uncertainty_sources
    
    def _assess_explanation_quality(self, explanations: Dict) -> Dict:
        """Assess quality of generated explanations"""
        quality_metrics = {}
        
        # Coverage - how many explanation methods succeeded
        total_methods = len(self.explanation_methods)
        successful_methods = sum(1 for method in self.explanation_methods 
                               if method in explanations and 'error' not in explanations[method])
        quality_metrics['coverage'] = successful_methods / total_methods
        
        # Consistency - check if different methods agree
        if 'lime' in explanations and 'gradients' in explanations:
            # Compare LIME and gradient explanations (simplified)
            quality_metrics['method_consistency'] = 0.7  # Placeholder
        
        # Completeness - check if all necessary components are present
        required_components = ['spatial_analysis', 'feature_importance']
        completeness = sum(1 for comp in required_components if comp in explanations) / len(required_components)
        quality_metrics['completeness'] = completeness
        
        # Overall quality score
        quality_metrics['overall'] = np.mean([
            quality_metrics['coverage'],
            quality_metrics.get('method_consistency', 0.5),
            quality_metrics['completeness']
        ])
        
        return quality_metrics
    
    def _create_detection_summary(self, 
                                model_outputs: Dict, 
                                confidence_metrics: ConfidenceMetrics) -> Dict:
        """Create high-level summary of detection"""
        summary = {}
        
        # Change detection result
        if 'change_probability' in model_outputs:
            change_prob = model_outputs['change_probability']
            if isinstance(change_prob, np.ndarray):
                summary['change_detected'] = np.mean(change_prob) > 0.5
                summary['change_probability'] = float(np.mean(change_prob))
                summary['change_area_percentage'] = float(np.sum(change_prob > 0.5) / change_prob.size * 100)
            else:
                summary['change_detected'] = change_prob > 0.5
                summary['change_probability'] = float(change_prob)
        
        # Change type
        max_type = max(confidence_metrics.change_type_confidence.keys(), 
                      key=lambda k: confidence_metrics.change_type_confidence[k])
        summary['predicted_change_type'] = max_type
        summary['change_type_confidence'] = confidence_metrics.change_type_confidence[max_type]
        
        # Overall assessment
        conf_level, conf_desc = ConfidenceEstimator().interpret_confidence_level(
            confidence_metrics.overall_confidence
        )
        summary['confidence_level'] = conf_level
        summary['confidence_description'] = conf_desc
        summary['overall_confidence'] = confidence_metrics.overall_confidence
        
        return summary
    
    def save_explanation(self, explanation: ExplanationReport, filepath: str):
        """Save explanation report to file"""
        # Convert to serializable format
        explanation_dict = asdict(explanation)
        
        # Handle numpy arrays
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            return obj
        
        # Recursively convert numpy objects
        def recursive_convert(obj):
            if isinstance(obj, dict):
                return {k: recursive_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_convert(item) for item in obj]
            else:
                return convert_numpy(obj)
        
        explanation_serializable = recursive_convert(explanation_dict)
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(explanation_serializable, f, indent=2, default=str)
        
        logger.info(f"Explanation saved to {filepath}")
    
    def generate_visualization_report(self, 
                                    explanation: ExplanationReport,
                                    output_dir: str = "explanation_report"):
        """Generate visual explanation report"""
        from pathlib import Path
        import matplotlib.pyplot as plt
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Confidence breakdown visualization
        if explanation.confidence_breakdown:
            self._plot_confidence_breakdown(
                explanation.confidence_breakdown,
                str(output_path / "confidence_breakdown.png")
            )
        
        # Feature importance plot
        if explanation.feature_importance:
            self._plot_feature_importance(
                explanation.feature_importance,
                str(output_path / "feature_importance.png")
            )
        
        # Spatial explanations
        if explanation.spatial_explanations:
            self._plot_spatial_explanations(
                explanation.spatial_explanations,
                str(output_path / "spatial_explanations.png")
            )
        
        # Generate HTML report
        self._generate_html_report(explanation, str(output_path / "report.html"))
        
        logger.info(f"Visualization report generated in {output_dir}")
    
    def _plot_confidence_breakdown(self, confidence: ConfidenceMetrics, filepath: str):
        """Plot confidence breakdown"""
        factors = confidence.reliability_factors
        
        plt.figure(figsize=(10, 6))
        plt.bar(factors.keys(), factors.values())
        plt.title('Confidence Breakdown by Factor')
        plt.ylabel('Confidence Score')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self, importance: Dict[str, float], filepath: str):
        """Plot feature importance"""
        plt.figure(figsize=(10, 6))
        features = list(importance.keys())
        values = list(importance.values())
        
        plt.barh(features, values)
        plt.title('Feature Importance')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_spatial_explanations(self, spatial_exp: Dict, filepath: str):
        """Plot spatial explanations"""
        n_plots = len([k for k in spatial_exp.keys() if isinstance(spatial_exp[k], np.ndarray)])
        if n_plots == 0:
            return
        
        fig, axes = plt.subplots(1, min(n_plots, 4), figsize=(15, 4))
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        for key, data in spatial_exp.items():
            if isinstance(data, np.ndarray) and len(data.shape) == 2 and plot_idx < 4:
                im = axes[plot_idx].imshow(data, cmap='viridis')
                axes[plot_idx].set_title(key.replace('_', ' ').title())
                axes[plot_idx].axis('off')
                plt.colorbar(im, ax=axes[plot_idx])
                plot_idx += 1
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_html_report(self, explanation: ExplanationReport, filepath: str):
        """Generate HTML explanation report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Change Detection Explanation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 3px solid #007acc; }}
                .confidence-high {{ color: #28a745; }}
                .confidence-medium {{ color: #ffc107; }}
                .confidence-low {{ color: #dc3545; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Change Detection Explanation Report</h1>
                <p>Generated: {explanation.metadata['timestamp']}</p>
            </div>
            
            <div class="section">
                <h2>Detection Summary</h2>
                <p><strong>Change Detected:</strong> {explanation.detection_summary.get('change_detected', 'Unknown')}</p>
                <p><strong>Change Type:</strong> {explanation.detection_summary.get('predicted_change_type', 'Unknown')}</p>
                <p><strong>Confidence Level:</strong> 
                   <span class="confidence-{explanation.detection_summary.get('confidence_level', 'unknown').lower()}">
                   {explanation.detection_summary.get('confidence_level', 'Unknown')}
                   </span>
                </p>
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
                {''.join(f'<li>{rec}</li>' for rec in explanation.recommendations)}
                </ul>
            </div>
            
            <div class="section">
                <h2>Quality Assessment</h2>
                <p>Overall explanation quality: {explanation.quality_assessment.get('overall', 'Unknown'):.2f}</p>
            </div>
        </body>
        </html>
        """
        
        with open(filepath, 'w') as f:
            f.write(html_content) 