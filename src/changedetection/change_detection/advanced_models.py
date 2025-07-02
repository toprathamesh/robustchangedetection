"""
Advanced Machine Learning Models for Change Detection
===================================================
State-of-the-art models for anthropogenic change detection including:
- Siamese Convolutional Neural Networks
- Ensemble methods
- Attention mechanisms
- Multi-scale analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from pathlib import Path
import json
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pickle

logger = logging.getLogger(__name__)


@dataclass
class ChangeDetectionResult:
    """Result from change detection model"""
    change_probability: np.ndarray
    change_type: str
    confidence: float
    feature_importance: Dict[str, float]
    attention_maps: Optional[Dict[str, np.ndarray]] = None


class SiameseEncoder(nn.Module):
    """
    Siamese encoder network for extracting features from image pairs
    Uses ResNet-like architecture with attention mechanisms
    """
    
    def __init__(self, input_channels: int = 6, feature_dim: int = 256):
        super(SiameseEncoder, self).__init__()
        
        # Convolutional feature extractor
        self.conv1 = self._make_conv_block(input_channels, 64, 7, 2, 3)
        self.conv2 = self._make_conv_block(64, 128, 3, 2, 1)
        self.conv3 = self._make_conv_block(128, 256, 3, 2, 1)
        self.conv4 = self._make_conv_block(256, 512, 3, 2, 1)
        
        # Attention mechanism
        self.spatial_attention = SpatialAttention(512)
        self.channel_attention = ChannelAttention(512)
        
        # Feature projection
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_projection = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def _make_conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Feature extraction
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        
        # Apply attention
        x4_spatial = self.spatial_attention(x4) * x4
        x4_channel = self.channel_attention(x4_spatial) * x4_spatial
        
        # Global pooling and projection
        features = self.global_pool(x4_channel).flatten(1)
        projected_features = self.feature_projection(features)
        
        return {
            'features': projected_features,
            'attention_maps': {
                'spatial': self.spatial_attention.attention_map,
                'channel': self.channel_attention.attention_weights
            },
            'multi_scale': [x1, x2, x3, x4]
        }


class SpatialAttention(nn.Module):
    """Spatial attention mechanism"""
    
    def __init__(self, channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(channels, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.attention_map = None
        
    def forward(self, x):
        # Global average and max pooling along channel dimension
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and apply convolution
        combined = torch.cat([avg_pool, max_pool], dim=1)
        attention = self.conv(combined)
        attention = self.sigmoid(attention)
        
        # Store attention map for visualization
        self.attention_map = attention.detach().cpu().numpy()
        
        return attention


class ChannelAttention(nn.Module):
    """Channel attention mechanism"""
    
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.attention_weights = None
        
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Global pooling
        avg_pool = self.avg_pool(x).view(b, c)
        max_pool = self.max_pool(x).view(b, c)
        
        # Attention weights
        avg_weights = self.fc(avg_pool)
        max_weights = self.fc(max_pool)
        
        attention = self.sigmoid(avg_weights + max_weights).view(b, c, 1, 1)
        
        # Store attention weights for analysis
        self.attention_weights = attention.detach().cpu().numpy()
        
        return attention


class SiameseChangeDetector(nn.Module):
    """
    Siamese network for change detection with anthropogenic classification
    """
    
    def __init__(self, 
                 input_channels: int = 3,
                 feature_dim: int = 256,
                 num_change_types: int = 6):
        super(SiameseChangeDetector, self).__init__()
        
        # Shared encoder for both images
        self.encoder = SiameseEncoder(input_channels, feature_dim)
        
        # Change detection head
        self.change_classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)  # Binary change detection
        )
        
        # Change type classifier
        self.type_classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_change_types)
        )
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(feature_dim * 2 + num_change_types + 1, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.change_types = [
            'no_change',
            'urban_development', 
            'deforestation',
            'mining',
            'agriculture_expansion',
            'infrastructure'
        ]
        
    def forward(self, before_image, after_image):
        # Extract features from both images
        before_features = self.encoder(before_image)
        after_features = self.encoder(after_image)
        
        # Concatenate features
        combined_features = torch.cat([
            before_features['features'], 
            after_features['features']
        ], dim=1)
        
        # Change detection
        change_logits = self.change_classifier(combined_features)
        change_prob = torch.sigmoid(change_logits)
        
        # Change type classification
        type_logits = self.type_classifier(combined_features)
        type_probs = F.softmax(type_logits, dim=1)
        
        # Confidence estimation
        confidence_input = torch.cat([
            combined_features,
            type_probs,
            change_prob
        ], dim=1)
        confidence = self.confidence_estimator(confidence_input)
        
        return {
            'change_probability': change_prob,
            'change_logits': change_logits,
            'type_probabilities': type_probs,
            'type_logits': type_logits,
            'confidence': confidence,
            'before_attention': before_features['attention_maps'],
            'after_attention': after_features['attention_maps'],
            'features': combined_features
        }
    
    def predict_change(self, before_image: np.ndarray, after_image: np.ndarray) -> ChangeDetectionResult:
        """Predict change from numpy arrays"""
        self.eval()
        
        # Preprocess images
        before_tensor = self._preprocess_image(before_image)
        after_tensor = self._preprocess_image(after_image)
        
        with torch.no_grad():
            outputs = self.forward(before_tensor.unsqueeze(0), after_tensor.unsqueeze(0))
            
            # Extract results
            change_prob = outputs['change_probability'].cpu().numpy()[0, 0]
            type_probs = outputs['type_probabilities'].cpu().numpy()[0]
            confidence = outputs['confidence'].cpu().numpy()[0, 0]
            
            # Determine change type
            predicted_type_idx = np.argmax(type_probs)
            predicted_type = self.change_types[predicted_type_idx]
            
            # Feature importance (approximation using gradients)
            feature_importance = self._calculate_feature_importance(outputs)
            
            # Attention maps for visualization
            attention_maps = {
                'before_spatial': outputs['before_attention']['spatial'][0, 0],
                'after_spatial': outputs['after_attention']['spatial'][0, 0],
                'before_channel': outputs['before_attention']['channel'][0, :, 0, 0],
                'after_channel': outputs['after_attention']['channel'][0, :, 0, 0]
            }
        
        return ChangeDetectionResult(
            change_probability=change_prob,
            change_type=predicted_type,
            confidence=confidence,
            feature_importance=feature_importance,
            attention_maps=attention_maps
        )
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input"""
        # Normalize to 0-1 if needed
        if image.max() > 1:
            image = image.astype(np.float32) / 255.0
        
        # Ensure correct shape (C, H, W)
        if len(image.shape) == 3 and image.shape[2] <= 3:
            image = np.transpose(image, (2, 0, 1))
        
        # Resize to model input size (256x256)
        import cv2
        if image.shape[1:] != (256, 256):
            image_resized = np.zeros((image.shape[0], 256, 256))
            for i in range(image.shape[0]):
                image_resized[i] = cv2.resize(image[i], (256, 256))
            image = image_resized
        
        return torch.tensor(image, dtype=torch.float32)
    
    def _calculate_feature_importance(self, outputs: Dict) -> Dict[str, float]:
        """Calculate approximate feature importance"""
        features = outputs['features']
        
        # Use gradient-based importance approximation
        feature_norms = torch.norm(features, dim=0).cpu().numpy()
        total_norm = np.sum(feature_norms)
        
        if total_norm > 0:
            importance_scores = feature_norms / total_norm
        else:
            importance_scores = np.ones(len(feature_norms)) / len(feature_norms)
        
        # Map to semantic names (simplified)
        feature_names = [
            'spectral_features', 'texture_features', 'spatial_features',
            'temporal_features', 'edge_features', 'contextual_features'
        ]
        
        # Group features by type
        group_size = len(importance_scores) // len(feature_names)
        grouped_importance = {}
        
        for i, name in enumerate(feature_names):
            start_idx = i * group_size
            end_idx = start_idx + group_size if i < len(feature_names) - 1 else len(importance_scores)
            grouped_importance[name] = np.mean(importance_scores[start_idx:end_idx])
        
        return grouped_importance


class EnsembleChangeDetector:
    """
    Ensemble model combining multiple approaches for robust change detection
    """
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.is_fitted = False
        
    def add_model(self, name: str, model, weight: float = 1.0):
        """Add a model to the ensemble"""
        self.models[name] = model
        self.weights[name] = weight
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            validation_data: Optional[Tuple] = None):
        """
        Fit ensemble models
        
        Args:
            X_train: Training features
            y_train: Training labels  
            validation_data: Optional validation data for weight optimization
        """
        
        # Traditional ML models
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        svm_model = SVC(
            kernel='rbf',
            probability=True,
            random_state=42
        )
        
        # Fit models
        logger.info("Fitting Random Forest...")
        rf_model.fit(X_train, y_train)
        
        logger.info("Fitting SVM...")
        svm_model.fit(X_train, y_train)
        
        # Add to ensemble
        self.add_model('random_forest', rf_model, 0.4)
        self.add_model('svm', svm_model, 0.3)
        
        # Optimize weights if validation data provided
        if validation_data:
            self._optimize_weights(validation_data)
        
        self.is_fitted = True
        logger.info("Ensemble training completed")
    
    def predict(self, X: np.ndarray) -> Dict:
        """Predict using ensemble"""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted yet")
        
        predictions = {}
        probabilities = {}
        
        # Get predictions from each model
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                probabilities[name] = proba
                predictions[name] = np.argmax(proba, axis=1)
            else:
                pred = model.predict(X)
                predictions[name] = pred
                # Create dummy probabilities
                probabilities[name] = np.zeros((len(X), 2))
                probabilities[name][np.arange(len(X)), pred] = 1.0
        
        # Weighted ensemble prediction
        weighted_proba = np.zeros_like(list(probabilities.values())[0])
        total_weight = sum(self.weights.values())
        
        for name, proba in probabilities.items():
            weight = self.weights[name] / total_weight
            weighted_proba += weight * proba
        
        ensemble_prediction = np.argmax(weighted_proba, axis=1)
        ensemble_confidence = np.max(weighted_proba, axis=1)
        
        return {
            'predictions': ensemble_prediction,
            'probabilities': weighted_proba,
            'confidence': ensemble_confidence,
            'individual_predictions': predictions,
            'individual_probabilities': probabilities
        }
    
    def _optimize_weights(self, validation_data: Tuple):
        """Optimize ensemble weights using validation data"""
        X_val, y_val = validation_data
        
        # Grid search for optimal weights
        from itertools import product
        
        best_accuracy = 0
        best_weights = self.weights.copy()
        
        # Generate weight combinations
        weight_options = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        for weights in product(weight_options, repeat=len(self.models)):
            if abs(sum(weights) - 1.0) < 0.01:  # Normalize to 1
                # Update weights
                temp_weights = dict(zip(self.models.keys(), weights))
                self.weights = temp_weights
                
                # Evaluate
                results = self.predict(X_val)
                accuracy = accuracy_score(y_val, results['predictions'])
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_weights = temp_weights.copy()
        
        self.weights = best_weights
        logger.info(f"Optimized weights: {self.weights}, Validation accuracy: {best_accuracy:.3f}")
    
    def get_feature_importance(self) -> Dict:
        """Get aggregated feature importance from ensemble"""
        if not self.is_fitted:
            return {}
        
        importance_dict = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict[name] = model.feature_importances_
        
        return importance_dict
    
    def save_model(self, filepath: str):
        """Save ensemble model"""
        model_data = {
            'models': {},
            'weights': self.weights,
            'is_fitted': self.is_fitted
        }
        
        # Save individual models
        for name, model in self.models.items():
            model_path = f"{filepath}_{name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            model_data['models'][name] = model_path
        
        # Save ensemble metadata
        with open(f"{filepath}_ensemble.json", 'w') as f:
            json.dump(model_data, f, indent=2)
        
        logger.info(f"Ensemble saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load ensemble model"""
        # Load metadata
        with open(f"{filepath}_ensemble.json", 'r') as f:
            model_data = json.load(f)
        
        # Load individual models
        self.models = {}
        for name, model_path in model_data['models'].items():
            with open(model_path, 'rb') as f:
                self.models[name] = pickle.load(f)
        
        self.weights = model_data['weights']
        self.is_fitted = model_data['is_fitted']
        
        logger.info(f"Ensemble loaded from {filepath}")


class MultiScaleChangeDetector:
    """
    Multi-scale change detection using pyramid analysis
    """
    
    def __init__(self, scales: List[float] = [0.5, 1.0, 2.0]):
        self.scales = scales
        self.detectors = {}
        
    def detect_changes_multiscale(self, 
                                before_image: np.ndarray,
                                after_image: np.ndarray,
                                base_detector) -> Dict:
        """
        Perform change detection at multiple scales
        
        Args:
            before_image: Before image
            after_image: After image  
            base_detector: Base change detection model
            
        Returns:
            Multi-scale change detection results
        """
        import cv2
        
        results = {}
        original_shape = before_image.shape[:2]
        
        for scale in self.scales:
            # Resize images
            new_size = (
                int(original_shape[1] * scale),
                int(original_shape[0] * scale)
            )
            
            before_scaled = cv2.resize(before_image, new_size)
            after_scaled = cv2.resize(after_image, new_size)
            
            # Detect changes at this scale
            if hasattr(base_detector, 'predict_change'):
                scale_result = base_detector.predict_change(before_scaled, after_scaled)
            else:
                # Fallback for non-neural models
                scale_result = self._detect_changes_traditional(
                    before_scaled, after_scaled
                )
            
            # Resize result back to original size
            if hasattr(scale_result, 'change_probability'):
                change_map = scale_result.change_probability
            else:
                change_map = scale_result
            
            if isinstance(change_map, np.ndarray) and len(change_map.shape) == 2:
                change_map_resized = cv2.resize(change_map, (original_shape[1], original_shape[0]))
            else:
                change_map_resized = change_map
            
            results[f'scale_{scale}'] = {
                'change_map': change_map_resized,
                'scale': scale,
                'result': scale_result
            }
        
        # Combine results across scales
        combined_result = self._combine_multiscale_results(results)
        
        return {
            'scale_results': results,
            'combined_result': combined_result,
            'scales_used': self.scales
        }
    
    def _detect_changes_traditional(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        """Traditional change detection fallback"""
        # Simple difference-based detection
        if len(before.shape) == 3:
            before_gray = np.mean(before, axis=2)
            after_gray = np.mean(after, axis=2)
        else:
            before_gray = before
            after_gray = after
        
        diff = np.abs(before_gray.astype(float) - after_gray.astype(float))
        normalized_diff = diff / 255.0
        
        return normalized_diff
    
    def _combine_multiscale_results(self, results: Dict) -> Dict:
        """Combine change detection results across scales"""
        # Extract change maps
        change_maps = []
        weights = []
        
        for scale_key, result in results.items():
            change_map = result['change_map']
            scale = result['scale']
            
            if isinstance(change_map, np.ndarray):
                change_maps.append(change_map)
                # Weight higher scales more for fine details
                weights.append(scale)
        
        if not change_maps:
            return {'error': 'No valid change maps found'}
        
        # Weighted combination
        change_maps = np.array(change_maps)
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize
        
        combined_map = np.average(change_maps, axis=0, weights=weights)
        
        # Calculate confidence based on agreement across scales
        variance_map = np.var(change_maps, axis=0)
        confidence_map = 1.0 / (1.0 + variance_map)  # High confidence where variance is low
        
        return {
            'combined_change_map': combined_map,
            'confidence_map': confidence_map,
            'scale_agreement': 1.0 - np.mean(variance_map),
            'mean_change_probability': np.mean(combined_map),
            'max_change_probability': np.max(combined_map)
        }


class ChangeDetectionTrainer:
    """
    Trainer for advanced change detection models
    """
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        # Loss functions
        self.change_criterion = nn.BCEWithLogitsLoss()
        self.type_criterion = nn.CrossEntropyLoss()
        self.confidence_criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=1e-4, 
            weight_decay=1e-4
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=100
        )
        
        # Training metrics
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
    
    def train_epoch(self, dataloader, epoch: int) -> Dict:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct_change = 0
        correct_type = 0
        total_samples = 0
        
        for batch_idx, (before_imgs, after_imgs, change_labels, type_labels) in enumerate(dataloader):
            before_imgs = before_imgs.to(self.device)
            after_imgs = after_imgs.to(self.device)
            change_labels = change_labels.to(self.device).float()
            type_labels = type_labels.to(self.device).long()
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(before_imgs, after_imgs)
            
            # Calculate losses
            change_loss = self.change_criterion(
                outputs['change_logits'].squeeze(), 
                change_labels
            )
            
            type_loss = self.type_criterion(
                outputs['type_logits'], 
                type_labels
            )
            
            # Confidence loss (using change probability as target)
            confidence_target = outputs['change_probability'].detach()
            confidence_loss = self.confidence_criterion(
                outputs['confidence'].squeeze(),
                confidence_target.squeeze()
            )
            
            # Combined loss
            total_loss_batch = change_loss + type_loss + 0.1 * confidence_loss
            
            # Backward pass
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += total_loss_batch.item()
            
            # Change detection accuracy
            change_pred = (outputs['change_probability'] > 0.5).float()
            correct_change += (change_pred.squeeze() == change_labels).sum().item()
            
            # Type classification accuracy
            type_pred = torch.argmax(outputs['type_probabilities'], dim=1)
            correct_type += (type_pred == type_labels).sum().item()
            
            total_samples += change_labels.size(0)
            
            if batch_idx % 10 == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {total_loss_batch.item():.4f}')
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(dataloader)
        change_accuracy = correct_change / total_samples
        type_accuracy = correct_type / total_samples
        
        return {
            'loss': avg_loss,
            'change_accuracy': change_accuracy,
            'type_accuracy': type_accuracy
        }
    
    def validate(self, dataloader) -> Dict:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct_change = 0
        correct_type = 0
        total_samples = 0
        
        all_change_preds = []
        all_change_labels = []
        all_type_preds = []
        all_type_labels = []
        
        with torch.no_grad():
            for before_imgs, after_imgs, change_labels, type_labels in dataloader:
                before_imgs = before_imgs.to(self.device)
                after_imgs = after_imgs.to(self.device)
                change_labels = change_labels.to(self.device).float()
                type_labels = type_labels.to(self.device).long()
                
                outputs = self.model(before_imgs, after_imgs)
                
                # Losses
                change_loss = self.change_criterion(
                    outputs['change_logits'].squeeze(), 
                    change_labels
                )
                type_loss = self.type_criterion(
                    outputs['type_logits'], 
                    type_labels
                )
                
                total_loss += (change_loss + type_loss).item()
                
                # Predictions
                change_pred = (outputs['change_probability'] > 0.5).float()
                type_pred = torch.argmax(outputs['type_probabilities'], dim=1)
                
                # Collect for metrics
                all_change_preds.extend(change_pred.cpu().numpy())
                all_change_labels.extend(change_labels.cpu().numpy())
                all_type_preds.extend(type_pred.cpu().numpy())
                all_type_labels.extend(type_labels.cpu().numpy())
                
                correct_change += (change_pred.squeeze() == change_labels).sum().item()
                correct_type += (type_pred == type_labels).sum().item()
                total_samples += change_labels.size(0)
        
        # Calculate detailed metrics
        change_precision, change_recall, change_f1, _ = precision_recall_fscore_support(
            all_change_labels, all_change_preds, average='binary'
        )
        
        type_precision, type_recall, type_f1, _ = precision_recall_fscore_support(
            all_type_labels, all_type_preds, average='weighted'
        )
        
        return {
            'loss': total_loss / len(dataloader),
            'change_accuracy': correct_change / total_samples,
            'type_accuracy': correct_type / total_samples,
            'change_precision': change_precision,
            'change_recall': change_recall,
            'change_f1': change_f1,
            'type_precision': type_precision,
            'type_recall': type_recall,
            'type_f1': type_f1
        }
    
    def train(self, 
              train_dataloader, 
              val_dataloader, 
              num_epochs: int = 100,
              save_best: bool = True,
              save_path: str = "best_model.pth"):
        """Full training loop"""
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training
            train_metrics = self.train_epoch(train_dataloader, epoch)
            
            # Validation
            val_metrics = self.validate(val_dataloader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}, "
                       f"Change Acc: {train_metrics['change_accuracy']:.4f}, "
                       f"Type Acc: {train_metrics['type_accuracy']:.4f}")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}, "
                       f"Change F1: {val_metrics['change_f1']:.4f}, "
                       f"Type F1: {val_metrics['type_f1']:.4f}")
            
            # Save training history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['train_accuracy'].append(train_metrics['change_accuracy'])
            self.training_history['val_accuracy'].append(val_metrics['change_accuracy'])
            
            # Save best model
            if save_best and val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'train_history': self.training_history
                }, save_path)
                logger.info(f"Saved best model with val_loss: {best_val_loss:.4f}")
        
        logger.info("Training completed!")
        return self.training_history 