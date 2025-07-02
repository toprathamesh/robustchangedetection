"""
Quality Metrics and Validation Framework
=======================================
Comprehensive performance metrics and validation tools for change detection models.
Includes precision, recall, F1-score, IoU, spatial accuracy metrics, and validation utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass, field
from pathlib import Path
import json
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score,
    jaccard_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import cv2
from scipy import ndimage, stats
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics for change detection"""
    # Basic classification metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    specificity: float
    
    # Advanced metrics
    iou_score: float  # Intersection over Union
    dice_coefficient: float
    kappa_score: float
    auc_roc: float
    avg_precision: float
    
    # Spatial metrics
    spatial_accuracy: float
    edge_accuracy: float
    boundary_f1: float
    
    # Type-specific metrics
    type_wise_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Confidence metrics
    confidence_calibration: float
    reliability_score: float
    
    # Additional info
    sample_size: int = 0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0


@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    metrics: ValidationMetrics
    confusion_matrices: Dict[str, np.ndarray]
    error_analysis: Dict[str, any]
    recommendations: List[str]
    plots_paths: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class ChangeDetectionValidator:
    """
    Comprehensive validation framework for change detection models
    """
    
    def __init__(self):
        self.change_types = [
            'no_change', 'urban_development', 'deforestation', 
            'mining', 'agriculture_expansion', 'infrastructure'
        ]
        
        self.spatial_kernels = {
            'edge_detection': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
            'smoothing': np.ones((3, 3)) / 9
        }
    
    def validate_model(self, 
                      model,
                      test_data: List[Tuple],
                      validation_type: str = 'comprehensive') -> ValidationReport:
        """
        Comprehensive model validation
        
        Args:
            model: Trained change detection model
            test_data: List of (before_img, after_img, true_change, true_type) tuples
            validation_type: Type of validation ('basic', 'comprehensive', 'spatial')
            
        Returns:
            Comprehensive validation report
        """
        logger.info(f"Starting {validation_type} validation with {len(test_data)} samples")
        
        # Extract predictions and ground truth
        predictions = []
        ground_truth = []
        confidence_scores = []
        
        for before_img, after_img, true_change, true_type in test_data:
            try:
                # Get model prediction
                if hasattr(model, 'predict_change'):
                    result = model.predict_change(before_img, after_img)
                    pred_change = (result.change_probability > 0.5).astype(int)
                    confidence_scores.append(result.confidence)
                else:
                    # Fallback for different model interfaces
                    pred_change = model.predict(np.stack([before_img, after_img]))
                    confidence_scores.append(0.5)  # Default confidence
                
                predictions.append(pred_change)
                ground_truth.append(true_change)
                
            except Exception as e:
                logger.warning(f"Error processing sample: {e}")
                continue
        
        if not predictions:
            raise ValueError("No valid predictions obtained from test data")
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        confidence_scores = np.array(confidence_scores)
        
        # Calculate metrics based on validation type
        if validation_type == 'basic':
            metrics = self._calculate_basic_metrics(predictions, ground_truth)
        elif validation_type == 'spatial':
            metrics = self._calculate_spatial_metrics(predictions, ground_truth, test_data)
        else:  # comprehensive
            metrics = self._calculate_comprehensive_metrics(
                predictions, ground_truth, confidence_scores, test_data
            )
        
        # Generate confusion matrices
        confusion_matrices = self._generate_confusion_matrices(predictions, ground_truth)
        
        # Error analysis
        error_analysis = self._perform_error_analysis(
            predictions, ground_truth, confidence_scores, test_data
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, error_analysis)
        
        return ValidationReport(
            metrics=metrics,
            confusion_matrices=confusion_matrices,
            error_analysis=error_analysis,
            recommendations=recommendations,
            metadata={
                'validation_type': validation_type,
                'test_samples': len(test_data),
                'validation_timestamp': datetime.now().isoformat()
            }
        )
    
    def _calculate_basic_metrics(self, predictions: np.ndarray, ground_truth: np.ndarray) -> ValidationMetrics:
        """Calculate basic classification metrics"""
        
        # Flatten for pixel-wise comparison
        pred_flat = predictions.flatten()
        true_flat = ground_truth.flatten()
        
        # Basic metrics
        accuracy = accuracy_score(true_flat, pred_flat)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_flat, pred_flat, average='binary', zero_division=0
        )
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(true_flat, pred_flat).ravel()
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # IoU and Dice
        iou = jaccard_score(true_flat, pred_flat, zero_division=0)
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        
        # Kappa score
        from sklearn.metrics import cohen_kappa_score
        kappa = cohen_kappa_score(true_flat, pred_flat)
        
        return ValidationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            specificity=specificity,
            iou_score=iou,
            dice_coefficient=dice,
            kappa_score=kappa,
            auc_roc=0.0,  # Not calculated in basic mode
            avg_precision=0.0,
            spatial_accuracy=0.0,
            edge_accuracy=0.0,
            boundary_f1=0.0,
            confidence_calibration=0.0,
            reliability_score=0.0,
            sample_size=len(true_flat),
            false_positive_rate=fpr,
            false_negative_rate=fnr
        )
    
    def _calculate_comprehensive_metrics(self, 
                                       predictions: np.ndarray, 
                                       ground_truth: np.ndarray,
                                       confidence_scores: np.ndarray,
                                       test_data: List) -> ValidationMetrics:
        """Calculate comprehensive metrics including spatial and confidence metrics"""
        
        # Start with basic metrics
        basic_metrics = self._calculate_basic_metrics(predictions, ground_truth)
        
        # AUC and average precision (requires probabilities)
        try:
            auc_roc = roc_auc_score(ground_truth.flatten(), confidence_scores.flatten())
            avg_precision = average_precision_score(ground_truth.flatten(), confidence_scores.flatten())
        except:
            auc_roc = 0.0
            avg_precision = 0.0
        
        # Spatial metrics
        spatial_accuracy = self._calculate_spatial_accuracy(predictions, ground_truth)
        edge_accuracy = self._calculate_edge_accuracy(predictions, ground_truth)
        boundary_f1 = self._calculate_boundary_f1(predictions, ground_truth)
        
        # Confidence calibration
        confidence_calibration = self._calculate_confidence_calibration(
            predictions, ground_truth, confidence_scores
        )
        
        # Reliability score
        reliability_score = self._calculate_reliability_score(
            predictions, confidence_scores
        )
        
        # Update comprehensive metrics
        return ValidationMetrics(
            accuracy=basic_metrics.accuracy,
            precision=basic_metrics.precision,
            recall=basic_metrics.recall,
            f1_score=basic_metrics.f1_score,
            specificity=basic_metrics.specificity,
            iou_score=basic_metrics.iou_score,
            dice_coefficient=basic_metrics.dice_coefficient,
            kappa_score=basic_metrics.kappa_score,
            auc_roc=auc_roc,
            avg_precision=avg_precision,
            spatial_accuracy=spatial_accuracy,
            edge_accuracy=edge_accuracy,
            boundary_f1=boundary_f1,
            confidence_calibration=confidence_calibration,
            reliability_score=reliability_score,
            sample_size=basic_metrics.sample_size,
            false_positive_rate=basic_metrics.false_positive_rate,
            false_negative_rate=basic_metrics.false_negative_rate
        )
    
    def _calculate_spatial_accuracy(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Calculate spatial accuracy considering neighborhood consistency"""
        if len(predictions.shape) < 3:
            return 0.0
        
        total_accuracy = 0
        valid_samples = 0
        
        for pred, true in zip(predictions, ground_truth):
            if len(pred.shape) == 2 and len(true.shape) == 2:
                # Apply smoothing to consider spatial context
                pred_smooth = cv2.filter2D(pred.astype(float), -1, self.spatial_kernels['smoothing'])
                true_smooth = cv2.filter2D(true.astype(float), -1, self.spatial_kernels['smoothing'])
                
                # Calculate correlation
                correlation = np.corrcoef(pred_smooth.flatten(), true_smooth.flatten())[0, 1]
                if not np.isnan(correlation):
                    total_accuracy += correlation
                    valid_samples += 1
        
        return total_accuracy / valid_samples if valid_samples > 0 else 0.0
    
    def _calculate_edge_accuracy(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Calculate accuracy at change boundaries"""
        if len(predictions.shape) < 3:
            return 0.0
        
        total_edge_accuracy = 0
        valid_samples = 0
        
        for pred, true in zip(predictions, ground_truth):
            if len(pred.shape) == 2 and len(true.shape) == 2:
                # Detect edges in ground truth
                true_edges = cv2.filter2D(true.astype(float), -1, self.spatial_kernels['edge_detection'])
                true_edges = (np.abs(true_edges) > 0.1).astype(int)
                
                if np.sum(true_edges) > 0:
                    # Calculate accuracy only at edge pixels
                    edge_accuracy = np.sum(pred[true_edges == 1] == true[true_edges == 1]) / np.sum(true_edges)
                    total_edge_accuracy += edge_accuracy
                    valid_samples += 1
        
        return total_edge_accuracy / valid_samples if valid_samples > 0 else 0.0
    
    def _calculate_boundary_f1(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Calculate F1 score specifically for boundary pixels"""
        if len(predictions.shape) < 3:
            return 0.0
        
        total_f1 = 0
        valid_samples = 0
        
        for pred, true in zip(predictions, ground_truth):
            if len(pred.shape) == 2 and len(true.shape) == 2:
                # Extract boundary pixels (dilation - erosion)
                kernel = np.ones((3, 3), np.uint8)
                true_dilated = cv2.dilate(true.astype(np.uint8), kernel, iterations=1)
                true_eroded = cv2.erode(true.astype(np.uint8), kernel, iterations=1)
                boundary_mask = (true_dilated - true_eroded) > 0
                
                if np.sum(boundary_mask) > 0:
                    pred_boundary = pred[boundary_mask]
                    true_boundary = true[boundary_mask]
                    
                    # Calculate F1 for boundary pixels
                    try:
                        _, _, f1, _ = precision_recall_fscore_support(
                            true_boundary, pred_boundary, average='binary', zero_division=0
                        )
                        total_f1 += f1
                        valid_samples += 1
                    except:
                        continue
        
        return total_f1 / valid_samples if valid_samples > 0 else 0.0
    
    def _calculate_confidence_calibration(self, 
                                        predictions: np.ndarray, 
                                        ground_truth: np.ndarray,
                                        confidence_scores: np.ndarray) -> float:
        """Calculate confidence calibration (how well confidence matches accuracy)"""
        try:
            # Bin predictions by confidence
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            calibration_error = 0
            total_samples = 0
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                # Find samples in this confidence bin
                in_bin = (confidence_scores > bin_lower) & (confidence_scores <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    # Calculate accuracy for this bin
                    bin_accuracy = (predictions[in_bin] == ground_truth[in_bin]).mean()
                    # Average confidence for this bin
                    bin_confidence = confidence_scores[in_bin].mean()
                    
                    # Add to calibration error
                    calibration_error += np.abs(bin_accuracy - bin_confidence) * prop_in_bin
                    total_samples += np.sum(in_bin)
            
            # Return inverse of calibration error (higher is better)
            return 1.0 - calibration_error
            
        except Exception as e:
            logger.warning(f"Error calculating confidence calibration: {e}")
            return 0.0
    
    def _calculate_reliability_score(self, 
                                   predictions: np.ndarray, 
                                   confidence_scores: np.ndarray) -> float:
        """Calculate reliability based on prediction consistency"""
        try:
            # Reliability based on variance of confidence scores
            confidence_std = np.std(confidence_scores)
            reliability = 1.0 / (1.0 + confidence_std)
            
            return reliability
            
        except Exception as e:
            logger.warning(f"Error calculating reliability score: {e}")
            return 0.0
    
    def _generate_confusion_matrices(self, 
                                   predictions: np.ndarray, 
                                   ground_truth: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate confusion matrices for different granularities"""
        matrices = {}
        
        # Binary confusion matrix
        pred_flat = predictions.flatten()
        true_flat = ground_truth.flatten()
        matrices['binary'] = confusion_matrix(true_flat, pred_flat)
        
        return matrices
    
    def _perform_error_analysis(self, 
                              predictions: np.ndarray, 
                              ground_truth: np.ndarray,
                              confidence_scores: np.ndarray,
                              test_data: List) -> Dict:
        """Perform detailed error analysis"""
        
        error_analysis = {}
        
        # False positive analysis
        fp_mask = (predictions == 1) & (ground_truth == 0)
        fp_rate = np.sum(fp_mask) / np.prod(predictions.shape)
        error_analysis['false_positive_rate'] = fp_rate
        
        # False negative analysis
        fn_mask = (predictions == 0) & (ground_truth == 1)
        fn_rate = np.sum(fn_mask) / np.prod(predictions.shape)
        error_analysis['false_negative_rate'] = fn_rate
        
        # Low confidence errors
        low_conf_threshold = 0.6
        low_conf_mask = confidence_scores < low_conf_threshold
        low_conf_errors = np.sum((predictions != ground_truth) & low_conf_mask)
        error_analysis['low_confidence_errors'] = low_conf_errors
        
        # High confidence errors (more concerning)
        high_conf_threshold = 0.8
        high_conf_mask = confidence_scores > high_conf_threshold
        high_conf_errors = np.sum((predictions != ground_truth) & high_conf_mask)
        error_analysis['high_confidence_errors'] = high_conf_errors
        
        return error_analysis
    
    def _generate_recommendations(self, 
                                metrics: ValidationMetrics, 
                                error_analysis: Dict) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Accuracy recommendations
        if metrics.accuracy < 0.8:
            recommendations.append("Model accuracy is below 80%. Consider retraining with more diverse data.")
        
        # Precision/Recall balance
        if metrics.precision > 0.9 and metrics.recall < 0.7:
            recommendations.append("High precision but low recall. Consider lowering decision threshold.")
        elif metrics.recall > 0.9 and metrics.precision < 0.7:
            recommendations.append("High recall but low precision. Consider raising decision threshold.")
        
        # Spatial accuracy
        if metrics.spatial_accuracy < 0.7:
            recommendations.append("Low spatial accuracy. Consider post-processing with spatial filters.")
        
        # Confidence calibration
        if metrics.confidence_calibration < 0.7:
            recommendations.append("Poor confidence calibration. Consider confidence recalibration techniques.")
        
        # Error analysis recommendations
        if error_analysis.get('high_confidence_errors', 0) > 100:
            recommendations.append("High number of confident incorrect predictions. Review model architecture.")
        
        if not recommendations:
            recommendations.append("Model performance is good across all metrics.")
        
        return recommendations
    
    def cross_validate_model(self, 
                           model_class,
                           training_data: List,
                           cv_folds: int = 5) -> Dict:
        """Perform cross-validation on the model"""
        
        logger.info(f"Starting {cv_folds}-fold cross-validation")
        
        # Prepare data for sklearn cross-validation
        X = []
        y = []
        
        for before_img, after_img, change_mask, _ in training_data:
            # Simple feature extraction for cross-validation
            feature_vector = np.concatenate([
                before_img.flatten()[:1000],  # Sample pixels
                after_img.flatten()[:1000]
            ])
            X.append(feature_vector)
            y.append(np.mean(change_mask))  # Average change
        
        X = np.array(X)
        y = (np.array(y) > 0.5).astype(int)  # Binary classification
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            model_class(), X, y, 
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            scoring='f1'
        )
        
        return {
            'cv_scores': cv_scores,
            'mean_cv_score': np.mean(cv_scores),
            'std_cv_score': np.std(cv_scores),
            'cv_folds': cv_folds
        }
    
    def generate_validation_plots(self, 
                                validation_report: ValidationReport,
                                output_dir: str = "validation_plots") -> List[str]:
        """Generate comprehensive validation plots"""
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plot_paths = []
        
        # 1. Confusion Matrix Plot
        plt.figure(figsize=(10, 8))
        cm = validation_report.confusion_matrices['binary']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        cm_path = f"{output_dir}/confusion_matrix.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(cm_path)
        
        # 2. Metrics Radar Chart
        plt.figure(figsize=(10, 10))
        metrics = validation_report.metrics
        
        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'IoU', 'Spatial Acc']
        values = [
            metrics.accuracy, metrics.precision, metrics.recall,
            metrics.f1_score, metrics.iou_score, metrics.spatial_accuracy
        ]
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        values = values + [values[0]]
        
        ax = plt.subplot(111, projection='polar')
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        plt.title('Model Performance Metrics')
        
        radar_path = f"{output_dir}/metrics_radar.png"
        plt.savefig(radar_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(radar_path)
        
        # 3. Error Analysis Bar Chart
        plt.figure(figsize=(12, 6))
        error_types = ['False Positive Rate', 'False Negative Rate']
        error_values = [metrics.false_positive_rate, metrics.false_negative_rate]
        
        plt.bar(error_types, error_values, color=['red', 'orange'])
        plt.title('Error Analysis')
        plt.ylabel('Error Rate')
        plt.ylim(0, max(error_values) * 1.2)
        
        for i, v in enumerate(error_values):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        error_path = f"{output_dir}/error_analysis.png"
        plt.savefig(error_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(error_path)
        
        logger.info(f"Generated {len(plot_paths)} validation plots in {output_dir}")
        return plot_paths
    
    def save_validation_report(self, 
                              validation_report: ValidationReport,
                              output_path: str):
        """Save comprehensive validation report"""
        
        # Convert report to serializable format
        report_dict = {
            'metrics': {
                'accuracy': validation_report.metrics.accuracy,
                'precision': validation_report.metrics.precision,
                'recall': validation_report.metrics.recall,
                'f1_score': validation_report.metrics.f1_score,
                'iou_score': validation_report.metrics.iou_score,
                'spatial_accuracy': validation_report.metrics.spatial_accuracy,
                'confidence_calibration': validation_report.metrics.confidence_calibration
            },
            'error_analysis': validation_report.error_analysis,
            'recommendations': validation_report.recommendations,
            'metadata': validation_report.metadata
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        logger.info(f"Validation report saved to {output_path}")


def validate_change_detection_model(model, test_data: List, output_dir: str = "validation_results") -> ValidationReport:
    """
    Convenience function for complete model validation
    
    Args:
        model: Trained change detection model
        test_data: Test dataset
        output_dir: Output directory for results
        
    Returns:
        Comprehensive validation report
    """
    
    validator = ChangeDetectionValidator()
    
    # Perform validation
    report = validator.validate_model(model, test_data, validation_type='comprehensive')
    
    # Generate plots
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plot_paths = validator.generate_validation_plots(report, output_dir)
    report.plots_paths = plot_paths
    
    # Save report
    validator.save_validation_report(report, f"{output_dir}/validation_report.json")
    
    return report


def calculate_model_performance_metrics(predictions: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
    """
    Quick function to calculate basic performance metrics
    
    Args:
        predictions: Model predictions
        ground_truth: Ground truth labels
        
    Returns:
        Dictionary of performance metrics
    """
    
    pred_flat = predictions.flatten()
    true_flat = ground_truth.flatten()
    
    accuracy = accuracy_score(true_flat, pred_flat)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_flat, pred_flat, average='binary', zero_division=0
    )
    iou = jaccard_score(true_flat, pred_flat, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'iou_score': iou
    } 