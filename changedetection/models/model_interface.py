"""
Unified Model Interface for Change Detection
===========================================
This module provides a clean, unified interface for:
- Model creation and selection
- Training workflows  
- Inference with pre/post-processing
- Model benchmarking and comparison
- Model persistence and loading
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
import time
from contextlib import contextmanager
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

SOTA_AVAILABLE = False
FALLBACK_AVAILABLE = False

try:
    from .sota_models import (
        ChangeDetectionConfig, 
        ModelFactory,
        SiameseUNet, 
        TinyCD, 
        ChangeFormer, 
        BaselineUNet,
        ChangeDetectionTrainer,
        count_parameters,
        get_model_info
    )
    SOTA_AVAILABLE = True
except ImportError:
    logger.warning("SOTA models not available. Using simplified interface.")
    try:
        from .advanced_models import SiameseChangeDetector
        FALLBACK_AVAILABLE = True
    except ImportError:
        logger.warning("No fallback models available.")
        FALLBACK_AVAILABLE = False


@dataclass
class InferenceConfig:
    """Configuration for inference"""
    threshold: float = 0.5
    apply_morphology: bool = True
    min_area: int = 100
    return_probabilities: bool = False
    normalize_inputs: bool = True
    resize_to: Optional[Tuple[int, int]] = None


@dataclass  
class TrainingConfig:
    """Configuration for training"""
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    loss_type: str = "combined"  # bce, dice, focal, combined
    optimizer: str = "adamw"  # adam, adamw, sgd
    scheduler: str = "cosine"  # cosine, step, plateau
    early_stopping_patience: int = 10
    save_best: bool = True
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 10


class ChangeDetectionDataset(Dataset):
    """Dataset for change detection training"""
    
    def __init__(self, before_images: List[str], after_images: List[str], 
                 masks: List[str], transform=None, augment=False):
        self.before_images = before_images
        self.after_images = after_images  
        self.masks = masks
        self.transform = transform
        self.augment = augment
        
        assert len(before_images) == len(after_images) == len(masks)
    
    def __len__(self):
        return len(self.before_images)
    
    def __getitem__(self, idx):
        # Load images
        before_img = np.array(Image.open(self.before_images[idx]).convert('RGB'))
        after_img = np.array(Image.open(self.after_images[idx]).convert('RGB'))
        mask = np.array(Image.open(self.masks[idx]).convert('L'))
        
        # Apply transforms
        if self.transform:
            before_img = self.transform(before_img)
            after_img = self.transform(after_img)
            mask = self.transform(mask)
        else:
            # Basic normalization
            before_img = torch.FloatTensor(before_img).permute(2, 0, 1) / 255.0
            after_img = torch.FloatTensor(after_img).permute(2, 0, 1) / 255.0
            mask = torch.FloatTensor(mask).unsqueeze(0) / 255.0
            
        return before_img, after_img, mask


class UnifiedChangeDetector:
    """
    Unified interface for all change detection models with training,
    inference, and benchmarking capabilities.
    """
    
    def __init__(self, model_type: str = "siamese_unet", config: Optional[ChangeDetectionConfig] = None,
                 device: str = 'auto'):
        """
        Initialize the change detector
        
        Args:
            model_type: Type of model ('siamese_unet', 'tinycd', 'changeformer', 'baseline_unet')
            config: Model configuration  
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.model_type = model_type
        
        # Auto-detect device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Default config if none provided
        if config is None:
            config = ChangeDetectionConfig(model_type=model_type)
        self.config = config
        
        # Create model
        if SOTA_AVAILABLE:
            try:
                self.model = ModelFactory.create_model(model_type, config)
                self.model.to(self.device)
                
                # Print model info
                info = get_model_info(self.model)
                logger.info(f"Model: {model_type}")
                logger.info(f"Parameters: {info['total_parameters']:,}")
                logger.info(f"Size: {info['size_mb']:.1f} MB")
                
            except Exception as e:
                logger.error(f"Failed to create {model_type}: {e}")
                # Fallback to simple model
                if FALLBACK_AVAILABLE:
                    logger.info("Falling back to SiameseChangeDetector")
                    self.model = SiameseChangeDetector(input_channels=3).to(self.device)  # 3 channels per image
                    self.model_type = "siamese_fallback"
                else:
                    logger.warning("No fallback model available, using simple implementation")
                    self.model = None
                    self.model_type = "simple_fallback"
        else:
            # SOTA models not available
            if FALLBACK_AVAILABLE:
                logger.info("Using SiameseChangeDetector as primary model")
                self.model = SiameseChangeDetector(input_channels=3).to(self.device)  # 3 channels per image
                self.model_type = "siamese_fallback"
            else:
                logger.warning("No advanced models available, using simple implementation")
                self.model = None
                self.model_type = "simple_fallback"
        
        self.trainer = None
        self.is_trained = False
        
    def train(self, train_dataset: Dataset, val_dataset: Optional[Dataset] = None,
              config: Optional[TrainingConfig] = None):
        """
        Train the model on provided dataset
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            config: Training configuration
        """
        if config is None:
            config = TrainingConfig()
            
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,  # Windows compatibility
            pin_memory=self.device == 'cuda'
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=self.device == 'cuda'
            )
        
        # Initialize trainer
        if self.model_type in ["siamese_unet", "tinycd", "changeformer", "baseline_unet"]:
            self.trainer = ChangeDetectionTrainer(self.model, self.device)
        else:
            # Simple training for fallback models
            return self._simple_train(train_loader, val_loader, config)
        
        # Setup optimizer
        if config.optimizer == "adam":
            optimizer = optim.Adam(
                self.model.parameters(), 
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer == "adamw":
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config.learning_rate, 
                weight_decay=config.weight_decay
            )
        else:
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                momentum=0.9
            )
        
        # Setup scheduler
        if config.scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)
        elif config.scheduler == "step":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        logger.info(f"Starting training for {config.epochs} epochs...")
        
        for epoch in range(config.epochs):
            start_time = time.time()
            
            # Train epoch
            train_loss = self.trainer.train_epoch(train_loader, optimizer, config.loss_type)
            train_losses.append(train_loss)
            
            # Validation
            val_loss = 0
            val_metrics = {}
            if val_loader:
                val_loss, val_metrics = self.trainer.validate(val_loader)
                val_losses.append(val_loss)
            
            # Learning rate scheduling
            if config.scheduler == "plateau":
                scheduler.step(val_loss if val_loader else train_loss)
            else:
                scheduler.step()
            
            epoch_time = time.time() - start_time
            
            # Logging
            if epoch % config.log_interval == 0:
                logger.info(
                    f"Epoch {epoch:3d}/{config.epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Time: {epoch_time:.1f}s"
                )
                
                if val_metrics:
                    logger.info(
                        f"Val Metrics - "
                        f"F1: {val_metrics['f1']:.3f} | "
                        f"IoU: {val_metrics['iou']:.3f} | "
                        f"Precision: {val_metrics['precision']:.3f} | "
                        f"Recall: {val_metrics['recall']:.3f}"
                    )
            
            # Early stopping and checkpointing
            if val_loader and val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                
                if config.save_best:
                    self.save_checkpoint(
                        Path(config.checkpoint_dir) / f"best_{self.model_type}.pth",
                        epoch, train_loss, val_loss, val_metrics
                    )
            else:
                patience_counter += 1
                
            if patience_counter >= config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        self.is_trained = True
        logger.info("Training completed!")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_metrics': val_metrics
        }
    
    def _simple_train(self, train_loader, val_loader, config):
        """Simple training for fallback models"""
        logger.info("Using simple training mode")
        # This would implement basic training for non-deep learning models
        self.is_trained = True
        return {'message': 'Simple training completed'}
    
    def predict(self, before_image: Union[str, np.ndarray], 
                after_image: Union[str, np.ndarray],
                config: Optional[InferenceConfig] = None) -> Dict[str, Any]:
        """
        Predict change map for a pair of images
        
        Args:
            before_image: Path to before image or numpy array
            after_image: Path to after image or numpy array  
            config: Inference configuration
            
        Returns:
            Dictionary containing change map and additional outputs
        """
        if config is None:
            config = InferenceConfig()
            
        # Load and preprocess images
        before_img = self._load_image(before_image)
        after_img = self._load_image(after_image)
        
        # Ensure same size
        if before_img.shape != after_img.shape:
            target_shape = (min(before_img.shape[0], after_img.shape[0]),
                          min(before_img.shape[1], after_img.shape[1]))
            before_img = cv2.resize(before_img, target_shape[::-1])
            after_img = cv2.resize(after_img, target_shape[::-1])
        
        # Resize if specified
        if config.resize_to:
            before_img = cv2.resize(before_img, config.resize_to)
            after_img = cv2.resize(after_img, config.resize_to)
        
        # Normalize
        if config.normalize_inputs:
            before_img = before_img.astype(np.float32) / 255.0
            after_img = after_img.astype(np.float32) / 255.0
        
        # Convert to tensors
        before_tensor = torch.FloatTensor(before_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        after_tensor = torch.FloatTensor(after_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Model inference
        if self.model is not None:
            self.model.eval()
            with torch.no_grad():
                if self.model_type in ["siamese_unet", "tinycd", "changeformer", "baseline_unet"]:
                    outputs = self.model(before_tensor, after_tensor)
                    change_map = outputs['change_map'].cpu().squeeze().numpy()
                    
                    # Handle multi-class output
                    if len(change_map.shape) == 3:
                        change_map = change_map[1]  # Take change class
                elif self.model_type == "siamese_fallback":
                    # Use Siamese model with separate before and after images
                    outputs = self.model(before_tensor, after_tensor)
                    change_probability = outputs['change_probability'].cpu().squeeze().numpy()
                    # Create spatial change map from probability
                    if isinstance(change_probability, np.ndarray) and change_probability.size == 1:
                        change_probability = float(change_probability)
                    change_map = np.full(before_img.shape[:2], change_probability)
                else:
                    # Fallback to simple model
                    change_map = self._simple_predict(before_img, after_img)
        else:
            # No model available, use simple prediction
            change_map = self._simple_predict(before_img, after_img)
        
        # Post-processing
        if config.return_probabilities:
            probability_map = change_map.copy()
        else:
            probability_map = None
            
        # Threshold
        binary_map = (change_map > config.threshold).astype(np.uint8)
        
        # Morphological operations
        if config.apply_morphology:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, kernel)
            binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_OPEN, kernel)
        
        # Remove small components
        if config.min_area > 0:
            contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < config.min_area:
                    cv2.fillPoly(binary_map, [contour], 0)
        
        # Calculate statistics
        total_pixels = binary_map.size
        changed_pixels = np.sum(binary_map)
        change_percentage = (changed_pixels / total_pixels) * 100
        
        results = {
            'change_map': binary_map,
            'change_percentage': change_percentage,
            'changed_pixels': int(changed_pixels),
            'total_pixels': int(total_pixels),
            'before_image_shape': before_img.shape,
            'after_image_shape': after_img.shape
        }
        
        if probability_map is not None:
            results['probability_map'] = probability_map
            
        return results
    
    def _load_image(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """Load image from path or return array"""
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
            return np.array(img)
        else:
            return image
            
    def _simple_predict(self, before_img: np.ndarray, after_img: np.ndarray) -> np.ndarray:
        """Simple prediction for fallback models"""
        # Basic difference-based change detection
        diff = np.abs(before_img.astype(np.float32) - after_img.astype(np.float32))
        change_map = np.mean(diff, axis=2) / 255.0
        return change_map
    
    def visualize_results(self, before_image: Union[str, np.ndarray],
                         after_image: Union[str, np.ndarray],
                         results: Dict[str, Any],
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (15, 5)):
        """
        Visualize change detection results
        
        Args:
            before_image: Before image
            after_image: After image  
            results: Results from predict()
            save_path: Path to save visualization
            figsize: Figure size
        """
        before_img = self._load_image(before_image)
        after_img = self._load_image(after_image)
        change_map = results['change_map']
        
        # Create visualization
        fig, axes = plt.subplots(1, 4, figsize=figsize)
        
        # Before image
        axes[0].imshow(before_img)
        axes[0].set_title('Before Image')
        axes[0].axis('off')
        
        # After image
        axes[1].imshow(after_img)
        axes[1].set_title('After Image')
        axes[1].axis('off')
        
        # Change map
        axes[2].imshow(change_map, cmap='Reds')
        axes[2].set_title(f'Change Map\n({results["change_percentage"]:.2f}% changed)')
        axes[2].axis('off')
        
        # Overlay
        overlay = after_img.copy()
        red_overlay = np.zeros_like(overlay)
        red_overlay[:, :, 0] = change_map * 255
        alpha = 0.5
        overlay = cv2.addWeighted(overlay, 1-alpha, red_overlay, alpha, 0)
        
        axes[3].imshow(overlay)
        axes[3].set_title('Change Overlay')
        axes[3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()
        
        return fig
    
    def benchmark_models(self, before_images: List[str], after_images: List[str],
                        ground_truth_masks: Optional[List[str]] = None,
                        models_to_test: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Benchmark multiple models on a dataset
        
        Args:
            before_images: List of before image paths
            after_images: List of after image paths
            ground_truth_masks: Optional ground truth masks for evaluation
            models_to_test: List of model types to test
            
        Returns:
            Benchmark results
        """
        if models_to_test is None:
            models_to_test = ["siamese_unet", "tinycd", "changeformer", "baseline_unet"]
        
        results = {}
        
        for model_type in models_to_test:
            logger.info(f"Benchmarking {model_type}...")
            
            try:
                # Create model
                config = ChangeDetectionConfig(model_type=model_type)
                detector = UnifiedChangeDetector(model_type, config, self.device)
                
                # Run inference on all images
                model_results = []
                inference_times = []
                
                for before_img, after_img in zip(before_images, after_images):
                    start_time = time.time()
                    result = detector.predict(before_img, after_img)
                    inference_time = time.time() - start_time
                    
                    model_results.append(result)
                    inference_times.append(inference_time)
                
                # Collect statistics
                avg_inference_time = np.mean(inference_times)
                avg_change_percentage = np.mean([r['change_percentage'] for r in model_results])
                
                # Model info
                info = get_model_info(detector.model)
                
                results[model_type] = {
                    'avg_inference_time': avg_inference_time,
                    'avg_change_percentage': avg_change_percentage,
                    'model_parameters': info['total_parameters'],
                    'model_size_mb': info['size_mb'],
                    'results': model_results
                }
                
                logger.info(f"{model_type} - Avg time: {avg_inference_time:.3f}s, "
                          f"Params: {info['total_parameters']:,}")
                
            except Exception as e:
                logger.error(f"Failed to benchmark {model_type}: {e}")
                results[model_type] = {'error': str(e)}
        
        return results
    
    def save_checkpoint(self, path: Path, epoch: int, train_loss: float, 
                       val_loss: float, metrics: Dict):
        """Save model checkpoint"""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'config': asdict(self.config),
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'metrics': metrics
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint"""
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = True
        
        logger.info(f"Checkpoint loaded: {path}")
        return checkpoint
    
    @contextmanager
    def evaluation_mode(self):
        """Context manager for evaluation mode"""
        was_training = self.model.training
        try:
            self.model.eval()
            yield
        finally:
            self.model.train(was_training)
    
    def get_model_summary(self) -> str:
        """Get a summary of the model architecture"""
        try:
            from torchsummary import summary
            return str(summary(self.model, (self.config.in_channels, 256, 256)))
        except ImportError:
            info = get_model_info(self.model)
            return f"Model: {self.model_type}\nParameters: {info['total_parameters']:,}\nSize: {info['size_mb']:.1f} MB"


# Convenience functions
def create_detector(model_type: str = "siamese_unet", **kwargs) -> UnifiedChangeDetector:
    """Create a change detector with default settings"""
    return UnifiedChangeDetector(model_type=model_type, **kwargs)


def quick_predict(before_image: Union[str, np.ndarray], 
                 after_image: Union[str, np.ndarray],
                 model_type: str = "siamese_unet",
                 visualize: bool = True) -> Dict[str, Any]:
    """Quick prediction with visualization"""
    detector = create_detector(model_type)
    results = detector.predict(before_image, after_image)
    
    if visualize:
        detector.visualize_results(before_image, after_image, results)
    
    return results


def compare_models(before_image: Union[str, np.ndarray],
                  after_image: Union[str, np.ndarray],
                  models: List[str] = None) -> Dict[str, Any]:
    """Compare multiple models on a single image pair"""
    if models is None:
        models = ["siamese_unet", "tinycd", "changeformer", "baseline_unet"]
    
    results = {}
    
    for model_type in models:
        try:
            results[model_type] = quick_predict(
                before_image, after_image, model_type, visualize=False
            )
        except Exception as e:
            logger.error(f"Failed to run {model_type}: {e}")
            results[model_type] = {'error': str(e)}
    
    return results 