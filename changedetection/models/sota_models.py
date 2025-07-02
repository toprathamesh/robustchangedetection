"""
State-of-the-Art Change Detection Models
======================================
Implementation of the latest and most effective deep learning models for 
multi-temporal satellite image change detection including:

- Siamese U-Net with pre-trained ResNet-34/50 encoder
- TinyCD: Lightweight Siamese U-Net with Mix and Attention Mask Block (MAMB)  
- ChangeFormer: Transformer-based model for spatial-temporal dependencies
- Baseline U-Net for comparison and benchmarking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet34_Weights, ResNet50_Weights
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from pathlib import Path
import json
import math

logger = logging.getLogger(__name__)


@dataclass
class ChangeDetectionConfig:
    """Configuration for change detection models"""
    model_type: str = "siamese_unet"  # siamese_unet, tinycd, changeformer, baseline_unet
    encoder_name: str = "resnet34"  # resnet34, resnet50, efficientnet-b0
    encoder_weights: str = "imagenet"  # imagenet, ssl, swsl, None
    in_channels: int = 3  # RGB or multi-spectral channels
    classes: int = 2  # binary change detection
    activation: Optional[str] = "sigmoid"
    attention_type: str = "scse"  # scse, cbam, eca
    decoder_channels: List[int] = None
    
    def __post_init__(self):
        if self.decoder_channels is None:
            self.decoder_channels = [256, 128, 64, 32, 16]


class ModelFactory:
    """Factory for creating change detection models"""
    
    @staticmethod
    def create_model(model_type: str, config: ChangeDetectionConfig):
        """Create a model based on type and configuration"""
        
        if model_type == "siamese_unet":
            return SiameseUNet(config)
        elif model_type == "tinycd":
            return TinyCD(config)
        elif model_type == "changeformer":
            return ChangeFormer(config)
        elif model_type == "baseline_unet":
            return BaselineUNet(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_available_models():
        """Get list of available model types"""
        return ["siamese_unet", "tinycd", "changeformer", "baseline_unet"]
