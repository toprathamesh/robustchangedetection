"""
SOTA Change Detection
====================
State-of-the-art deep learning models for multi-temporal change detection.

Available Models:
- Siamese U-Net with pre-trained encoders
- TinyCD with Mix and Attention Mask Block (MAMB)
- ChangeFormer with Transformer architecture  
- Baseline U-Net for comparison

Author: Change Detection Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Change Detection Team"

# Import main classes for easy access
try:
    from .models.model_interface import (
        UnifiedChangeDetector,
        create_detector,
        quick_predict,
        compare_models
    )
    from .models.sota_models import (
        ChangeDetectionConfig,
        ModelFactory
    )
    
    __all__ = [
        'UnifiedChangeDetector',
        'create_detector', 
        'quick_predict',
        'compare_models',
        'ChangeDetectionConfig',
        'ModelFactory'
    ]
    
except ImportError:
    # Fallback if dependencies not available
    __all__ = [] 