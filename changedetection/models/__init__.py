"""
Change Detection Models
======================
State-of-the-art deep learning models for change detection.
"""

# Import main classes
try:
    from .model_interface import (
        UnifiedChangeDetector,
        InferenceConfig,
        TrainingConfig,
        create_detector,
        quick_predict,
        compare_models
    )
    from .sota_models import (
        ChangeDetectionConfig,
        ModelFactory,
        SiameseUNet,
        TinyCD,
        ChangeFormer,
        BaselineUNet
    )
    from .advanced_models import (
        SiameseChangeDetector,
        ChangeDetectionResult
    )
    
    __all__ = [
        # Main interface
        'UnifiedChangeDetector',
        'InferenceConfig', 
        'TrainingConfig',
        'create_detector',
        'quick_predict',
        'compare_models',
        
        # SOTA models
        'ChangeDetectionConfig',
        'ModelFactory', 
        'SiameseUNet',
        'TinyCD',
        'ChangeFormer',
        'BaselineUNet',
        
        # Advanced models
        'SiameseChangeDetector',
        'ChangeDetectionResult'
    ]
    
except ImportError as e:
    # Graceful fallback
    print(f"Warning: Some models may not be available: {e}")
    __all__ = [] 