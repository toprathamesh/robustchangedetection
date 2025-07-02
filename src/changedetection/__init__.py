"""
Change Detection System
======================
A robust, automated satellite-based change detection and alerting system.

This package provides comprehensive tools for:
- Temporal analysis with multi-year baselines
- Advanced ML models for anthropogenic change classification
- Spectral indices calculation and analysis
- Cloud masking with atmospheric correction
- Automated workflows and monitoring
- Professional GIS outputs and reporting
"""

__version__ = "1.0.0"
__author__ = "Change Detection Team"
__license__ = "MIT"

# Lazy imports to avoid Django configuration issues
def _lazy_import_django_modules():
    """Import Django-dependent modules only when Django is configured."""
    try:
        from .core import models as core_models
        from .change_detection import quality_metrics
        return {'core_models': core_models, 'quality_metrics': quality_metrics}
    except Exception:
        return {}

def _lazy_import_core_modules():
    """Import core non-Django modules."""
    from .change_detection import advanced_models, explainability
    from .data_processing import (
        temporal_analysis, 
        spectral_indices, 
        advanced_cloud_masking,
        automated_workflows,
        gis_outputs
    )
    return {
        'advanced_models': advanced_models, 
        'explainability': explainability,
        'temporal_analysis': temporal_analysis,
        'spectral_indices': spectral_indices, 
        'advanced_cloud_masking': advanced_cloud_masking,
        'automated_workflows': automated_workflows,
        'gis_outputs': gis_outputs
    }

# Import core modules immediately (non-Django dependent)
_core_modules = _lazy_import_core_modules()
globals().update(_core_modules)

# Django modules will be imported when needed
__all__ = [
    'advanced_models', 
    'explainability',
    'temporal_analysis',
    'spectral_indices', 
    'advanced_cloud_masking',
    'automated_workflows',
    'gis_outputs'
] 