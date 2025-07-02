# Core app initialization

def _get_models():
    """Lazy import of models to avoid Django settings access during import."""
    try:
        from django.conf import settings
        from django.core.exceptions import ImproperlyConfigured
        if hasattr(settings, 'USE_SIMPLE_MODELS') and settings.USE_SIMPLE_MODELS:
            from . import simple_models
            return simple_models
        else:
            from . import models
            return models
    except (ImportError, ImproperlyConfigured):
        # Fallback to simple models
        from . import simple_models
        return simple_models

# Don't import models at module level to avoid Django configuration issues
# Models will be imported when Django is properly configured 