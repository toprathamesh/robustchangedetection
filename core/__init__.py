# Core app initialization 

# Import simple models for local development without PostGIS
try:
    from django.conf import settings
    if hasattr(settings, 'USE_SIMPLE_MODELS') and settings.USE_SIMPLE_MODELS:
        from .simple_models import *
    else:
        from .models import *
except ImportError:
    # Fallback to simple models
    from .simple_models import * 