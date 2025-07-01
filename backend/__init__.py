# Change Detection Backend Package

# This will make sure the app is always imported when
# Django starts so that shared_task will use this app.
from .celery import app as celery_app

__all__ = ('celery_app',)

# Import simple models when needed
try:
    from django.conf import settings
    if getattr(settings, 'USE_SIMPLE_MODELS', False):
        pass
except Exception:
    pass
