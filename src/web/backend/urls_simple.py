"""
Simplified URL configuration for robustchangedetection project.
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/auth/', include('rest_framework.urls')),
    path('api/core/', include('changedetection.core.urls')),
    path('api/change-detection/', include('changedetection.change_detection.urls')),
    path('api/data-processing/', include('changedetection.data_processing.urls')),
    path('api/alerts/', include('changedetection.alerts.urls')),
    path('', include('changedetection.core.urls')),  # Frontend routes
]

# Serve media files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL,
                          document_root=settings.STATIC_ROOT)
