from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/auth/', include('rest_framework.urls')),
    path('api/core/', include('core.urls')),
    path('api/change-detection/', include('change_detection.urls')),
    path('api/data-processing/', include('data_processing.urls')),
    path('api/alerts/', include('alerts.urls')),
    path('', include('core.urls')),  # Frontend routes
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT) 