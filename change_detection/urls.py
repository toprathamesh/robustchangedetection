from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

# API Router
router = DefaultRouter()
router.register(r'jobs', views.ChangeDetectionJobViewSet, basename='changedetectionjob')
router.register(r'results', views.ChangeDetectionResultViewSet, basename='changedetectionresult')
router.register(r'thresholds', views.AlertThresholdViewSet, basename='alertthreshold')

app_name = 'change_detection'

urlpatterns = [
    # API URLs
    path('api/', include(router.urls)),
    
    # Processing endpoints
    path('process/<uuid:aoi_id>/', views.trigger_change_detection, name='trigger_change_detection'),
    path('download/<uuid:job_id>/', views.download_results, name='download_results'),
] 