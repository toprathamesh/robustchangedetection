from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

app_name = 'data_processing'

urlpatterns = [
    # Image processing endpoints
    path('download/<uuid:aoi_id>/', views.download_satellite_images, name='download_images'),
    path('preprocess/<uuid:image_id>/', views.preprocess_image, name='preprocess_image'),
    path('status/<uuid:job_id>/', views.processing_status, name='processing_status'),
    
    # Bhoonidhi integration
    path('bhoonidhi/search/', views.search_bhoonidhi_images, name='bhoonidhi_search'),
    path('bhoonidhi/download/', views.download_bhoonidhi_image, name='bhoonidhi_download'),
] 