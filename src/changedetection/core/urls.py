from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

# API Router
router = DefaultRouter()
router.register(r'profiles', views.UserProfileViewSet, basename='userprofile')
router.register(r'aois', views.AreaOfInterestViewSet, basename='aoi')
router.register(r'images', views.SatelliteImageViewSet, basename='satelliteimage')

app_name = 'core'

urlpatterns = [
    # Web interface URLs
    path('', views.index, name='index'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('map/', views.map_view, name='map'),
    path('aoi/<uuid:aoi_id>/', views.aoi_detail, name='aoi_detail'),
    
    # AJAX endpoints
    path('ajax/aoi-geojson/', views.get_aoi_geojson, name='aoi_geojson'),
    
    # Authentication
    path('login/', views.CustomLoginView.as_view(), name='login'),
    path('logout/', views.CustomLogoutView.as_view(), name='logout'),
    
    # API URLs
    path('api/', include(router.urls)),
] 