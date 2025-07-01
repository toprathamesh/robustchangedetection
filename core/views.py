from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth import views as auth_views
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.contrib.auth.models import User
from .models import UserProfile, AreaOfInterest, SatelliteImage
from .serializers import (
    UserProfileSerializer, AreaOfInterestSerializer, 
    SatelliteImageSerializer, AOICreateSerializer
)


# Web Interface Views
def index(request):
    """Main dashboard view"""
    return render(request, 'core/index.html')


@login_required
def dashboard(request):
    """User dashboard with AOIs and recent activity"""
    user_aois = AreaOfInterest.objects.filter(user=request.user)
    recent_images = SatelliteImage.objects.filter(
        aoi__user=request.user
    ).order_by('-created_at')[:10]
    
    context = {
        'aois': user_aois,
        'recent_images': recent_images,
        'total_aois': user_aois.count(),
        'active_aois': user_aois.filter(is_active=True).count(),
    }
    return render(request, 'core/dashboard.html', context)


@login_required
def aoi_detail(request, aoi_id):
    """Detailed view of a specific AOI"""
    aoi = get_object_or_404(AreaOfInterest, id=aoi_id, user=request.user)
    images = aoi.images.all().order_by('-acquisition_date')
    
    context = {
        'aoi': aoi,
        'images': images,
    }
    return render(request, 'core/aoi_detail.html', context)


@login_required
def map_view(request):
    """Interactive map for AOI creation and visualization"""
    return render(request, 'core/map.html')


# API Views
class UserProfileViewSet(viewsets.ModelViewSet):
    """ViewSet for user profiles"""
    serializer_class = UserProfileSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return UserProfile.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class AreaOfInterestViewSet(viewsets.ModelViewSet):
    """ViewSet for Areas of Interest"""
    permission_classes = [IsAuthenticated]

    def get_serializer_class(self):
        if self.action == 'create':
            return AOICreateSerializer
        return AreaOfInterestSerializer

    def get_queryset(self):
        return AreaOfInterest.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    @action(detail=True, methods=['post'])
    def toggle_monitoring(self, request, pk=None):
        """Toggle monitoring status for an AOI"""
        aoi = self.get_object()
        aoi.is_active = not aoi.is_active
        aoi.save()
        
        serializer = self.get_serializer(aoi)
        return Response(serializer.data)

    @action(detail=True, methods=['get'])
    def images(self, request, pk=None):
        """Get all images for a specific AOI"""
        aoi = self.get_object()
        images = aoi.images.all().order_by('-acquisition_date')
        
        serializer = SatelliteImageSerializer(images, many=True)
        return Response(serializer.data)

    @action(detail=False, methods=['get'])
    def active(self, request):
        """Get all active AOIs for the user"""
        active_aois = self.get_queryset().filter(is_active=True)
        serializer = self.get_serializer(active_aois, many=True)
        return Response(serializer.data)


class SatelliteImageViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for satellite images (read-only)"""
    serializer_class = SatelliteImageSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return SatelliteImage.objects.filter(aoi__user=self.request.user)

    @action(detail=False, methods=['get'])
    def recent(self, request):
        """Get recent satellite images"""
        recent_images = self.get_queryset().order_by('-created_at')[:20]
        serializer = self.get_serializer(recent_images, many=True)
        return Response(serializer.data)

    @action(detail=False, methods=['get'])
    def by_aoi(self, request):
        """Get images filtered by AOI"""
        aoi_id = request.query_params.get('aoi_id')
        if not aoi_id:
            return Response(
                {'error': 'aoi_id parameter is required'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        images = self.get_queryset().filter(aoi__id=aoi_id)
        serializer = self.get_serializer(images, many=True)
        return Response(serializer.data)


# AJAX endpoints for the web interface
@login_required
@csrf_exempt
def get_aoi_geojson(request):
    """Get AOIs as GeoJSON for map display"""
    aois = AreaOfInterest.objects.filter(user=request.user)
    
    features = []
    for aoi in aois:
        feature = {
            'type': 'Feature',
            'properties': {
                'id': str(aoi.id),
                'name': aoi.name,
                'description': aoi.description,
                'satellite_source': aoi.satellite_source,
                'is_active': aoi.is_active,
                'area_km2': aoi.area_km2,
                'created_at': aoi.created_at.isoformat(),
            },
            'geometry': {
                'type': 'Polygon',
                'coordinates': list(aoi.geometry.coords)
            }
        }
        features.append(feature)
    
    geojson = {
        'type': 'FeatureCollection',
        'features': features
    }
    
    return JsonResponse(geojson)


# Authentication views
class CustomLoginView(auth_views.LoginView):
    template_name = 'registration/login.html'
    redirect_authenticated_user = True


class CustomLogoutView(auth_views.LogoutView):
    next_page = '/' 