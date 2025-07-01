from rest_framework import serializers
from rest_framework_gis.serializers import GeoFeatureModelSerializer
from django.contrib.auth.models import User
from .models import UserProfile, AreaOfInterest, SatelliteImage


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name', 'date_joined']
        read_only_fields = ['id', 'date_joined']


class UserProfileSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    
    class Meta:
        model = UserProfile
        fields = ['user', 'email_notifications', 'api_key', 'created_at', 'updated_at']
        read_only_fields = ['api_key', 'created_at', 'updated_at']


class AreaOfInterestSerializer(GeoFeatureModelSerializer):
    user = serializers.StringRelatedField(read_only=True)
    area_km2 = serializers.FloatField(read_only=True)
    
    class Meta:
        model = AreaOfInterest
        geo_field = 'geometry'
        fields = [
            'id', 'user', 'name', 'description', 'geometry', 'area_km2',
            'satellite_source', 'cloud_cover_threshold', 'change_threshold',
            'is_active', 'monitoring_frequency_days', 'last_checked',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'user', 'area_km2', 'last_checked', 'created_at', 'updated_at']

    def validate_geometry(self, value):
        """Validate AOI size constraints"""
        from django.conf import settings
        
        # Transform to equal-area projection for area calculation
        geom_area = value.transform(3857, clone=True).area / 1_000_000  # km2
        
        max_area = getattr(settings, 'MAX_AOI_SIZE_KM2', 1000)
        if geom_area > max_area:
            raise serializers.ValidationError(
                f"AOI area ({geom_area:.2f} km²) exceeds maximum allowed size ({max_area} km²)"
            )
        
        return value


class SatelliteImageSerializer(serializers.ModelSerializer):
    aoi_name = serializers.CharField(source='aoi.name', read_only=True)
    
    class Meta:
        model = SatelliteImage
        fields = [
            'id', 'aoi', 'aoi_name', 'satellite', 'acquisition_date',
            'cloud_cover', 'scene_id', 'file_path', 'file_size_mb',
            'is_processed', 'processing_error', 'created_at', 'processed_at'
        ]
        read_only_fields = [
            'id', 'file_size_mb', 'is_processed', 'processing_error',
            'created_at', 'processed_at'
        ]


class AOICreateSerializer(serializers.ModelSerializer):
    """Simplified serializer for creating AOIs"""
    
    class Meta:
        model = AreaOfInterest
        fields = [
            'name', 'description', 'geometry', 'satellite_source',
            'cloud_cover_threshold', 'change_threshold', 'monitoring_frequency_days'
        ]

    def validate_geometry(self, value):
        """Validate AOI size constraints"""
        from django.conf import settings
        
        # Transform to equal-area projection for area calculation
        geom_area = value.transform(3857, clone=True).area / 1_000_000  # km2
        
        max_area = getattr(settings, 'MAX_AOI_SIZE_KM2', 1000)
        if geom_area > max_area:
            raise serializers.ValidationError(
                f"AOI area ({geom_area:.2f} km²) exceeds maximum allowed size ({max_area} km²)"
            )
        
        return value 