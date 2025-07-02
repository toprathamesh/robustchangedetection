from rest_framework import serializers
from django.contrib.auth.models import User
from .models import UserProfile, AreaOfInterest, SatelliteImage


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name']


class UserProfileSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    
    class Meta:
        model = UserProfile
        fields = '__all__'


class AreaOfInterestSerializer(serializers.ModelSerializer):
    geometry_geojson = serializers.ReadOnlyField()
    
    class Meta:
        model = AreaOfInterest
        fields = '__all__'
        read_only_fields = ['id', 'user', 'created_at', 'updated_at']


class SatelliteImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = SatelliteImage
        fields = '__all__'
        read_only_fields = ['id', 'created_at']


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