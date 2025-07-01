from django.db import models
from django.contrib.auth.models import User
import uuid
from datetime import datetime


class UserProfile(models.Model):
    """Extended user profile"""
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    organization = models.CharField(max_length=200, blank=True)
    phone_number = models.CharField(max_length=20, blank=True)
    notification_preferences = models.JSONField(default=dict)
    api_key = models.CharField(max_length=100, unique=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user.username} Profile"

    class Meta:
        verbose_name = "User Profile"
        verbose_name_plural = "User Profiles"


class AreaOfInterest(models.Model):
    """Area of Interest for monitoring"""
    SATELLITE_CHOICES = [
        ('sentinel-2', 'Sentinel-2'),
        ('landsat-8', 'Landsat-8'),
        ('landsat-9', 'Landsat-9'),
        ('bhoonidhi', 'Bhoonidhi'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='aois')
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    
    # Simple coordinate storage instead of GIS
    center_lat = models.FloatField()
    center_lng = models.FloatField()
    bbox_north = models.FloatField()
    bbox_south = models.FloatField()
    bbox_east = models.FloatField()
    bbox_west = models.FloatField()
    
    satellite_source = models.CharField(max_length=20, choices=SATELLITE_CHOICES, default='sentinel-2')
    cloud_cover_threshold = models.FloatField(default=20.0)
    change_threshold = models.FloatField(default=0.3)
    is_active = models.BooleanField(default=True)
    
    # Calculated fields
    area_km2 = models.FloatField(default=0.0)
    last_checked = models.DateTimeField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = "Area of Interest"
        verbose_name_plural = "Areas of Interest"

    def __str__(self):
        return f"{self.name} ({self.user.username})"
    
    @property
    def geometry_geojson(self):
        """Return GeoJSON representation of the bounding box"""
        return {
            "type": "Polygon",
            "coordinates": [[
                [self.bbox_west, self.bbox_south],
                [self.bbox_east, self.bbox_south],
                [self.bbox_east, self.bbox_north],
                [self.bbox_west, self.bbox_north],
                [self.bbox_west, self.bbox_south]
            ]]
        }


class SatelliteImage(models.Model):
    """Satellite image metadata and file information"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    aoi = models.ForeignKey(AreaOfInterest, on_delete=models.CASCADE, related_name='images')
    satellite = models.CharField(max_length=50)
    scene_id = models.CharField(max_length=100, unique=True)
    acquisition_date = models.DateTimeField()
    cloud_cover = models.FloatField()
    file_path = models.CharField(max_length=500)
    file_size_mb = models.FloatField(default=0.0)
    is_processed = models.BooleanField(default=False)
    processing_error = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-acquisition_date']
        verbose_name = "Satellite Image"
        verbose_name_plural = "Satellite Images"

    def __str__(self):
        return f"{self.satellite} - {self.scene_id}" 