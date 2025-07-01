from django.db import models
from django.contrib.auth.models import User
from core.simple_models import AreaOfInterest
import uuid


class ChangeDetectionJob(models.Model):
    """Change detection processing job"""
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('cancelled', 'Cancelled'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    aoi = models.ForeignKey(AreaOfInterest, on_delete=models.CASCADE)
    
    before_image_path = models.CharField(max_length=500)
    after_image_path = models.CharField(max_length=500)
    before_date = models.DateField()
    after_date = models.DateField()
    
    model_name = models.CharField(max_length=100, default='local_unet')
    change_threshold = models.FloatField(default=0.3)
    minimum_change_area_m2 = models.FloatField(default=100.0)
    
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    progress_percentage = models.IntegerField(default=0)
    error_message = models.TextField(blank=True)
    
    # Results
    change_map_path = models.CharField(max_length=500, blank=True)
    total_changed_area_m2 = models.FloatField(null=True, blank=True)
    change_percentage = models.FloatField(null=True, blank=True)
    
    # Timing
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    processing_time_seconds = models.IntegerField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Change Detection Job {self.id} - {self.status}"


class ChangeDetectionResult(models.Model):
    """Detailed change detection results"""
    CHANGE_TYPE_CHOICES = [
        ('urban_expansion', 'Urban Expansion'),
        ('deforestation', 'Deforestation'),
        ('water_change', 'Water Body Change'),
        ('vegetation_loss', 'Vegetation Loss'),
        ('other', 'Other Change'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    job = models.OneToOneField(ChangeDetectionJob, on_delete=models.CASCADE, related_name='detailed_result')
    
    change_type = models.CharField(max_length=30, choices=CHANGE_TYPE_CHOICES, default='other')
    confidence_score = models.FloatField()
    
    # Area calculations
    total_aoi_area_m2 = models.FloatField()
    changed_area_m2 = models.FloatField()
    unchanged_area_m2 = models.FloatField()
    change_percentage = models.FloatField()
    
    # Change polygons as JSON (simplified without PostGIS)
    change_polygons_geojson = models.JSONField(null=True, blank=True)
    
    # Specific change types
    urban_change_m2 = models.FloatField(default=0.0)
    vegetation_change_m2 = models.FloatField(default=0.0)
    water_change_m2 = models.FloatField(default=0.0)
    
    # Quality metrics
    processing_quality = models.CharField(max_length=20, default='good')
    cloud_interference = models.FloatField(default=0.0)
    shadow_interference = models.FloatField(default=0.0)
    
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Result for Job {self.job.id} - {self.change_type}"


class AlertThreshold(models.Model):
    """Alert threshold configuration"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    aoi = models.ForeignKey(AreaOfInterest, on_delete=models.CASCADE)
    
    minimum_change_percentage = models.FloatField(default=5.0)
    minimum_change_area_m2 = models.FloatField(default=1000.0)
    
    # Alert types
    alert_on_urban_expansion = models.BooleanField(default=True)
    alert_on_deforestation = models.BooleanField(default=True)
    alert_on_water_change = models.BooleanField(default=False)
    alert_on_vegetation_loss = models.BooleanField(default=True)
    
    # Frequency control
    max_alerts_per_week = models.IntegerField(default=3)
    last_alert_sent = models.DateTimeField(null=True, blank=True)
    
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Alert threshold for {self.aoi.name}"


class ModelMetrics(models.Model):
    """Model performance metrics"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    model_name = models.CharField(max_length=100)
    version = models.CharField(max_length=20)
    
    # Performance metrics
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()
    iou_score = models.FloatField()
    
    # Training info
    training_data_size = models.IntegerField()
    validation_data_size = models.IntegerField()
    epochs_trained = models.IntegerField()
    
    # Model file info
    model_file_path = models.CharField(max_length=500)
    model_size_mb = models.FloatField()
    
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.model_name} v{self.version} - F1: {self.f1_score:.3f}" 