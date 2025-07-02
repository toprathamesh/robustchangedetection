from rest_framework import serializers
from .models import ChangeDetectionJob, ChangeDetectionResult, AlertThreshold, ModelMetrics


class ChangeDetectionJobSerializer(serializers.ModelSerializer):
    user = serializers.StringRelatedField(read_only=True)
    aoi_name = serializers.CharField(source='aoi.name', read_only=True)
    
    class Meta:
        model = ChangeDetectionJob
        fields = [
            'id', 'user', 'aoi', 'aoi_name', 'before_image_path', 'after_image_path',
            'before_date', 'after_date', 'model_name', 'change_threshold',
            'minimum_change_area_m2', 'status', 'progress_percentage', 
            'error_message', 'change_map_path', 'total_changed_area_m2',
            'change_percentage', 'created_at', 'started_at', 'completed_at',
            'processing_time_seconds'
        ]
        read_only_fields = [
            'id', 'user', 'status', 'progress_percentage', 'error_message',
            'change_map_path', 'total_changed_area_m2', 'change_percentage',
            'created_at', 'started_at', 'completed_at', 'processing_time_seconds'
        ]


class ChangeDetectionResultSerializer(serializers.ModelSerializer):
    job_id = serializers.CharField(source='job.id', read_only=True)
    aoi_name = serializers.CharField(source='job.aoi.name', read_only=True)
    
    class Meta:
        model = ChangeDetectionResult
        fields = [
            'id', 'job_id', 'aoi_name', 'change_type', 'confidence_score',
            'total_aoi_area_m2', 'changed_area_m2', 'unchanged_area_m2',
            'change_percentage', 'change_polygons', 'urban_change_m2',
            'vegetation_change_m2', 'water_change_m2', 'processing_quality',
            'cloud_interference', 'shadow_interference', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']


class AlertThresholdSerializer(serializers.ModelSerializer):
    user = serializers.StringRelatedField(read_only=True)
    aoi_name = serializers.CharField(source='aoi.name', read_only=True)
    
    class Meta:
        model = AlertThreshold
        fields = [
            'id', 'user', 'aoi', 'aoi_name', 'minimum_change_percentage',
            'minimum_change_area_m2', 'alert_on_urban_expansion',
            'alert_on_deforestation', 'alert_on_water_change',
            'alert_on_vegetation_loss', 'max_alerts_per_week',
            'last_alert_sent', 'is_active', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'user', 'last_alert_sent', 'created_at', 'updated_at']


class ModelMetricsSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelMetrics
        fields = [
            'id', 'model_name', 'version', 'accuracy', 'precision', 'recall',
            'f1_score', 'iou_score', 'training_data_size', 'validation_data_size',
            'epochs_trained', 'model_file_path', 'model_size_mb', 'is_active',
            'created_at'
        ]
        read_only_fields = ['id', 'created_at'] 