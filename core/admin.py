from django.contrib import admin
from django.contrib.gis.admin import OSMGeoAdmin
from .models import UserProfile, AreaOfInterest, SatelliteImage


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ['user', 'email_notifications', 'created_at']
    list_filter = ['email_notifications', 'created_at']
    search_fields = ['user__username', 'user__email']
    readonly_fields = ['api_key', 'created_at', 'updated_at']


@admin.register(AreaOfInterest)
class AreaOfInterestAdmin(OSMGeoAdmin):
    list_display = ['name', 'user', 'satellite_source', 'area_km2', 'is_active', 'created_at']
    list_filter = ['satellite_source', 'is_active', 'created_at']
    search_fields = ['name', 'user__username']
    readonly_fields = ['id', 'area_km2', 'created_at', 'updated_at']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'description', 'user', 'geometry')
        }),
        ('Satellite Configuration', {
            'fields': ('satellite_source', 'cloud_cover_threshold', 'change_threshold')
        }),
        ('Monitoring Settings', {
            'fields': ('is_active', 'monitoring_frequency_days', 'last_checked')
        }),
        ('Metadata', {
            'fields': ('id', 'area_km2', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )


@admin.register(SatelliteImage)
class SatelliteImageAdmin(admin.ModelAdmin):
    list_display = ['scene_id', 'satellite', 'aoi', 'acquisition_date', 'cloud_cover', 'is_processed']
    list_filter = ['satellite', 'is_processed', 'acquisition_date']
    search_fields = ['scene_id', 'aoi__name']
    readonly_fields = ['id', 'file_size_mb', 'created_at', 'processed_at']
    
    fieldsets = (
        ('Image Information', {
            'fields': ('aoi', 'satellite', 'scene_id', 'acquisition_date', 'cloud_cover')
        }),
        ('File Details', {
            'fields': ('file_path', 'file_size_mb')
        }),
        ('Processing Status', {
            'fields': ('is_processed', 'processing_error', 'processed_at')
        }),
        ('Metadata', {
            'fields': ('id', 'created_at'),
            'classes': ('collapse',)
        }),
    ) 