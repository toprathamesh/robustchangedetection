from django.contrib import admin
from .models import AreaOfInterest, SatelliteImage, UserProfile


@admin.register(AreaOfInterest)
class AreaOfInterestAdmin(admin.ModelAdmin):
    list_display = ['name', 'user', 'satellite_source', 'is_active', 'created_at']
    list_filter = ['satellite_source', 'is_active', 'created_at']
    search_fields = ['name', 'user__username', 'description']
    readonly_fields = ['id', 'created_at', 'updated_at']


@admin.register(SatelliteImage)
class SatelliteImageAdmin(admin.ModelAdmin):
    list_display = ['scene_id', 'satellite', 'aoi', 'acquisition_date', 'cloud_cover', 'is_processed']
    list_filter = ['satellite', 'is_processed', 'acquisition_date']
    search_fields = ['scene_id', 'aoi__name']
    readonly_fields = ['id', 'created_at']


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ['user', 'organization', 'created_at']
    search_fields = ['user__username', 'organization']
    readonly_fields = ['id', 'api_key', 'created_at', 'updated_at'] 