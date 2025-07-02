from django.shortcuts import get_object_or_404
from django.http import JsonResponse, HttpResponse, FileResponse
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .models import ChangeDetectionJob, ChangeDetectionResult, AlertThreshold
from .serializers import (
    ChangeDetectionJobSerializer, ChangeDetectionResultSerializer, 
    AlertThresholdSerializer
)
from ..alerts.tasks import process_change_detection_job
from ..core.models import AreaOfInterest
import json
import os
import zipfile
from django.conf import settings


class ChangeDetectionJobViewSet(viewsets.ModelViewSet):
    """ViewSet for change detection jobs"""
    serializer_class = ChangeDetectionJobSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return ChangeDetectionJob.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    @action(detail=True, methods=['post'])
    def cancel(self, request, pk=None):
        """Cancel a running job"""
        job = self.get_object()
        if job.status in ['pending', 'processing']:
            job.status = 'cancelled'
            job.save()
            return Response({'status': 'cancelled'})
        return Response(
            {'error': 'Job cannot be cancelled'}, 
            status=status.HTTP_400_BAD_REQUEST
        )


class ChangeDetectionResultViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for change detection results"""
    serializer_class = ChangeDetectionResultSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return ChangeDetectionResult.objects.filter(job__user=self.request.user)

    @action(detail=True, methods=['get'])
    def export_geojson(self, request, pk=None):
        """Export change polygons as GeoJSON"""
        result = self.get_object()
        if result.change_polygons:
            geojson = {
                'type': 'FeatureCollection',
                'features': [{
                    'type': 'Feature',
                    'properties': {
                        'change_type': result.change_type,
                        'confidence': result.confidence_score,
                        'area_m2': result.changed_area_m2,
                        'job_id': str(result.job.id)
                    },
                    'geometry': result.change_polygons.__geo_interface__
                }]
            }
            return Response(geojson)
        return Response({'error': 'No change polygons available'})


class AlertThresholdViewSet(viewsets.ModelViewSet):
    """ViewSet for alert thresholds"""
    serializer_class = AlertThresholdSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return AlertThreshold.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


@login_required
@csrf_exempt
def trigger_change_detection(request, aoi_id):
    """Trigger change detection for an AOI"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)
    
    try:
        aoi = get_object_or_404(AreaOfInterest, id=aoi_id, user=request.user)
        
        data = json.loads(request.body)
        before_date = data.get('before_date')
        after_date = data.get('after_date')
        
        if not before_date or not after_date:
            return JsonResponse({'error': 'before_date and after_date required'}, status=400)
        
        # Create job
        job = ChangeDetectionJob.objects.create(
            user=request.user,
            aoi=aoi,
            before_date=before_date,
            after_date=after_date,
            before_image_path=data.get('before_image_path', ''),
            after_image_path=data.get('after_image_path', ''),
            change_threshold=data.get('change_threshold', aoi.change_threshold)
        )
        
        # Trigger background processing
        process_change_detection_job.delay(str(job.id))
        
        return JsonResponse({
            'job_id': str(job.id),
            'status': 'pending',
            'message': 'Change detection job queued successfully'
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@login_required
def download_results(request, job_id):
    """Download change detection results as a ZIP file"""
    job = get_object_or_404(ChangeDetectionJob, id=job_id, user=request.user)
    
    if job.status != 'completed':
        return JsonResponse({'error': 'Job not completed'}, status=400)
    
    # Create ZIP file with results
    import tempfile
    import shutil
    
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, f'change_detection_{job.id}.zip')
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            # Add change map if available
            if job.change_map_path and os.path.exists(job.change_map_path):
                zipf.write(job.change_map_path, 'change_map.png')
            
            # Add GeoJSON if available
            if hasattr(job, 'detailed_result') and job.detailed_result.change_polygons:
                geojson = {
                    'type': 'FeatureCollection',
                    'features': [{
                        'type': 'Feature',
                        'properties': {
                            'change_type': job.detailed_result.change_type,
                            'confidence': job.detailed_result.confidence_score,
                            'area_m2': job.detailed_result.changed_area_m2
                        },
                        'geometry': job.detailed_result.change_polygons.__geo_interface__
                    }]
                }
                
                geojson_path = os.path.join(temp_dir, 'changes.geojson')
                with open(geojson_path, 'w') as f:
                    json.dump(geojson, f, indent=2)
                zipf.write(geojson_path, 'changes.geojson')
            
            # Add metadata
            metadata = {
                'job_id': str(job.id),
                'aoi_name': job.aoi.name,
                'before_date': job.before_date.isoformat(),
                'after_date': job.after_date.isoformat(),
                'change_percentage': job.change_percentage,
                'total_changed_area_m2': job.total_changed_area_m2,
                'processing_time_seconds': job.processing_time_seconds
            }
            
            metadata_path = os.path.join(temp_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            zipf.write(metadata_path, 'metadata.json')
        
        # Return file response
        response = FileResponse(
            open(zip_path, 'rb'),
            as_attachment=True,
            filename=f'change_detection_{job.id}.zip',
            content_type='application/zip'
        )
        return response 