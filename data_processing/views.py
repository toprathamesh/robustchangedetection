from django.shortcuts import get_object_or_404
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from core.models import AreaOfInterest, SatelliteImage
from .satellite_apis import SatelliteDataManager, BhoonidihiAPI
import json
import logging

logger = logging.getLogger(__name__)


@login_required
@csrf_exempt
def download_satellite_images(request, aoi_id):
    """Download satellite images for an AOI"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)
    
    try:
        aoi = get_object_or_404(AreaOfInterest, id=aoi_id, user=request.user)
        data = json.loads(request.body)
        
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        if not start_date or not end_date:
            return JsonResponse({'error': 'start_date and end_date required'}, status=400)
        
        # Use satellite data manager
        data_manager = SatelliteDataManager()
        
        scenes = data_manager.search_scenes(
            aoi.geometry,
            start_date,
            end_date,
            aoi.satellite_source,
            aoi.cloud_cover_threshold
        )
        
        downloaded_scenes = []
        for scene in scenes[:5]:  # Limit to 5 scenes
            try:
                output_path = data_manager.download_scene(scene, 'satellite_data')
                if output_path:
                    # Create SatelliteImage record
                    sat_image = SatelliteImage.objects.create(
                        aoi=aoi,
                        satellite=scene.satellite,
                        acquisition_date=scene.acquisition_date,
                        cloud_cover=scene.cloud_cover,
                        scene_id=scene.scene_id,
                        file_path=output_path,
                        file_size_mb=scene.file_size_mb or 0
                    )
                    downloaded_scenes.append({
                        'id': str(sat_image.id),
                        'scene_id': scene.scene_id,
                        'acquisition_date': scene.acquisition_date.isoformat(),
                        'cloud_cover': scene.cloud_cover
                    })
            except Exception as e:
                logger.error(f"Error downloading scene {scene.scene_id}: {str(e)}")
        
        data_manager.close()
        
        return JsonResponse({
            'downloaded_scenes': downloaded_scenes,
            'total_found': len(scenes),
            'total_downloaded': len(downloaded_scenes)
        })
        
    except Exception as e:
        logger.error(f"Error in download_satellite_images: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


@login_required
@csrf_exempt
def preprocess_image(request, image_id):
    """Preprocess a satellite image"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)
    
    try:
        image = get_object_or_404(SatelliteImage, id=image_id, aoi__user=request.user)
        
        if image.is_processed:
            return JsonResponse({'message': 'Image already processed'})
        
        # Import preprocessing functions
        from .preprocessing import CloudShadowMask, RadiometricNormalization
        
        try:
            # Apply cloud/shadow masking
            masker = CloudShadowMask()
            masked_image_path = masker.process(image.file_path)
            
            # Apply radiometric normalization
            normalizer = RadiometricNormalization()
            processed_image_path = normalizer.process(masked_image_path)
            
            # Update image record
            image.file_path = processed_image_path
            image.is_processed = True
            image.save()
            
            return JsonResponse({
                'status': 'success',
                'processed_path': processed_image_path
            })
            
        except Exception as e:
            image.processing_error = str(e)
            image.save()
            raise e
        
    except Exception as e:
        logger.error(f"Error preprocessing image {image_id}: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


@login_required
def processing_status(request, job_id):
    """Get processing status for a job"""
    # This would be implemented based on your background job system
    return JsonResponse({
        'job_id': job_id,
        'status': 'completed',
        'progress': 100
    })


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def search_bhoonidhi_images(request):
    """Search for images on Bhoonidhi"""
    try:
        data = request.data
        
        # Extract search parameters
        bbox = data.get('bbox')  # [minx, miny, maxx, maxy]
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        cloud_cover = data.get('cloud_cover', 20)
        
        if not all([bbox, start_date, end_date]):
            return Response({'error': 'bbox, start_date, and end_date are required'}, status=400)
        
        # Use Bhoonidhi API
        bhoonidhi = BhoonidihiAPI()
        scenes = bhoonidhi.search_scenes(bbox, start_date, end_date, cloud_cover)
        
        # Format response
        formatted_scenes = []
        for scene in scenes:
            formatted_scenes.append({
                'scene_id': scene.scene_id,
                'satellite': scene.satellite,
                'acquisition_date': scene.acquisition_date.isoformat(),
                'cloud_cover': scene.cloud_cover,
                'resolution': '5m',  # Bhoonidhi specific
                'bands': ['Red', 'Green', 'NIR'],
                'download_url': scene.download_url,
                'thumbnail_url': scene.thumbnail_url
            })
        
        return Response({
            'scenes': formatted_scenes,
            'total_count': len(formatted_scenes)
        })
        
    except Exception as e:
        logger.error(f"Error searching Bhoonidhi: {str(e)}")
        return Response({'error': str(e)}, status=500)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def download_bhoonidhi_image(request):
    """Download image from Bhoonidhi"""
    try:
        data = request.data
        scene_id = data.get('scene_id')
        aoi_id = data.get('aoi_id')
        
        if not scene_id or not aoi_id:
            return Response({'error': 'scene_id and aoi_id are required'}, status=400)
        
        aoi = get_object_or_404(AreaOfInterest, id=aoi_id, user=request.user)
        
        # Download from Bhoonidhi
        bhoonidhi = BhoonidihiAPI()
        download_path = bhoonidhi.download_scene(scene_id, 'satellite_data')
        
        if download_path:
            # Create SatelliteImage record
            sat_image = SatelliteImage.objects.create(
                aoi=aoi,
                satellite='bhoonidhi',
                acquisition_date=data.get('acquisition_date'),
                cloud_cover=data.get('cloud_cover', 0),
                scene_id=scene_id,
                file_path=download_path,
                file_size_mb=0  # Will be calculated later
            )
            
            return Response({
                'image_id': str(sat_image.id),
                'download_path': download_path,
                'status': 'success'
            })
        else:
            return Response({'error': 'Download failed'}, status=500)
        
    except Exception as e:
        logger.error(f"Error downloading from Bhoonidhi: {str(e)}")
        return Response({'error': str(e)}, status=500) 