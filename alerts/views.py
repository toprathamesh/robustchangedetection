from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404
from .tasks import send_change_detection_alert
from core.models import AreaOfInterest


@login_required
def send_test_alert(request):
    """Send a test alert email"""
    try:
        # Use first AOI of the user for testing
        aoi = AreaOfInterest.objects.filter(user=request.user).first()
        if not aoi:
            return JsonResponse({'error': 'No AOI found'}, status=400)
        
        # Send test alert
        result = send_change_detection_alert.delay(
            request.user.id, 
            str(aoi.id), 
            15.5,  # Test change percentage
            2500   # Test change area
        )
        
        return JsonResponse({
            'message': 'Test alert queued',
            'task_id': result.id
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@login_required
def configure_alerts(request, aoi_id):
    """Configure alert settings for an AOI"""
    aoi = get_object_or_404(AreaOfInterest, id=aoi_id, user=request.user)
    
    if request.method == 'POST':
        # Implementation for saving alert configuration
        return JsonResponse({'message': 'Alert configuration saved'})
    
    # Return current configuration
    return JsonResponse({
        'aoi_id': str(aoi.id),
        'current_settings': {
            'email_notifications': True,
            'threshold_percentage': 5.0
        }
    }) 