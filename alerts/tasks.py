from celery import shared_task
from django.core.mail import send_mail
from django.conf import settings
from django.contrib.auth.models import User
from django.utils import timezone
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)


@shared_task
def send_change_detection_alert(user_id, aoi_id, change_percentage, change_area_m2):
    """Send email alert for detected changes"""
    try:
        from core.models import User, AreaOfInterest
        from change_detection.models import AlertThreshold
        
        user = User.objects.get(id=user_id)
        aoi = AreaOfInterest.objects.get(id=aoi_id)
        
        # Check if user has email notifications enabled
        if not hasattr(user, 'userprofile') or not user.userprofile.email_notifications:
            return "Email notifications disabled for user"
        
        # Check alert thresholds
        try:
            threshold = AlertThreshold.objects.get(user=user, aoi=aoi)
            if not threshold.is_active:
                return "Alerts disabled for this AOI"
            
            # Check if change meets threshold
            if change_percentage < threshold.minimum_change_percentage:
                return "Change below threshold percentage"
            
            if change_area_m2 < threshold.minimum_change_area_m2:
                return "Change below threshold area"
            
            # Check cooldown period
            if threshold.last_alert_sent:
                hours_since_last = (timezone.now() - threshold.last_alert_sent).total_seconds() / 3600
                if hours_since_last < 24:  # 24 hour cooldown
                    return "Still in cooldown period"
        
        except AlertThreshold.DoesNotExist:
            # Use default thresholds
            if change_percentage < 5.0 or change_area_m2 < 1000.0:
                return "Change below default thresholds"
        
        # Send email
        subject = f"Change Detected in {aoi.name}"
        message = f"""
        Change Detection Alert
        
        Area of Interest: {aoi.name}
        Change Detected: {change_percentage:.2f}% of area
        Changed Area: {change_area_m2:.0f} square meters
        
        Description: {aoi.description or 'No description provided'}
        
        Please log in to view detailed results:
        http://localhost:8000/aoi/{aoi.id}/
        
        This is an automated message from the Change Detection System.
        """
        
        send_mail(
            subject=subject,
            message=message,
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[user.email],
            fail_silently=False,
        )
        
        # Update last alert time
        try:
            threshold = AlertThreshold.objects.get(user=user, aoi=aoi)
            threshold.last_alert_sent = timezone.now()
            threshold.save()
        except AlertThreshold.DoesNotExist:
            pass
        
        logger.info(f"Change detection alert sent to {user.email} for AOI {aoi.name}")
        return "Alert sent successfully"
        
    except Exception as e:
        logger.error(f"Error sending change detection alert: {str(e)}")
        return f"Error: {str(e)}"


@shared_task
def process_change_detection_job(job_id):
    """Process a change detection job"""
    try:
        from change_detection.models import ChangeDetectionJob, ChangeDetectionResult
        from change_detection.ml_models import ChangeDetectionInference
        import numpy as np
        from PIL import Image
        import os
        
        job = ChangeDetectionJob.objects.get(id=job_id)
        job.status = 'processing'
        job.started_at = timezone.now()
        job.save()
        
        # Load images
        before_image = np.array(Image.open(job.before_image_path))
        after_image = np.array(Image.open(job.after_image_path))
        
        # Use local model for inference
        from ml_models.local_models import get_change_detection_model
        
        try:
            model = get_change_detection_model()
            change_map, confidence = model.predict(before_image, after_image, job.change_threshold)
        except Exception as e:
            logger.error(f"Error in model prediction: {e}")
            # Fallback: simple difference detection
            if len(before_image.shape) == 3:
                before_gray = np.mean(before_image, axis=2)
                after_gray = np.mean(after_image, axis=2)
            else:
                before_gray = before_image
                after_gray = after_image
            
            change_map = np.abs(before_gray.astype(float) - after_gray.astype(float)) > (job.change_threshold * 255)
            confidence = 0.5
        
        # Calculate statistics
        total_pixels = change_map.size
        changed_pixels = np.sum(change_map)
        change_percentage = (changed_pixels / total_pixels) * 100
        
        # Estimate area (rough calculation based on pixel size)
        pixel_area_m2 = 100  # Assuming 10m pixels for Sentinel-2
        total_changed_area_m2 = changed_pixels * pixel_area_m2
        
        # Update job
        job.status = 'completed'
        job.completed_at = timezone.now()
        job.processing_time_seconds = (job.completed_at - job.started_at).total_seconds()
        job.change_percentage = change_percentage
        job.total_changed_area_m2 = total_changed_area_m2
        
        # Save change map
        change_map_path = os.path.join(settings.MEDIA_ROOT, 'change_maps', f'change_map_{job.id}.png')
        os.makedirs(os.path.dirname(change_map_path), exist_ok=True)
        Image.fromarray((change_map * 255).astype(np.uint8)).save(change_map_path)
        job.change_map_path = change_map_path
        job.save()
        
        # Create detailed results
        result = ChangeDetectionResult.objects.create(
            job=job,
            confidence_score=confidence,
            total_aoi_area_m2=total_pixels * pixel_area_m2,
            changed_area_m2=total_changed_area_m2,
            unchanged_area_m2=(total_pixels - changed_pixels) * pixel_area_m2,
            change_percentage=change_percentage
        )
        
        # Send alert if significant change detected
        if change_percentage > 5.0:  # Default threshold
            send_change_detection_alert.delay(
                job.user.id, job.aoi.id, change_percentage, total_changed_area_m2
            )
        
        logger.info(f"Change detection job {job.id} completed successfully")
        return "Job completed successfully"
        
    except Exception as e:
        logger.error(f"Error processing change detection job {job_id}: {str(e)}")
        job = ChangeDetectionJob.objects.get(id=job_id)
        job.status = 'failed'
        job.error_message = str(e)
        job.completed_at = timezone.now()
        if job.started_at:
            job.processing_time_seconds = (job.completed_at - job.started_at).total_seconds()
        job.save()
        return f"Error: {str(e)}"


@shared_task
def check_for_new_imagery():
    """Periodic task to check for new satellite imagery"""
    try:
        from core.models import AreaOfInterest
        from data_processing.satellite_apis import SatelliteDataManager
        from datetime import datetime, timedelta
        
        # Get active AOIs that haven't been checked recently
        cutoff_time = timezone.now() - timedelta(days=1)
        aois_to_check = AreaOfInterest.objects.filter(
            is_active=True,
            last_checked__lt=cutoff_time
        ) | AreaOfInterest.objects.filter(
            is_active=True,
            last_checked__isnull=True
        )
        
        data_manager = SatelliteDataManager()
        
        for aoi in aois_to_check:
            try:
                # Search for new scenes
                end_date = datetime.now()
                start_date = end_date - timedelta(days=aoi.monitoring_frequency_days)
                
                scenes = data_manager.search_scenes(
                    aoi.geometry,
                    start_date,
                    end_date,
                    aoi.satellite_source,
                    aoi.cloud_cover_threshold
                )
                
                logger.info(f"Found {len(scenes)} scenes for AOI {aoi.name}")
                
                # Update last checked time
                aoi.last_checked = timezone.now()
                aoi.save()
                
            except Exception as e:
                logger.error(f"Error checking imagery for AOI {aoi.id}: {str(e)}")
        
        data_manager.close()
        return f"Checked {aois_to_check.count()} AOIs for new imagery"
        
    except Exception as e:
        logger.error(f"Error in check_for_new_imagery task: {str(e)}")
        return f"Error: {str(e)}" 