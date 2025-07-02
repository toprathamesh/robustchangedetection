"""
Automated Data Acquisition and Processing Workflows
==================================================
Orchestrates automated satellite data acquisition, processing, and analysis
workflows for continuous monitoring of AOIs.
"""

import asyncio
try:
    import schedule
    HAS_SCHEDULE = True
except ImportError:
    HAS_SCHEDULE = False
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from .satellite_apis import SatelliteDataManager, SatelliteScene
from .temporal_analysis import TemporalAnalyzer, TimeSeriesPoint
from .spectral_indices import SpectralIndicesCalculator
from .advanced_cloud_masking import CloudMaskingPipeline
from ..change_detection.advanced_models import SiameseChangeDetector
from ..change_detection.explainability import ExplainabilityEngine, ConfidenceEstimator

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProcessingResult:
    """Result from data processing workflow"""
    workflow_id: str
    status: WorkflowStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    processed_scenes: List[str] = field(default_factory=list)
    change_detections: List[Dict] = field(default_factory=list)
    error_message: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class AOIMonitoringConfig:
    """Configuration for AOI monitoring"""
    aoi_id: str
    aoi_name: str
    geometry: Dict  # GeoJSON geometry
    satellite_sources: List[str] = field(default_factory=lambda: ["sentinel2", "landsat8"])
    monitoring_frequency_days: int = 7
    max_cloud_cover: float = 20.0
    change_detection_threshold: float = 0.7
    email_alerts: bool = True
    notification_emails: List[str] = field(default_factory=list)
    active: bool = True
    last_processed: Optional[datetime] = None


class AutomatedDataAcquisitionWorkflow:
    """
    Automated workflow for continuous satellite data acquisition and processing
    """
    
    def __init__(self, config_file: str = "workflow_config.json"):
        self.config_file = config_file
        self.workflows: Dict[str, AOIMonitoringConfig] = {}
        self.active_tasks: Dict[str, asyncio.Task] = {}
        
        # Initialize components
        self.satellite_manager = SatelliteDataManager()
        self.temporal_analyzer = TemporalAnalyzer()
        self.spectral_calculator = SpectralIndicesCalculator()
        self.cloud_masking = CloudMaskingPipeline()
        self.change_detector = None  # Will be loaded when needed
        self.explainability_engine = ExplainabilityEngine()
        self.confidence_estimator = ConfidenceEstimator()
        
        # Workflow settings
        self.max_concurrent_workflows = 5
        self.data_retention_days = 365
        self.processing_queue: List[str] = []
        
        # Load configuration
        self.load_configuration()
    
    def load_configuration(self):
        """Load workflow configuration from file"""
        try:
            if Path(self.config_file).exists():
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                    
                for aoi_id, config in config_data.items():
                    self.workflows[aoi_id] = AOIMonitoringConfig(
                        aoi_id=aoi_id,
                        **config
                    )
                    
                logger.info(f"Loaded {len(self.workflows)} AOI monitoring configurations")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    def save_configuration(self):
        """Save workflow configuration to file"""
        try:
            config_data = {}
            for aoi_id, config in self.workflows.items():
                config_dict = {
                    'aoi_name': config.aoi_name,
                    'geometry': config.geometry,
                    'satellite_sources': config.satellite_sources,
                    'monitoring_frequency_days': config.monitoring_frequency_days,
                    'max_cloud_cover': config.max_cloud_cover,
                    'change_detection_threshold': config.change_detection_threshold,
                    'email_alerts': config.email_alerts,
                    'notification_emails': config.notification_emails,
                    'active': config.active,
                    'last_processed': config.last_processed.isoformat() if config.last_processed else None
                }
                config_data[aoi_id] = config_dict
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def add_aoi_monitoring(self, config: AOIMonitoringConfig):
        """Add new AOI for automated monitoring"""
        self.workflows[config.aoi_id] = config
        self.save_configuration()
        logger.info(f"Added AOI monitoring for {config.aoi_name} ({config.aoi_id})")
    
    def remove_aoi_monitoring(self, aoi_id: str):
        """Remove AOI from monitoring"""
        if aoi_id in self.workflows:
            # Cancel any active tasks
            if aoi_id in self.active_tasks:
                self.active_tasks[aoi_id].cancel()
                del self.active_tasks[aoi_id]
            
            del self.workflows[aoi_id]
            self.save_configuration()
            logger.info(f"Removed AOI monitoring for {aoi_id}")
    
    def start_monitoring(self):
        """Start automated monitoring for all active AOIs"""
        logger.info("Starting automated monitoring workflows")
        
        # Schedule regular checks
        schedule.every(1).hours.do(self._check_for_new_data)
        schedule.every(1).days.do(self._cleanup_old_data)
        schedule.every(7).days.do(self._generate_weekly_reports)
        
        # Run initial check
        asyncio.create_task(self._check_for_new_data())
        
        # Start scheduler loop
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    async def _check_for_new_data(self):
        """Check for new satellite data for all monitored AOIs"""
        logger.info("Checking for new satellite data...")
        
        current_time = datetime.now()
        tasks = []
        
        for aoi_id, config in self.workflows.items():
            if not config.active:
                continue
            
            # Check if it's time to process this AOI
            if config.last_processed:
                time_since_last = current_time - config.last_processed
                if time_since_last.days < config.monitoring_frequency_days:
                    continue
            
            # Create processing task
            if aoi_id not in self.active_tasks or self.active_tasks[aoi_id].done():
                task = asyncio.create_task(self._process_aoi(config))
                self.active_tasks[aoi_id] = task
                tasks.append(task)
        
        # Wait for tasks to complete (with timeout)
        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=3600  # 1 hour timeout
                )
            except asyncio.TimeoutError:
                logger.warning("Some processing tasks timed out")
    
    async def _process_aoi(self, config: AOIMonitoringConfig) -> ProcessingResult:
        """Process data for a single AOI"""
        workflow_id = f"{config.aoi_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        logger.info(f"Starting processing workflow {workflow_id} for AOI {config.aoi_name}")
        
        try:
            result = ProcessingResult(
                workflow_id=workflow_id,
                status=WorkflowStatus.RUNNING,
                start_time=start_time
            )
            
            # Step 1: Search for new satellite scenes
            scenes = await self._search_satellite_scenes(config)
            if not scenes:
                logger.info(f"No new scenes found for AOI {config.aoi_id}")
                result.status = WorkflowStatus.COMPLETED
                result.end_time = datetime.now()
                return result
            
            logger.info(f"Found {len(scenes)} new scenes for AOI {config.aoi_id}")
            
            # Step 2: Download and process scenes
            processed_scenes = []
            change_detections = []
            
            for scene in scenes[:5]:  # Limit to 5 scenes per run
                try:
                    # Download scene
                    scene_path = await self._download_scene(scene, config)
                    if not scene_path:
                        continue
                    
                    # Process scene
                    processing_results = await self._process_scene(scene_path, config)
                    
                    processed_scenes.append(scene.scene_id)
                    
                    # Check for changes
                    if processing_results.get('change_detected'):
                        change_detections.append(processing_results)
                        
                        # Send alert if significant change detected
                        if (processing_results.get('confidence', 0) > config.change_detection_threshold
                            and config.email_alerts):
                            await self._send_change_alert(config, processing_results)
                    
                except Exception as e:
                    logger.error(f"Error processing scene {scene.scene_id}: {e}")
                    continue
            
            # Update workflow status
            result.processed_scenes = processed_scenes
            result.change_detections = change_detections
            result.status = WorkflowStatus.COMPLETED
            result.end_time = datetime.now()
            
            # Update last processed time
            config.last_processed = datetime.now()
            self.save_configuration()
            
            logger.info(f"Completed workflow {workflow_id}: processed {len(processed_scenes)} scenes, "
                       f"detected {len(change_detections)} changes")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in workflow {workflow_id}: {e}")
            result.status = WorkflowStatus.FAILED
            result.error_message = str(e)
            result.end_time = datetime.now()
            return result
    
    async def _search_satellite_scenes(self, config: AOIMonitoringConfig) -> List[SatelliteScene]:
        """Search for satellite scenes for the AOI"""
        from shapely.geometry import shape
        
        # Convert GeoJSON geometry to Shapely polygon
        geometry = shape(config.geometry)
        
        # Define search period
        end_date = datetime.now()
        start_date = end_date - timedelta(days=config.monitoring_frequency_days * 2)
        
        all_scenes = []
        
        # Search each satellite source
        for satellite in config.satellite_sources:
            try:
                scenes = self.satellite_manager.search_scenes(
                    aoi_geometry=geometry,
                    start_date=start_date,
                    end_date=end_date,
                    satellite=satellite,
                    max_cloud_cover=config.max_cloud_cover
                )
                all_scenes.extend(scenes)
            except Exception as e:
                logger.error(f"Error searching {satellite} scenes: {e}")
        
        # Filter out already processed scenes
        new_scenes = []
        for scene in all_scenes:
            if not self._is_scene_processed(scene.scene_id, config.aoi_id):
                new_scenes.append(scene)
        
        return new_scenes
    
    async def _download_scene(self, scene: SatelliteScene, config: AOIMonitoringConfig) -> Optional[str]:
        """Download satellite scene"""
        output_dir = f"data/satellite/{config.aoi_id}/{scene.satellite}"
        
        try:
            scene_path = self.satellite_manager.download_scene(scene, output_dir)
            if scene_path:
                # Mark scene as processed
                self._mark_scene_processed(scene.scene_id, config.aoi_id)
            return scene_path
        except Exception as e:
            logger.error(f"Error downloading scene {scene.scene_id}: {e}")
            return None
    
    async def _process_scene(self, scene_path: str, config: AOIMonitoringConfig) -> Dict:
        """Process downloaded satellite scene"""
        try:
            import rasterio
            import numpy as np
            
            # Load the scene
            with rasterio.open(scene_path) as src:
                image = src.read()
                # Transpose to (H, W, C) format
                image = np.transpose(image, (1, 2, 0))
                bands_info = {
                    'bands': {f'band_{i}': i for i in range(image.shape[2])}
                }
            
            # Apply cloud masking
            cloud_result = self.cloud_masking.process_image(
                image=image,
                bands_info=bands_info,
                metadata={},
                method='ml',
                apply_atmospheric_correction=True
            )
            
            # Calculate spectral indices
            indices_result = self.spectral_calculator.calculate_all_indices(
                image=image,
                mask=cloud_result.combined_mask
            )
            
            # Create time series point
            acquisition_date = datetime.now()  # Would be extracted from scene metadata
            time_point = TimeSeriesPoint(
                date=acquisition_date,
                values=indices_result['indices'],
                cloud_cover=cloud_result.metadata.get('cloud_percentage', 0),
                satellite=scene_path.split('/')[-3] if len(scene_path.split('/')) > 3 else 'unknown'
            )
            
            # Add to temporal analysis
            self.temporal_analyzer.add_observation(time_point)
            
            # Check for anthropogenic changes
            change_analysis = None
            if len(self.temporal_analyzer.time_series_data) > 10:  # Need sufficient history
                try:
                    if not self.temporal_analyzer.is_fitted:
                        self.temporal_analyzer.fit_anomaly_detector()
                    
                    change_analysis = self.temporal_analyzer.detect_anthropogenic_change(
                        time_point, return_detailed_analysis=True
                    )
                except Exception as e:
                    logger.error(f"Error in change analysis: {e}")
            
            # Compile results
            processing_results = {
                'scene_path': scene_path,
                'acquisition_date': acquisition_date.isoformat(),
                'cloud_coverage': cloud_result.metadata.get('cloud_percentage', 0),
                'quality_score': cloud_result.quality_score,
                'spectral_indices': {k: float(np.nanmean(v)) for k, v in indices_result['indices'].items()},
                'change_detected': False,
                'confidence': 0.0
            }
            
            if change_analysis:
                processing_results.update({
                    'change_detected': change_analysis['is_anthropogenic'],
                    'confidence': change_analysis['confidence'],
                    'change_type': change_analysis.get('analysis_details', {}).get('recommendation', 'Unknown'),
                    'anomaly_score': change_analysis['anomaly_score']
                })
            
            return processing_results
            
        except Exception as e:
            logger.error(f"Error processing scene {scene_path}: {e}")
            return {'error': str(e)}
    
    async def _send_change_alert(self, config: AOIMonitoringConfig, change_data: Dict):
        """Send alert notification for detected change"""
        try:
            from django.core.mail import send_mail
            from django.conf import settings
            
            subject = f"Change Detected - {config.aoi_name}"
            
            message = f"""
            A significant change has been detected in AOI: {config.aoi_name}
            
            Detection Details:
            - Date: {change_data.get('acquisition_date', 'Unknown')}
            - Confidence: {change_data.get('confidence', 0)*100:.1f}%
            - Change Type: {change_data.get('change_type', 'Unknown')}
            - Anomaly Score: {change_data.get('anomaly_score', 0):.3f}
            
            Please review the area for potential anthropogenic activities.
            
            This is an automated message from the Change Detection System.
            """
            
            if config.notification_emails:
                send_mail(
                    subject=subject,
                    message=message,
                    from_email=settings.DEFAULT_FROM_EMAIL,
                    recipient_list=config.notification_emails,
                    fail_silently=False
                )
                
            logger.info(f"Change alert sent for AOI {config.aoi_id}")
            
        except Exception as e:
            logger.error(f"Error sending change alert: {e}")
    
    def _is_scene_processed(self, scene_id: str, aoi_id: str) -> bool:
        """Check if scene has already been processed"""
        processed_file = f"data/processed/{aoi_id}/processed_scenes.txt"
        
        try:
            if Path(processed_file).exists():
                with open(processed_file, 'r') as f:
                    processed_scenes = f.read().splitlines()
                return scene_id in processed_scenes
        except Exception:
            pass
        
        return False
    
    def _mark_scene_processed(self, scene_id: str, aoi_id: str):
        """Mark scene as processed"""
        processed_file = f"data/processed/{aoi_id}/processed_scenes.txt"
        
        try:
            Path(processed_file).parent.mkdir(parents=True, exist_ok=True)
            with open(processed_file, 'a') as f:
                f.write(f"{scene_id}\n")
        except Exception as e:
            logger.error(f"Error marking scene as processed: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old processed data"""
        logger.info("Performing data cleanup...")
        
        cutoff_date = datetime.now() - timedelta(days=self.data_retention_days)
        
        # Clean up old satellite data
        data_dir = Path("data/satellite")
        if data_dir.exists():
            for aoi_dir in data_dir.iterdir():
                if aoi_dir.is_dir():
                    for file_path in aoi_dir.rglob("*"):
                        if file_path.is_file():
                            file_date = datetime.fromtimestamp(file_path.stat().st_mtime)
                            if file_date < cutoff_date:
                                try:
                                    file_path.unlink()
                                    logger.debug(f"Deleted old file: {file_path}")
                                except Exception as e:
                                    logger.error(f"Error deleting file {file_path}: {e}")
    
    def _generate_weekly_reports(self):
        """Generate weekly monitoring reports"""
        logger.info("Generating weekly reports...")
        
        try:
            # Generate summary report for each AOI
            for aoi_id, config in self.workflows.items():
                if not config.active:
                    continue
                
                # Collect data from past week
                report_data = self._collect_weekly_data(aoi_id)
                
                # Generate report
                report_path = self._create_weekly_report(config, report_data)
                
                if report_path and config.email_alerts:
                    self._email_weekly_report(config, report_path)
                    
        except Exception as e:
            logger.error(f"Error generating weekly reports: {e}")
    
    def _collect_weekly_data(self, aoi_id: str) -> Dict:
        """Collect data for weekly report"""
        # This would collect processed data from the past week
        # For now, return mock data structure
        return {
            'scenes_processed': 0,
            'changes_detected': 0,
            'average_cloud_cover': 0.0,
            'data_quality_score': 0.0,
            'anomalies': []
        }
    
    def _create_weekly_report(self, config: AOIMonitoringConfig, data: Dict) -> Optional[str]:
        """Create weekly monitoring report"""
        try:
            report_dir = Path(f"reports/{config.aoi_id}")
            report_dir.mkdir(parents=True, exist_ok=True)
            
            report_file = report_dir / f"weekly_report_{datetime.now().strftime('%Y%m%d')}.json"
            
            report_content = {
                'aoi_id': config.aoi_id,
                'aoi_name': config.aoi_name,
                'report_date': datetime.now().isoformat(),
                'period': 'weekly',
                'data': data
            }
            
            with open(report_file, 'w') as f:
                json.dump(report_content, f, indent=2)
            
            return str(report_file)
            
        except Exception as e:
            logger.error(f"Error creating weekly report: {e}")
            return None
    
    def _email_weekly_report(self, config: AOIMonitoringConfig, report_path: str):
        """Email weekly report to stakeholders"""
        try:
            from django.core.mail import EmailMessage
            from django.conf import settings
            
            subject = f"Weekly Monitoring Report - {config.aoi_name}"
            
            message = f"""
            Weekly monitoring report for AOI: {config.aoi_name}
            
            Report attached.
            
            This is an automated message from the Change Detection System.
            """
            
            email = EmailMessage(
                subject=subject,
                body=message,
                from_email=settings.DEFAULT_FROM_EMAIL,
                to=config.notification_emails
            )
            
            email.attach_file(report_path)
            email.send()
            
            logger.info(f"Weekly report emailed for AOI {config.aoi_id}")
            
        except Exception as e:
            logger.error(f"Error emailing weekly report: {e}")
    
    def get_workflow_status(self, aoi_id: str) -> Dict:
        """Get current status of workflow for an AOI"""
        if aoi_id not in self.workflows:
            return {'error': 'AOI not found'}
        
        config = self.workflows[aoi_id]
        
        status = {
            'aoi_id': aoi_id,
            'aoi_name': config.aoi_name,
            'active': config.active,
            'last_processed': config.last_processed.isoformat() if config.last_processed else None,
            'monitoring_frequency_days': config.monitoring_frequency_days,
            'satellite_sources': config.satellite_sources,
            'is_running': aoi_id in self.active_tasks and not self.active_tasks[aoi_id].done()
        }
        
        return status
    
    def get_all_workflow_status(self) -> List[Dict]:
        """Get status of all workflows"""
        return [self.get_workflow_status(aoi_id) for aoi_id in self.workflows.keys()]
    
    def stop_monitoring(self):
        """Stop all monitoring workflows"""
        logger.info("Stopping automated monitoring...")
        
        # Cancel all active tasks
        for task in self.active_tasks.values():
            if not task.done():
                task.cancel()
        
        self.active_tasks.clear()
        
        # Clear scheduled jobs
        schedule.clear()
        
        # Close resources
        self.satellite_manager.close()
        
        logger.info("Monitoring stopped")


# Global workflow manager instance
workflow_manager = AutomatedDataAcquisitionWorkflow()


def start_automated_monitoring():
    """Start the automated monitoring system"""
    workflow_manager.start_monitoring()


def stop_automated_monitoring():
    """Stop the automated monitoring system"""
    workflow_manager.stop_monitoring()


def add_aoi_monitoring(aoi_id: str, aoi_name: str, geometry: Dict, 
                      satellite_sources: List[str] = None,
                      notification_emails: List[str] = None) -> bool:
    """Add new AOI for automated monitoring"""
    try:
        config = AOIMonitoringConfig(
            aoi_id=aoi_id,
            aoi_name=aoi_name,
            geometry=geometry,
            satellite_sources=satellite_sources or ["sentinel2", "landsat8"],
            notification_emails=notification_emails or []
        )
        
        workflow_manager.add_aoi_monitoring(config)
        return True
        
    except Exception as e:
        logger.error(f"Error adding AOI monitoring: {e}")
        return False


def get_monitoring_status() -> List[Dict]:
    """Get status of all monitoring workflows"""
    return workflow_manager.get_all_workflow_status() 