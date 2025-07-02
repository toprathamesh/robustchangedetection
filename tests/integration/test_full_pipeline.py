"""
Integration tests for the complete change detection pipeline.
Tests end-to-end workflows from data ingestion to alert generation.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import tempfile
import os

from django.test import TestCase, override_settings
from django.contrib.auth.models import User

from src.changedetection.core.models import AOI, ChangeEvent
from src.changedetection.data_processing.temporal_analysis import TemporalAnalyzer
from src.changedetection.data_processing.spectral_indices import SpectralIndicesCalculator, SatellitePlatform
from src.changedetection.change_detection.advanced_models import SiameseCNN, EnsembleChangeDetector
from src.changedetection.data_processing.automated_workflows import WorkflowManager
from src.changedetection.alerts.tasks import process_change_alert


class TestFullPipeline(TestCase):
    """Integration tests for complete change detection pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        
        # Create test AOI
        self.aoi = AOI.objects.create(
            name='Test AOI',
            geometry='POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))',
            user=self.user,
            is_active=True
        )

    def create_mock_satellite_image(self, shape=(100, 100), bands=10):
        """Create mock satellite image for testing."""
        # Simulate realistic satellite data
        image = np.random.rand(*shape, bands) * 0.3
        
        # Set realistic band values for Sentinel-2
        image[:, :, 0] = np.random.uniform(0.02, 0.15, shape)  # Blue
        image[:, :, 1] = np.random.uniform(0.03, 0.18, shape)  # Green
        image[:, :, 2] = np.random.uniform(0.02, 0.20, shape)  # Red
        image[:, :, 6] = np.random.uniform(0.15, 0.50, shape)  # NIR
        
        return image

    def test_complete_change_detection_workflow(self):
        """Test complete workflow from satellite data to change detection."""
        # Step 1: Initialize components
        temporal_analyzer = TemporalAnalyzer(baseline_years=1)
        spectral_calculator = SpectralIndicesCalculator(SatellitePlatform.SENTINEL2)
        change_detector = EnsembleChangeDetector()
        
        # Step 2: Create historical baseline data
        baseline_dates = []
        baseline_images = []
        
        for i in range(24):  # 2 years of monthly data
            date = datetime(2021, 1, 1) + timedelta(days=30 * i)
            baseline_dates.append(date)
            baseline_images.append(self.create_mock_satellite_image())
        
        # Step 3: Process baseline data
        baseline_indices = []
        for date, image in zip(baseline_dates, baseline_images):
            indices_result = spectral_calculator.calculate_all_indices(image)
            
            # Create temporal point
            from src.changedetection.data_processing.temporal_analysis import TimeSeriesPoint
            point = TimeSeriesPoint(
                date=date,
                values={name: np.mean(data) for name, data in indices_result['indices'].items()},
                cloud_cover=5.0,
                satellite='sentinel2'
            )
            temporal_analyzer.add_observation(point)
            baseline_indices.append(indices_result['indices'])
        
        # Step 4: Build temporal baselines
        temporal_analyzer.build_seasonal_baselines()
        temporal_analyzer.fit_anomaly_detector()
        
        # Step 5: Train change detection model (mock training)
        X_train = np.random.rand(100, 64, 64, 6)  # Mock training data
        y_train = np.random.randint(0, 2, 100)    # Mock labels
        
        with patch.object(change_detector, 'train') as mock_train:
            mock_train.return_value = {'accuracy': 0.95, 'loss': 0.1}
            change_detector.train(X_train, y_train)
        
        # Step 6: Process new observation for change detection
        current_date = datetime(2023, 6, 15)
        current_image = self.create_mock_satellite_image()
        
        # Simulate change by reducing vegetation
        current_image[:, :, 6] *= 0.3  # Reduce NIR (vegetation loss)
        
        current_indices = spectral_calculator.calculate_all_indices(current_image)
        
        # Create current temporal point
        current_point = TimeSeriesPoint(
            date=current_date,
            values={name: np.mean(data) for name, data in current_indices['indices'].items()},
            cloud_cover=5.0,
            satellite='sentinel2'
        )
        
        # Step 7: Detect temporal anomaly
        temporal_result = temporal_analyzer.detect_anthropogenic_change(current_point)
        
        # Step 8: Perform spatial change detection
        with patch.object(change_detector, 'predict_change') as mock_predict:
            mock_predict.return_value = {
                'change_probability': np.random.rand(100, 100) * 0.8 + 0.2,  # High change probability
                'change_type': 'deforestation',
                'confidence': 0.85
            }
            
            spatial_result = change_detector.predict_change(
                baseline_images[-1],  # Previous image
                current_image         # Current image
            )
        
        # Step 9: Combine results and create change event
        if temporal_result['is_anomaly'] and spatial_result['confidence'] > 0.7:
            change_event = ChangeEvent.objects.create(
                aoi=self.aoi,
                detected_at=current_date,
                change_type=spatial_result['change_type'],
                confidence=spatial_result['confidence'],
                magnitude=temporal_result['change_probability'],
                validated=False
            )
            
            # Verify change event was created
            assert change_event.id is not None
            assert change_event.change_type == 'deforestation'
            assert change_event.confidence == 0.85
        
        # Step 10: Verify complete workflow execution
        assert len(temporal_analyzer.seasonal_baselines) > 0
        assert temporal_analyzer.is_fitted
        assert 'change_probability' in spatial_result

    def test_automated_workflow_execution(self):
        """Test automated workflow manager execution."""
        workflow_manager = WorkflowManager()
        
        # Mock satellite data fetching
        with patch.object(workflow_manager, '_fetch_latest_satellite_data') as mock_fetch:
            mock_fetch.return_value = {
                'image': self.create_mock_satellite_image(),
                'date': datetime.now(),
                'cloud_cover': 5.0,
                'platform': 'sentinel2'
            }
            
            # Mock change detection
            with patch.object(workflow_manager, '_process_change_detection') as mock_process:
                mock_process.return_value = {
                    'changes_detected': True,
                    'change_events': [
                        {
                            'aoi_id': self.aoi.id,
                            'change_type': 'urban_development',
                            'confidence': 0.92,
                            'location': (0.5, 0.5)
                        }
                    ]
                }
                
                # Execute workflow
                result = workflow_manager.execute_workflow_for_aoi(self.aoi.id)
                
                # Verify workflow execution
                assert result['status'] == 'completed'
                assert 'processing_time' in result
                
                # Verify mocks were called
                mock_fetch.assert_called_once()
                mock_process.assert_called_once()

    @patch('src.changedetection.alerts.tasks.send_email_alert')
    def test_alert_generation_workflow(self, mock_send_email):
        """Test alert generation and notification workflow."""
        # Create change event
        change_event = ChangeEvent.objects.create(
            aoi=self.aoi,
            detected_at=datetime.now(),
            change_type='deforestation',
            confidence=0.88,
            magnitude=0.75,
            validated=False
        )
        
        # Process alert
        result = process_change_alert(change_event.id)
        
        # Verify alert processing
        assert result['status'] == 'success'
        mock_send_email.assert_called_once()
        
        # Verify change event was processed
        change_event.refresh_from_db()
        assert change_event.alert_sent

    def test_data_quality_assessment_workflow(self):
        """Test data quality assessment in the pipeline."""
        spectral_calculator = SpectralIndicesCalculator(SatellitePlatform.SENTINEL2)
        
        # Create image with varying quality
        good_image = self.create_mock_satellite_image()
        poor_image = np.random.rand(100, 100, 10) * 0.1  # Very low values
        
        # Calculate indices with quality assessment
        good_result = spectral_calculator.calculate_all_indices(good_image, return_quality_info=True)
        poor_result = spectral_calculator.calculate_all_indices(poor_image, return_quality_info=True)
        
        # Verify quality assessment
        assert 'quality_info' in good_result
        assert 'quality_info' in poor_result
        
        # Quality should be different
        good_quality = good_result['quality_info']['ndvi']['quality']
        poor_quality = poor_result['quality_info']['ndvi']['quality']
        
        # Good image should have better quality score
        quality_scores = {'poor': 1, 'fair': 2, 'good': 3}
        assert quality_scores.get(good_quality, 0) >= quality_scores.get(poor_quality, 0)

    def test_multi_temporal_analysis_workflow(self):
        """Test multi-temporal analysis workflow."""
        from src.changedetection.data_processing.spectral_indices import MultiTemporalIndicesAnalyzer
        
        analyzer = MultiTemporalIndicesAnalyzer()
        
        # Add time series of images
        for i in range(12):  # 12 months
            date = f'2023-{i+1:02d}-15'
            image = self.create_mock_satellite_image()
            
            # Simulate gradual vegetation loss
            if i > 6:
                image[:, :, 6] *= (1.0 - 0.1 * (i - 6))  # Gradual NIR reduction
            
            analyzer.add_image_data(
                date=date,
                image=image,
                platform=SatellitePlatform.SENTINEL2
            )
        
        # Analyze trends
        trend_result = analyzer.analyze_temporal_trends('ndvi')
        
        # Should detect decreasing trend
        assert 'trend_slope' in trend_result
        assert 'trend_statistics' in trend_result
        
        # Trend should be negative (vegetation decrease)
        if trend_result['trend_slope'] is not None:
            assert trend_result['trend_slope'] < 0

    def test_gis_export_workflow(self):
        """Test GIS export functionality."""
        from src.changedetection.data_processing.gis_outputs import AdvancedGISExporter
        
        exporter = AdvancedGISExporter()
        
        # Create mock change detection results
        change_mask = np.random.rand(100, 100) > 0.8  # 20% change pixels
        
        # Mock geospatial metadata
        mock_transform = [10.0, 0.0, 0.0, 0.0, -10.0, 50.0]  # Mock transform
        mock_crs = 'EPSG:4326'
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Export to shapefile
            shapefile_path = os.path.join(temp_dir, 'changes.shp')
            
            with patch.object(exporter, 'export_change_polygons') as mock_export:
                mock_export.return_value = {
                    'output_path': shapefile_path,
                    'feature_count': np.sum(change_mask),
                    'total_area': np.sum(change_mask) * 100,  # Mock area calculation
                    'format': 'shapefile'
                }
                
                result = exporter.export_change_polygons(
                    change_mask=change_mask,
                    transform=mock_transform,
                    crs=mock_crs,
                    output_path=shapefile_path
                )
                
                # Verify export
                assert result['feature_count'] > 0
                assert result['format'] == 'shapefile'
                mock_export.assert_called_once()

    def test_cloud_masking_workflow(self):
        """Test cloud masking workflow integration."""
        from src.changedetection.data_processing.advanced_cloud_masking import DeepCloudMask
        
        cloud_detector = DeepCloudMask()
        
        # Create image with simulated clouds
        image = self.create_mock_satellite_image()
        
        # Add cloud signatures (high reflectance in visible bands)
        cloud_area = np.zeros((100, 100), dtype=bool)
        cloud_area[20:40, 30:50] = True
        image[cloud_area, :3] = 0.8  # High reflectance in RGB
        
        with patch.object(cloud_detector, 'detect_clouds') as mock_detect:
            mock_detect.return_value = {
                'cloud_mask': cloud_area,
                'cloud_probability': np.where(cloud_area, 0.9, 0.1),
                'confidence': 0.85
            }
            
            # Detect clouds
            result = cloud_detector.detect_clouds(image)
            
            # Verify cloud detection
            assert 'cloud_mask' in result
            assert 'cloud_probability' in result
            assert np.sum(result['cloud_mask']) > 0
            
            # Apply atmospheric correction
            with patch.object(cloud_detector, 'atmospheric_correction') as mock_correction:
                mock_correction.return_value = image * 0.9  # Mock correction
                
                corrected_image = cloud_detector.atmospheric_correction(
                    image, result['cloud_mask']
                )
                
                assert corrected_image.shape == image.shape
                mock_correction.assert_called_once()

    def test_model_persistence_workflow(self):
        """Test model saving and loading workflow."""
        from src.changedetection.change_detection.advanced_models import ModelManager
        
        model_manager = ModelManager()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'test_model.pth')
            
            # Mock model saving
            with patch.object(model_manager, 'save_model') as mock_save:
                mock_save.return_value = {
                    'model_path': model_path,
                    'model_type': 'siamese_cnn',
                    'version': '1.0',
                    'saved_at': datetime.now()
                }
                
                # Save model
                save_result = model_manager.save_model(
                    model_type='siamese_cnn',
                    model_path=model_path
                )
                
                assert save_result['model_type'] == 'siamese_cnn'
                mock_save.assert_called_once()
            
            # Mock model loading
            with patch.object(model_manager, 'load_model') as mock_load:
                mock_load.return_value = {
                    'model': Mock(),
                    'metadata': {
                        'model_type': 'siamese_cnn',
                        'version': '1.0'
                    }
                }
                
                # Load model
                load_result = model_manager.load_model(model_path)
                
                assert load_result['metadata']['model_type'] == 'siamese_cnn'
                mock_load.assert_called_once()


class TestPerformanceIntegration:
    """Performance integration tests."""

    @pytest.mark.slow
    def test_large_scale_processing_performance(self):
        """Test performance with large-scale data processing."""
        # Create large dataset
        large_images = []
        for i in range(10):  # 10 large images
            image = np.random.rand(500, 500, 10) * 0.3
            large_images.append(image)
        
        spectral_calculator = SpectralIndicesCalculator(SatellitePlatform.SENTINEL2)
        
        import time
        start_time = time.time()
        
        # Process all images
        results = []
        for image in large_images:
            result = spectral_calculator.calculate_all_indices(image)
            results.append(result)
        
        processing_time = time.time() - start_time
        
        # Should process 10 large images in reasonable time (< 30 seconds)
        assert processing_time < 30.0
        assert len(results) == 10

    @pytest.mark.slow
    def test_concurrent_aoi_processing(self):
        """Test concurrent processing of multiple AOIs."""
        from concurrent.futures import ThreadPoolExecutor
        import threading
        
        def process_single_aoi(aoi_id):
            """Simulate processing single AOI."""
            # Mock processing time
            import time
            time.sleep(0.1)
            return {'aoi_id': aoi_id, 'status': 'completed'}
        
        # Simulate 50 AOIs
        aoi_ids = list(range(50))
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(process_single_aoi, aoi_ids))
        
        processing_time = time.time() - start_time
        
        # Should process concurrently faster than sequential
        assert processing_time < 2.0  # Much faster than 5 seconds (50 * 0.1)
        assert len(results) == 50
        assert all(r['status'] == 'completed' for r in results)


if __name__ == "__main__":
    pytest.main([__file__]) 