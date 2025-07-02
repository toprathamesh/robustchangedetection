"""
Test fixtures providing sample data for consistent testing across the test suite.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os

from src.changedetection.data_processing.temporal_analysis import TimeSeriesPoint


class SampleDataGenerator:
    """Generator for various types of sample data used in testing."""

    @staticmethod
    def create_satellite_image(
        height=100, 
        width=100, 
        bands=10, 
        platform='sentinel2',
        add_noise=True,
        vegetation_cover=0.7
    ):
        """
        Create realistic sample satellite image.
        
        Args:
            height: Image height in pixels
            width: Image width in pixels
            bands: Number of spectral bands
            platform: Satellite platform ('sentinel2', 'landsat8')
            add_noise: Whether to add realistic noise
            vegetation_cover: Fraction of image covered by vegetation (0-1)
        
        Returns:
            numpy.ndarray: Synthetic satellite image
        """
        image = np.zeros((height, width, bands))
        
        if platform == 'sentinel2':
            # Sentinel-2 band configuration
            # Create different land cover types
            
            # Vegetation areas
            veg_mask = np.random.rand(height, width) < vegetation_cover
            
            # Vegetation reflectance values
            image[veg_mask, 0] = np.random.uniform(0.02, 0.08, np.sum(veg_mask))  # Blue
            image[veg_mask, 1] = np.random.uniform(0.03, 0.12, np.sum(veg_mask))  # Green
            image[veg_mask, 2] = np.random.uniform(0.02, 0.08, np.sum(veg_mask))  # Red
            image[veg_mask, 6] = np.random.uniform(0.30, 0.50, np.sum(veg_mask))  # NIR
            
            # Non-vegetation areas (soil, urban)
            non_veg_mask = ~veg_mask
            image[non_veg_mask, 0] = np.random.uniform(0.08, 0.20, np.sum(non_veg_mask))  # Blue
            image[non_veg_mask, 1] = np.random.uniform(0.10, 0.25, np.sum(non_veg_mask))  # Green
            image[non_veg_mask, 2] = np.random.uniform(0.08, 0.30, np.sum(non_veg_mask))  # Red
            image[non_veg_mask, 6] = np.random.uniform(0.10, 0.25, np.sum(non_veg_mask))  # NIR
            
            # SWIR bands
            image[:, :, 8] = np.random.uniform(0.05, 0.25, (height, width))   # SWIR1
            image[:, :, 9] = np.random.uniform(0.02, 0.15, (height, width))   # SWIR2
            
        elif platform == 'landsat8':
            # Similar for Landsat-8 with appropriate scaling
            veg_mask = np.random.rand(height, width) < vegetation_cover
            
            image[veg_mask, 1] = np.random.uniform(0.02, 0.08, np.sum(veg_mask))   # Blue
            image[veg_mask, 2] = np.random.uniform(0.03, 0.12, np.sum(veg_mask))   # Green
            image[veg_mask, 3] = np.random.uniform(0.02, 0.08, np.sum(veg_mask))   # Red
            image[veg_mask, 4] = np.random.uniform(0.30, 0.50, np.sum(veg_mask))   # NIR
            
        if add_noise:
            # Add realistic sensor noise
            noise = np.random.normal(0, 0.005, image.shape)
            image = np.clip(image + noise, 0, 1)
        
        return image

    @staticmethod
    def create_time_series_data(
        start_date='2021-01-01',
        num_years=2,
        observations_per_month=3,
        add_seasonal_pattern=True,
        add_trend=False,
        add_anomalies=False,
        anomaly_probability=0.05
    ):
        """
        Create time series of spectral indices for testing.
        
        Args:
            start_date: Start date as string 'YYYY-MM-DD'
            num_years: Number of years of data
            observations_per_month: Number of observations per month
            add_seasonal_pattern: Whether to add seasonal variations
            add_trend: Whether to add long-term trend
            add_anomalies: Whether to inject anomalies
            anomaly_probability: Probability of anomaly at each time point
        
        Returns:
            list: List of TimeSeriesPoint objects
        """
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        time_series = []
        
        for month_offset in range(num_years * 12):
            for obs in range(observations_per_month):
                # Calculate date
                date = start_dt + timedelta(days=30 * month_offset + obs * 10)
                
                # Base NDVI value
                base_ndvi = 0.6
                
                if add_seasonal_pattern:
                    # Add seasonal variation (peak in summer)
                    seasonal_factor = 0.3 * np.sin(2 * np.pi * date.month / 12 - np.pi/2)
                    base_ndvi += seasonal_factor
                
                if add_trend:
                    # Add linear trend (slight decrease over time)
                    trend_factor = -0.1 * (month_offset / (num_years * 12))
                    base_ndvi += trend_factor
                
                # Add noise
                noise = np.random.normal(0, 0.05)
                ndvi = base_ndvi + noise
                
                # Check for anomaly
                is_anomaly = False
                if add_anomalies and np.random.rand() < anomaly_probability:
                    # Create anomaly (sudden drop in NDVI)
                    ndvi *= 0.3
                    is_anomaly = True
                
                # Create other indices
                ndbi = 0.2 + np.random.normal(0, 0.02)
                ndwi = -0.1 + np.random.normal(0, 0.03)
                
                # If anomaly, adjust other indices too
                if is_anomaly:
                    ndbi += 0.3  # Increase built-up
                    ndwi -= 0.1  # Decrease water
                
                values = {
                    'ndvi': np.clip(ndvi, -1, 1),
                    'ndbi': np.clip(ndbi, -1, 1),
                    'ndwi': np.clip(ndwi, -1, 1)
                }
                
                point = TimeSeriesPoint(
                    date=date,
                    values=values,
                    cloud_cover=np.random.uniform(0, 20),
                    satellite='sentinel2',
                    quality_flag='good' if not is_anomaly else 'anomaly'
                )
                
                time_series.append(point)
        
        return time_series

    @staticmethod
    def create_change_scenario(
        before_image_shape=(100, 100),
        change_type='deforestation',
        change_area_fraction=0.1,
        change_intensity=0.8
    ):
        """
        Create before/after image pair with specific change scenario.
        
        Args:
            before_image_shape: Shape of the images (height, width)
            change_type: Type of change ('deforestation', 'urban_development', 'water_change')
            change_area_fraction: Fraction of image affected by change
            change_intensity: Intensity of the change (0-1)
        
        Returns:
            tuple: (before_image, after_image, change_mask)
        """
        height, width = before_image_shape
        
        # Create before image
        before_image = SampleDataGenerator.create_satellite_image(
            height, width, bands=10, vegetation_cover=0.8
        )
        
        # Create change mask
        change_mask = np.random.rand(height, width) < change_area_fraction
        
        # Create after image (copy of before)
        after_image = before_image.copy()
        
        if change_type == 'deforestation':
            # Reduce vegetation indices in change areas
            after_image[change_mask, 6] *= (1 - change_intensity)  # Reduce NIR
            after_image[change_mask, 2] *= (1 + change_intensity * 0.5)  # Increase Red slightly
            
        elif change_type == 'urban_development':
            # Increase built-up indices
            after_image[change_mask, 6] *= (1 - change_intensity * 0.8)  # Reduce NIR
            after_image[change_mask, 8] += change_intensity * 0.3  # Increase SWIR1
            after_image[change_mask, 9] += change_intensity * 0.2  # Increase SWIR2
            
        elif change_type == 'water_change':
            # Modify water-related bands
            after_image[change_mask, 1] *= (1 + change_intensity * 0.3)  # Increase Green
            after_image[change_mask, 6] *= (1 - change_intensity * 0.6)  # Reduce NIR
        
        # Ensure values stay within valid range
        after_image = np.clip(after_image, 0, 1)
        
        return before_image, after_image, change_mask

    @staticmethod
    def create_aoi_geometries():
        """
        Create sample AOI geometries for testing.
        
        Returns:
            list: List of AOI geometries in WKT format
        """
        geometries = [
            # Small square AOI
            'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))',
            
            # Larger rectangular AOI
            'POLYGON((0 0, 5 0, 5 3, 0 3, 0 0))',
            
            # Irregular polygon
            'POLYGON((0 0, 3 0, 4 2, 2 3, 0 2, 0 0))',
            
            # Multi-polygon AOI
            'MULTIPOLYGON(((0 0, 1 0, 1 1, 0 1, 0 0)), ((2 2, 3 2, 3 3, 2 3, 2 2)))'
        ]
        
        return geometries

    @staticmethod
    def create_model_training_data(
        num_samples=1000,
        image_size=64,
        change_probability=0.3
    ):
        """
        Create synthetic training data for ML models.
        
        Args:
            num_samples: Number of training samples
            image_size: Size of image patches
            change_probability: Probability of change in samples
        
        Returns:
            tuple: (X_train, y_train) - Training images and labels
        """
        X_train = []
        y_train = []
        
        for i in range(num_samples):
            # Create before image
            before_img = SampleDataGenerator.create_satellite_image(
                height=image_size, width=image_size, bands=6
            )
            
            # Determine if this sample has change
            has_change = np.random.rand() < change_probability
            
            if has_change:
                # Create change scenario
                _, after_img, _ = SampleDataGenerator.create_change_scenario(
                    before_image_shape=(image_size, image_size),
                    change_type=np.random.choice(['deforestation', 'urban_development']),
                    change_area_fraction=np.random.uniform(0.1, 0.5),
                    change_intensity=np.random.uniform(0.5, 1.0)
                )
                after_img = after_img[:, :, :6]  # Take only first 6 bands
                label = 1
            else:
                # No change - after image same as before with small variations
                after_img = before_img + np.random.normal(0, 0.01, before_img.shape)
                after_img = np.clip(after_img, 0, 1)
                label = 0
            
            # Stack before and after images
            sample = np.concatenate([before_img, after_img], axis=2)
            
            X_train.append(sample)
            y_train.append(label)
        
        return np.array(X_train), np.array(y_train)

    @staticmethod
    def save_sample_data(output_dir='tests/fixtures/data'):
        """
        Save sample data to files for reuse in tests.
        
        Args:
            output_dir: Directory to save sample data
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save sample images
        sample_image = SampleDataGenerator.create_satellite_image()
        np.save(os.path.join(output_dir, 'sample_image.npy'), sample_image)
        
        # Save time series data
        time_series = SampleDataGenerator.create_time_series_data()
        
        # Convert to DataFrame for easier handling
        ts_data = []
        for point in time_series:
            record = {
                'date': point.date.isoformat(),
                'cloud_cover': point.cloud_cover,
                'satellite': point.satellite,
                'quality_flag': point.quality_flag
            }
            record.update(point.values)
            ts_data.append(record)
        
        ts_df = pd.DataFrame(ts_data)
        ts_df.to_csv(os.path.join(output_dir, 'sample_time_series.csv'), index=False)
        
        # Save change scenarios
        scenarios = {}
        for change_type in ['deforestation', 'urban_development', 'water_change']:
            before, after, mask = SampleDataGenerator.create_change_scenario(
                change_type=change_type
            )
            scenarios[change_type] = {
                'before': before.tolist(),
                'after': after.tolist(),
                'mask': mask.tolist()
            }
        
        with open(os.path.join(output_dir, 'change_scenarios.json'), 'w') as f:
            json.dump(scenarios, f)
        
        # Save AOI geometries
        geometries = SampleDataGenerator.create_aoi_geometries()
        
        with open(os.path.join(output_dir, 'aoi_geometries.json'), 'w') as f:
            json.dump({'geometries': geometries}, f)
        
        print(f"Sample data saved to {output_dir}")


# Pytest fixtures for easy use in tests
import pytest

@pytest.fixture
def sample_satellite_image():
    """Fixture providing a sample satellite image."""
    return SampleDataGenerator.create_satellite_image()

@pytest.fixture
def sample_time_series():
    """Fixture providing sample time series data."""
    return SampleDataGenerator.create_time_series_data()

@pytest.fixture
def deforestation_scenario():
    """Fixture providing a deforestation change scenario."""
    return SampleDataGenerator.create_change_scenario(change_type='deforestation')

@pytest.fixture
def urban_development_scenario():
    """Fixture providing an urban development change scenario."""
    return SampleDataGenerator.create_change_scenario(change_type='urban_development')

@pytest.fixture
def model_training_data():
    """Fixture providing model training data."""
    return SampleDataGenerator.create_model_training_data(num_samples=100)

@pytest.fixture
def sample_aoi_geometries():
    """Fixture providing sample AOI geometries."""
    return SampleDataGenerator.create_aoi_geometries()


if __name__ == "__main__":
    # Generate and save sample data when run directly
    SampleDataGenerator.save_sample_data() 