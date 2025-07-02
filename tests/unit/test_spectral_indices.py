"""
Unit tests for spectral indices module.
Tests calculation of various spectral indices and quality assessment.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.changedetection.data_processing.spectral_indices import (
    SpectralIndicesCalculator,
    BandConfiguration,
    SatellitePlatform,
    MultiTemporalIndicesAnalyzer
)


class TestBandConfiguration:
    """Test BandConfiguration class."""

    def test_sentinel2_config(self):
        """Test Sentinel-2 band configuration."""
        config = BandConfiguration.get_sentinel2_config()
        
        assert config.platform == SatellitePlatform.SENTINEL2
        assert 'red' in config.bands
        assert 'nir' in config.bands
        assert 'blue' in config.bands
        assert config.scale_factor == 0.0001

    def test_landsat8_config(self):
        """Test Landsat-8 band configuration."""
        config = BandConfiguration.get_landsat8_config()
        
        assert config.platform == SatellitePlatform.LANDSAT8
        assert 'red' in config.bands
        assert 'nir' in config.bands
        assert config.scale_factor == 0.0001


class TestSpectralIndicesCalculator:
    """Test SpectralIndicesCalculator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = SpectralIndicesCalculator(SatellitePlatform.SENTINEL2)

    def create_mock_image(self, height=100, width=100):
        """Create mock satellite image for testing."""
        # Create mock Sentinel-2 image with 10 bands
        image = np.random.rand(height, width, 10) * 0.3  # Typical reflectance values
        
        # Set realistic band values
        image[:, :, 0] = np.random.uniform(0.02, 0.15, (height, width))  # Blue
        image[:, :, 1] = np.random.uniform(0.03, 0.18, (height, width))  # Green
        image[:, :, 2] = np.random.uniform(0.02, 0.20, (height, width))  # Red
        image[:, :, 6] = np.random.uniform(0.15, 0.50, (height, width))  # NIR
        image[:, :, 8] = np.random.uniform(0.05, 0.25, (height, width))  # SWIR1
        image[:, :, 9] = np.random.uniform(0.02, 0.15, (height, width))  # SWIR2
        
        return image

    def test_initialization(self):
        """Test calculator initialization."""
        assert self.calculator.platform == SatellitePlatform.SENTINEL2
        assert self.calculator.band_config.platform == SatellitePlatform.SENTINEL2

    def test_calculate_all_indices(self):
        """Test calculating all available indices."""
        image = self.create_mock_image()
        
        result = self.calculator.calculate_all_indices(image)
        
        assert 'indices' in result
        assert 'quality_info' in result
        assert 'metadata' in result
        
        # Check that basic indices are calculated
        indices = result['indices']
        assert 'ndvi' in indices
        assert 'ndwi' in indices
        assert 'ndbi' in indices

    def test_calculate_all_indices_no_quality(self):
        """Test calculating indices without quality assessment."""
        image = self.create_mock_image()
        
        result = self.calculator.calculate_all_indices(image, return_quality_info=False)
        
        assert result['quality_info'] == {}

    def test_ndvi_calculation(self):
        """Test NDVI calculation specifically."""
        # Create image with known values
        height, width = 50, 50
        image = np.zeros((height, width, 10))
        
        # Set red and NIR bands with known values
        image[:, :, 2] = 0.1   # Red
        image[:, :, 6] = 0.4   # NIR
        
        result = self.calculator.calculate_all_indices(image)
        ndvi = result['indices']['ndvi']
        
        # NDVI = (NIR - Red) / (NIR + Red) = (0.4 - 0.1) / (0.4 + 0.1) = 0.6
        expected_ndvi = 0.6
        assert np.allclose(ndvi, expected_ndvi, atol=0.01)

    def test_ndwi_calculation(self):
        """Test NDWI calculation."""
        height, width = 50, 50
        image = np.zeros((height, width, 10))
        
        # Set green and NIR bands
        image[:, :, 1] = 0.2   # Green
        image[:, :, 6] = 0.3   # NIR
        
        result = self.calculator.calculate_all_indices(image)
        ndwi = result['indices']['ndwi']
        
        # NDWI = (Green - NIR) / (Green + NIR) = (0.2 - 0.3) / (0.2 + 0.3) = -0.2
        expected_ndwi = -0.2
        assert np.allclose(ndwi, expected_ndwi, atol=0.01)

    def test_evi_calculation(self):
        """Test EVI calculation."""
        height, width = 50, 50
        image = np.zeros((height, width, 10))
        
        # Set RGB and NIR bands
        image[:, :, 0] = 0.05  # Blue
        image[:, :, 1] = 0.1   # Green
        image[:, :, 2] = 0.08  # Red
        image[:, :, 6] = 0.4   # NIR
        
        result = self.calculator.calculate_all_indices(image)
        evi = result['indices']['evi']
        
        # EVI should be calculated correctly
        assert np.all(np.isfinite(evi))
        assert np.all(evi >= -1) and np.all(evi <= 1)

    def test_safe_divide(self):
        """Test safe division utility."""
        numerator = np.array([1, 2, 3])
        denominator = np.array([2, 0, 1])  # Include division by zero
        
        result = self.calculator._safe_divide(numerator, denominator, fill_value=-999)
        
        assert result[0] == 0.5    # 1/2
        assert result[1] == -999   # 2/0 -> fill_value
        assert result[2] == 3.0    # 3/1

    def test_get_band(self):
        """Test band extraction."""
        image = self.create_mock_image()
        
        red_band = self.calculator._get_band(image, 'red')
        
        assert red_band.shape == (100, 100)
        assert np.all(red_band >= 0)

    def test_invalid_band_name(self):
        """Test handling of invalid band names."""
        image = self.create_mock_image()
        
        with pytest.raises(KeyError):
            self.calculator._get_band(image, 'invalid_band')

    def test_quality_assessment(self):
        """Test index quality assessment."""
        # Create image with good NDVI values
        height, width = 50, 50
        image = np.zeros((height, width, 10))
        image[:, :, 2] = 0.1   # Red
        image[:, :, 6] = 0.4   # NIR
        
        result = self.calculator.calculate_all_indices(image)
        quality_info = result['quality_info']['ndvi']
        
        assert 'quality' in quality_info
        assert 'statistics' in quality_info
        assert quality_info['quality'] in ['poor', 'fair', 'good']

    def test_change_vector_analysis(self):
        """Test change vector analysis between two time periods."""
        # Create before and after indices
        before_indices = {
            'ndvi': np.random.uniform(0.6, 0.8, (50, 50)),
            'ndbi': np.random.uniform(0.1, 0.3, (50, 50))
        }
        
        after_indices = {
            'ndvi': np.random.uniform(0.2, 0.4, (50, 50)),  # Decreased vegetation
            'ndbi': np.random.uniform(0.5, 0.7, (50, 50))   # Increased built-up
        }
        
        result = self.calculator.calculate_change_vector_analysis(
            before_indices, after_indices
        )
        
        assert 'change_magnitude' in result
        assert 'change_direction' in result
        assert 'magnitude_statistics' in result
        assert result['change_magnitude'].shape == (50, 50)

    def test_detect_change_hotspots(self):
        """Test change hotspot detection."""
        # Create change magnitude map with some hotspots
        change_magnitude = np.random.exponential(0.1, (100, 100))
        # Add some hotspots
        change_magnitude[20:30, 20:30] = 0.8  # High change area
        
        result = self.calculator.detect_change_hotspots(
            change_magnitude, 
            percentile_threshold=90.0
        )
        
        assert 'hotspot_mask' in result
        assert 'hotspot_clusters' in result
        assert 'statistics' in result

    def test_image_shape_validation(self):
        """Test validation of image shape."""
        # Create 2D image (should fail)
        image_2d = np.random.rand(100, 100)
        
        with pytest.raises(ValueError, match="Image must be 3D array"):
            self.calculator.calculate_all_indices(image_2d)

    def test_mask_application(self):
        """Test application of mask during calculation."""
        image = self.create_mock_image()
        mask = np.zeros((100, 100), dtype=bool)
        mask[40:60, 40:60] = True  # Mask central area
        
        result = self.calculator.calculate_all_indices(image, mask=mask)
        
        # Check that masked pixels are handled
        assert 'metadata' in result
        assert result['metadata']['masked_pixels'] == np.sum(mask)


class TestMultiTemporalIndicesAnalyzer:
    """Test MultiTemporalIndicesAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = MultiTemporalIndicesAnalyzer()

    def test_initialization(self):
        """Test analyzer initialization."""
        assert len(self.analyzer.time_series) == 0

    def test_add_image_data(self):
        """Test adding image data to time series."""
        image = np.random.rand(50, 50, 10) * 0.3
        
        self.analyzer.add_image_data(
            date='2023-06-15',
            image=image,
            platform=SatellitePlatform.SENTINEL2
        )
        
        assert len(self.analyzer.time_series) == 1

    def test_analyze_temporal_trends(self):
        """Test temporal trend analysis."""
        # Add multiple time points
        for i in range(12):  # 12 months of data
            image = np.random.rand(30, 30, 10) * 0.3
            date = f'2023-{i+1:02d}-15'
            
            self.analyzer.add_image_data(
                date=date,
                image=image,
                platform=SatellitePlatform.SENTINEL2
            )
        
        result = self.analyzer.analyze_temporal_trends('ndvi')
        
        assert 'trend_slope' in result
        assert 'trend_statistics' in result


# Edge cases and error handling tests
class TestEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = SpectralIndicesCalculator(SatellitePlatform.GENERIC)

    def test_zero_denominator_handling(self):
        """Test handling of zero denominators in index calculations."""
        height, width = 50, 50
        image = np.zeros((height, width, 6))  # All zeros
        
        result = self.calculator.calculate_all_indices(image)
        
        # Should handle zero denominators gracefully
        indices = result['indices']
        for index_name, index_data in indices.items():
            assert np.all(np.isfinite(index_data) | (index_data == 0))

    def test_extreme_values(self):
        """Test handling of extreme input values."""
        height, width = 50, 50
        image = np.ones((height, width, 6)) * 2.0  # Values > 1 (unusual but possible)
        
        result = self.calculator.calculate_all_indices(image)
        
        # Should still calculate indices
        assert 'ndvi' in result['indices']

    def test_nan_handling(self):
        """Test handling of NaN values in input."""
        height, width = 50, 50
        image = np.random.rand(height, width, 6) * 0.3
        image[20:30, 20:30, :] = np.nan  # Introduce NaN values
        
        result = self.calculator.calculate_all_indices(image)
        
        # Should handle NaN values
        assert 'ndvi' in result['indices']

    def test_empty_change_vectors(self):
        """Test change vector analysis with empty inputs."""
        before_indices = {}
        after_indices = {}
        
        with pytest.raises(ValueError, match="No common indices"):
            self.calculator.calculate_change_vector_analysis(
                before_indices, after_indices
            )


# Performance tests
@pytest.mark.slow
def test_large_image_performance():
    """Test performance with large images."""
    calculator = SpectralIndicesCalculator(SatellitePlatform.SENTINEL2)
    
    # Create large image
    large_image = np.random.rand(1000, 1000, 10) * 0.3
    
    import time
    start_time = time.time()
    
    result = calculator.calculate_all_indices(large_image)
    
    processing_time = time.time() - start_time
    
    # Should process large image in reasonable time (< 5 seconds)
    assert processing_time < 5.0
    assert 'ndvi' in result['indices']


# Integration tests
def test_full_spectral_analysis_workflow():
    """Integration test for complete spectral analysis workflow."""
    calculator = SpectralIndicesCalculator(SatellitePlatform.SENTINEL2)
    
    # Create realistic image
    height, width = 200, 200
    image = np.random.rand(height, width, 10) * 0.3
    
    # Add realistic band values
    image[:, :, 2] = np.random.uniform(0.05, 0.15, (height, width))  # Red
    image[:, :, 6] = np.random.uniform(0.20, 0.45, (height, width))  # NIR
    
    # Calculate indices
    result = calculator.calculate_all_indices(image, return_quality_info=True)
    
    # Verify comprehensive results
    assert len(result['indices']) >= 5  # Should have multiple indices
    assert len(result['quality_info']) == len(result['indices'])
    
    # Test change analysis
    # Create second image with changes
    image2 = image.copy()
    image2[:, :, 6] *= 0.5  # Reduce NIR (vegetation loss)
    
    result2 = calculator.calculate_all_indices(image2)
    
    # Analyze changes
    change_result = calculator.calculate_change_vector_analysis(
        result['indices'], result2['indices']
    )
    
    assert change_result['change_magnitude'].max() > 0  # Should detect changes


if __name__ == "__main__":
    pytest.main([__file__]) 