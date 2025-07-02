"""
Unit tests for temporal analysis module.
Tests the core functionality of temporal baselines, anomaly detection, and trend analysis.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.changedetection.data_processing.temporal_analysis import (
    TemporalAnalyzer, 
    TimeSeriesPoint, 
    SeasonalBaseline,
    ChangeTypeClassifier
)


class TestTimeSeriesPoint:
    """Test TimeSeriesPoint data structure."""

    def test_time_series_point_creation(self):
        """Test creating a TimeSeriesPoint instance."""
        date = datetime(2023, 6, 15)
        values = {'ndvi': 0.8, 'ndbi': 0.2}
        
        point = TimeSeriesPoint(
            date=date,
            values=values,
            cloud_cover=10.0,
            satellite='sentinel2'
        )
        
        assert point.date == date
        assert point.values == values
        assert point.cloud_cover == 10.0
        assert point.satellite == 'sentinel2'
        assert point.quality_flag == 'good'  # default


class TestSeasonalBaseline:
    """Test SeasonalBaseline data structure."""

    def test_seasonal_baseline_creation(self):
        """Test creating a SeasonalBaseline instance."""
        baseline = SeasonalBaseline(
            month=6,
            mean_values={'ndvi': 0.75},
            std_values={'ndvi': 0.1},
            percentiles={'ndvi': {5: 0.5, 95: 0.9}},
            sample_count=20,
            confidence_interval={'ndvi': (0.7, 0.8)}
        )
        
        assert baseline.month == 6
        assert baseline.mean_values['ndvi'] == 0.75
        assert baseline.sample_count == 20


class TestTemporalAnalyzer:
    """Test TemporalAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.analyzer = TemporalAnalyzer(
            baseline_years=2,
            min_observations_per_month=3,
            anomaly_threshold=2.0
        )

    def test_initialization(self):
        """Test TemporalAnalyzer initialization."""
        assert self.analyzer.baseline_years == 2
        assert self.analyzer.min_observations_per_month == 3
        assert self.analyzer.anomaly_threshold == 2.0
        assert len(self.analyzer.time_series_data) == 0
        assert not self.analyzer.is_fitted

    def test_add_observation(self):
        """Test adding time series observations."""
        observation = TimeSeriesPoint(
            date=datetime(2023, 6, 15),
            values={'ndvi': 0.8},
            cloud_cover=5.0,
            satellite='sentinel2'
        )
        
        self.analyzer.add_observation(observation)
        assert len(self.analyzer.time_series_data) == 1
        assert self.analyzer.time_series_data[0] == observation

    def test_dataframe_to_timeseries_conversion(self):
        """Test converting DataFrame to TimeSeriesPoint list."""
        df = pd.DataFrame({
            'date': ['2023-06-15', '2023-07-15'],
            'ndvi': [0.8, 0.7],
            'ndbi': [0.2, 0.3],
            'cloud_cover': [5.0, 10.0],
            'satellite': ['sentinel2', 'sentinel2']
        })
        
        time_series = self.analyzer._dataframe_to_timeseries(df)
        
        assert len(time_series) == 2
        assert time_series[0].values['ndvi'] == 0.8
        assert time_series[1].values['ndbi'] == 0.3

    def create_sample_data(self, start_date, num_months=24):
        """Helper method to create sample time series data."""
        for i in range(num_months):
            for day in [5, 15, 25]:  # 3 observations per month
                date = start_date + timedelta(days=30*i + day)
                # Simulate seasonal NDVI pattern
                seasonal_base = 0.6 + 0.3 * np.sin(2 * np.pi * date.month / 12)
                noise = np.random.normal(0, 0.05)
                
                observation = TimeSeriesPoint(
                    date=date,
                    values={'ndvi': seasonal_base + noise},
                    cloud_cover=np.random.uniform(0, 20),
                    satellite='sentinel2'
                )
                self.analyzer.add_observation(observation)

    def test_build_seasonal_baselines(self):
        """Test building seasonal baselines from historical data."""
        # Create sample data
        start_date = datetime(2021, 1, 1)
        self.create_sample_data(start_date)
        
        # Build baselines
        self.analyzer.build_seasonal_baselines()
        
        # Check that baselines were created
        assert len(self.analyzer.seasonal_baselines) > 0
        
        # Check specific month baseline
        june_baseline = self.analyzer.seasonal_baselines.get(6)
        if june_baseline:
            assert 'ndvi' in june_baseline.mean_values
            assert june_baseline.sample_count >= self.analyzer.min_observations_per_month

    def test_fit_anomaly_detector(self):
        """Test fitting the anomaly detection model."""
        # Create sample data
        start_date = datetime(2021, 1, 1)
        self.create_sample_data(start_date)
        
        # Build baselines first
        self.analyzer.build_seasonal_baselines()
        
        # Fit anomaly detector
        self.analyzer.fit_anomaly_detector()
        
        assert self.analyzer.is_fitted

    def test_detect_anthropogenic_change_normal(self):
        """Test change detection with normal observation."""
        # Setup with sample data
        start_date = datetime(2021, 1, 1)
        self.create_sample_data(start_date)
        self.analyzer.build_seasonal_baselines()
        self.analyzer.fit_anomaly_detector()
        
        # Create normal observation
        normal_obs = TimeSeriesPoint(
            date=datetime(2023, 6, 15),
            values={'ndvi': 0.85},  # Normal June value
            cloud_cover=5.0,
            satellite='sentinel2'
        )
        
        result = self.analyzer.detect_anthropogenic_change(normal_obs)
        
        assert 'is_anomaly' in result
        assert 'change_probability' in result
        assert 'confidence' in result

    def test_detect_anthropogenic_change_anomaly(self):
        """Test change detection with anomalous observation."""
        # Setup with sample data
        start_date = datetime(2021, 1, 1)
        self.create_sample_data(start_date)
        self.analyzer.build_seasonal_baselines()
        self.analyzer.fit_anomaly_detector()
        
        # Create anomalous observation
        anomaly_obs = TimeSeriesPoint(
            date=datetime(2023, 6, 15),
            values={'ndvi': 0.2},  # Very low June value (potential deforestation)
            cloud_cover=5.0,
            satellite='sentinel2'
        )
        
        result = self.analyzer.detect_anthropogenic_change(anomaly_obs)
        
        assert result['is_anomaly'] is True
        assert result['change_probability'] > 0.5

    def test_analyze_trends(self):
        """Test trend analysis functionality."""
        # Setup with sample data
        start_date = datetime(2021, 1, 1)
        self.create_sample_data(start_date)
        
        result = self.analyzer.analyze_trends('ndvi', window_months=12)
        
        assert 'trend_slope' in result
        assert 'trend_significance' in result
        assert 'trend_direction' in result

    def test_save_and_load_baselines(self, tmp_path):
        """Test saving and loading seasonal baselines."""
        # Setup and create baselines
        start_date = datetime(2021, 1, 1)
        self.create_sample_data(start_date)
        self.analyzer.build_seasonal_baselines()
        
        # Save baselines
        save_path = tmp_path / "baselines.pkl"
        self.analyzer.save_baselines(str(save_path))
        
        # Create new analyzer and load baselines
        new_analyzer = TemporalAnalyzer()
        new_analyzer.load_baselines(str(save_path))
        
        # Verify baselines were loaded
        assert len(new_analyzer.seasonal_baselines) == len(self.analyzer.seasonal_baselines)

    def test_insufficient_data_error(self):
        """Test error handling with insufficient data."""
        # Try to build baselines with no data
        with pytest.raises(ValueError, match="Need at least 12 months"):
            self.analyzer.build_seasonal_baselines()


class TestChangeTypeClassifier:
    """Test ChangeTypeClassifier class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = ChangeTypeClassifier()

    def test_initialization(self):
        """Test ChangeTypeClassifier initialization."""
        assert len(self.classifier.change_signatures) > 0
        assert 'urban_development' in self.classifier.change_signatures

    def test_classify_urban_development(self):
        """Test classification of urban development."""
        before_indices = {'ndvi': 0.8, 'ndbi': 0.1}
        after_indices = {'ndvi': 0.2, 'ndbi': 0.7}  # Vegetation loss, built-up increase
        temporal_context = {'season': 'summer', 'trend': 'decreasing'}
        
        result = self.classifier.classify_change(
            before_indices, after_indices, temporal_context
        )
        
        assert result['predicted_type'] in self.classifier.change_signatures.keys()
        assert 'confidence' in result
        assert 'type_scores' in result

    def test_classify_deforestation(self):
        """Test classification of deforestation."""
        before_indices = {'ndvi': 0.9, 'ndbi': 0.1}
        after_indices = {'ndvi': 0.2, 'ndbi': 0.2}  # Significant vegetation loss
        temporal_context = {'season': 'summer', 'trend': 'stable'}
        
        result = self.classifier.classify_change(
            before_indices, after_indices, temporal_context
        )
        
        assert 'predicted_type' in result
        assert result['confidence'] > 0

    def test_classify_no_significant_change(self):
        """Test classification when there's no significant change."""
        before_indices = {'ndvi': 0.8, 'ndbi': 0.2}
        after_indices = {'ndvi': 0.78, 'ndbi': 0.22}  # Minor variations
        temporal_context = {'season': 'summer', 'trend': 'stable'}
        
        result = self.classifier.classify_change(
            before_indices, after_indices, temporal_context
        )
        
        # Should classify as no significant change or low confidence
        assert result['predicted_type'] is not None


# Integration test fixtures
@pytest.fixture
def sample_temporal_data():
    """Fixture providing sample temporal data for testing."""
    data = []
    start_date = datetime(2021, 1, 1)
    
    for i in range(24):  # 2 years of data
        for day in [5, 15, 25]:
            date = start_date + timedelta(days=30*i + day)
            seasonal_base = 0.6 + 0.3 * np.sin(2 * np.pi * date.month / 12)
            
            point = TimeSeriesPoint(
                date=date,
                values={'ndvi': seasonal_base + np.random.normal(0, 0.02)},
                cloud_cover=np.random.uniform(0, 15),
                satellite='sentinel2'
            )
            data.append(point)
    
    return data


def test_full_temporal_analysis_workflow(sample_temporal_data):
    """Integration test for complete temporal analysis workflow."""
    analyzer = TemporalAnalyzer()
    
    # Add all data
    for point in sample_temporal_data:
        analyzer.add_observation(point)
    
    # Build baselines
    analyzer.build_seasonal_baselines()
    assert len(analyzer.seasonal_baselines) > 0
    
    # Fit anomaly detector
    analyzer.fit_anomaly_detector()
    assert analyzer.is_fitted
    
    # Test detection on new observation
    test_obs = TimeSeriesPoint(
        date=datetime(2023, 6, 15),
        values={'ndvi': 0.85},
        cloud_cover=5.0,
        satellite='sentinel2'
    )
    
    result = analyzer.detect_anthropogenic_change(test_obs)
    assert 'is_anomaly' in result
    assert 'change_probability' in result


# Performance tests
@pytest.mark.slow
def test_temporal_analysis_performance():
    """Test temporal analysis performance with large dataset."""
    analyzer = TemporalAnalyzer()
    
    # Create large dataset (5 years, daily observations)
    start_date = datetime(2018, 1, 1)
    
    import time
    start_time = time.time()
    
    for i in range(365 * 5):  # 5 years daily
        date = start_date + timedelta(days=i)
        seasonal_base = 0.6 + 0.3 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
        
        point = TimeSeriesPoint(
            date=date,
            values={'ndvi': seasonal_base + np.random.normal(0, 0.02)},
            cloud_cover=np.random.uniform(0, 20),
            satellite='sentinel2'
        )
        analyzer.add_observation(point)
    
    # Build baselines
    analyzer.build_seasonal_baselines()
    analyzer.fit_anomaly_detector()
    
    processing_time = time.time() - start_time
    
    # Should process in reasonable time (< 10 seconds for 5 years daily data)
    assert processing_time < 10.0
    assert analyzer.is_fitted


if __name__ == "__main__":
    pytest.main([__file__]) 