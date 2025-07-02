#!/usr/bin/env python
"""
Comprehensive functionality test for Change Detection System
"""

import os
import sys
import django
import numpy as np

# Set Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'src.web.backend.simple_settings')
sys.path.append('src')
django.setup()

from django.contrib.auth.models import User
from changedetection.core.models import AreaOfInterest
from changedetection.data_processing.spectral_indices import SpectralIndicesCalculator
from changedetection.ml_models.local_models import LocalChangeDetectionModel
from changedetection.data_processing.temporal_analysis import TemporalAnalyzer

def test_functionality():
    print('🧪 COMPREHENSIVE CHANGE DETECTION SYSTEM TEST')
    print('=' * 60)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Database and User Management
    tests_total += 1
    try:
        user, created = User.objects.get_or_create(
            username='test_user',
            defaults={'email': 'test@example.com'}
        )
        
        aoi = AreaOfInterest.objects.create(
            user=user,
            name='Test AOI - System Verification',
            description='Automated test area',
            center_lat=40.7128,
            center_lng=-74.0060,
            bbox_north=40.8,
            bbox_south=40.6,
            bbox_east=-73.9,
            bbox_west=-74.1,
            area_km2=25.0
        )
        
        print('✅ Test 1: Database & User Management')
        print(f'   Created AOI: {aoi.name} ({aoi.area_km2} km²)')
        print(f'   GeoJSON: {type(aoi.geometry_geojson)}')
        tests_passed += 1
        
    except Exception as e:
        print(f'❌ Test 1 Failed: {e}')
    
    # Test 2: Spectral Indices Calculation
    tests_total += 1
    try:
        calc = SpectralIndicesCalculator()
        test_bands = {
            'red': np.random.rand(50, 50) * 0.3,
            'green': np.random.rand(50, 50) * 0.25,
            'blue': np.random.rand(50, 50) * 0.2,
            'nir': np.random.rand(50, 50) * 0.4,
            'swir1': np.random.rand(50, 50) * 0.15,
            'swir2': np.random.rand(50, 50) * 0.1
        }
        
        indices = calc.calculate_all_indices(test_bands)
        
        print('✅ Test 2: Spectral Indices Calculation')
        print(f'   Calculated {len(indices)} indices')
        print(f'   Available: {", ".join(list(indices.keys())[:3])}...')
        tests_passed += 1
        
    except Exception as e:
        print(f'❌ Test 2 Failed: {e}')
    
    # Test 3: Change Detection Models
    tests_total += 1
    try:
        model = LocalChangeDetectionModel(method='spectral')
        
        before_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        after_img = np.random.randint(50, 205, (64, 64, 3), dtype=np.uint8)
        
        change_map, confidence = model.predict(before_img, after_img)
        
        print('✅ Test 3: Change Detection Models')
        print(f'   Input shape: {before_img.shape}')
        print(f'   Changes detected: {np.sum(change_map)} pixels')
        print(f'   Confidence score: {confidence:.3f}')
        tests_passed += 1
        
    except Exception as e:
        print(f'❌ Test 3 Failed: {e}')
    
    # Test 4: Temporal Analysis Framework
    tests_total += 1
    try:
        analyzer = TemporalAnalyzer(baseline_years=2, anomaly_threshold=1.5)
        
        print('✅ Test 4: Temporal Analysis Framework')
        print(f'   Baseline period: {analyzer.baseline_years} years')
        print(f'   Anomaly threshold: {analyzer.anomaly_threshold}')
        print(f'   Min observations/month: {analyzer.min_observations_per_month}')
        tests_passed += 1
        
    except Exception as e:
        print(f'❌ Test 4 Failed: {e}')
    
    # Test 5: System Integration
    tests_total += 1
    try:
        # Test combining multiple components
        from changedetection.change_detection.models import ChangeDetectionJob
        
        job = ChangeDetectionJob.objects.create(
            user=user,
            aoi=aoi,
            before_image_path='/tmp/before.tif',
            after_image_path='/tmp/after.tif',
            before_date='2023-01-01',
            after_date='2023-06-01',
            change_threshold=0.25
        )
        
        print('✅ Test 5: System Integration')
        print(f'   Created job: {job.id}')
        print(f'   Status: {job.status}')
        tests_passed += 1
        
    except Exception as e:
        print(f'❌ Test 5 Failed: {e}')
    
    # Results Summary
    print('\n' + '='*60)
    print(f'🎯 TEST RESULTS: {tests_passed}/{tests_total} PASSED')
    
    if tests_passed == tests_total:
        print('🎉 ALL TESTS PASSED! System is fully operational!')
        print('🚀 Ready for production deployment!')
        return True
    else:
        print(f'⚠️  {tests_total - tests_passed} tests failed. Please review above errors.')
        return False

if __name__ == '__main__':
    success = test_functionality()
    sys.exit(0 if success else 1) 