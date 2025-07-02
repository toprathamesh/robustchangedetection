#!/usr/bin/env python
"""
Installation verification script for Change Detection System.
Tests the reorganized codebase and ensures all core functionality works.
"""

import sys
import os
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

def test_package_imports():
    """Test that all core packages can be imported."""
    print("🔍 Testing package imports...")
    
    try:
        import changedetection
        print("✅ Main package import successful")
        print(f"📦 Version: {changedetection.__version__}")
    except Exception as e:
        print(f"❌ Main package import failed: {e}")
        return False
    
    # Test core module imports
    modules_to_test = [
        'changedetection.core.models',
        'changedetection.data_processing.temporal_analysis',
        'changedetection.data_processing.spectral_indices',
        'changedetection.change_detection.advanced_models',
        'changedetection.change_detection.explainability',
        'changedetection.data_processing.automated_workflows',
        'changedetection.alerts.tasks'
    ]
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✅ {module}")
        except Exception as e:
            print(f"❌ {module}: {e}")
            return False
    
    return True

def test_core_functionality():
    """Test core functionality without external dependencies."""
    print("\n🧪 Testing core functionality...")
    
    try:
        # Test temporal analysis
        from changedetection.data_processing.temporal_analysis import TemporalAnalyzer, TimeSeriesPoint
        from datetime import datetime
        
        analyzer = TemporalAnalyzer()
        point = TimeSeriesPoint(
            date=datetime.now(),
            values={'ndvi': 0.8},
            cloud_cover=5.0,
            satellite='sentinel2'
        )
        analyzer.add_observation(point)
        print("✅ Temporal analysis functionality")
        
    except Exception as e:
        print(f"❌ Temporal analysis: {e}")
        return False
    
    try:
        # Test spectral indices
        from changedetection.data_processing.spectral_indices import SpectralIndicesCalculator, SatellitePlatform
        import numpy as np
        
        calculator = SpectralIndicesCalculator(SatellitePlatform.SENTINEL2)
        test_image = np.random.rand(50, 50, 10) * 0.3
        result = calculator.calculate_all_indices(test_image)
        assert 'indices' in result
        assert 'ndvi' in result['indices']
        print("✅ Spectral indices functionality")
        
    except Exception as e:
        print(f"❌ Spectral indices: {e}")
        return False
    
    try:
        # Test change detection models (basic initialization)
        from changedetection.change_detection.advanced_models import EnsembleChangeDetector
        
        detector = EnsembleChangeDetector()
        print("✅ Change detection models")
        
    except Exception as e:
        print(f"❌ Change detection models: {e}")
        return False
    
    return True

def test_django_configuration():
    """Test Django configuration and apps."""
    print("\n🌐 Testing Django configuration...")
    
    try:
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'web.backend.simple_settings')
        
        import django
        django.setup()
        
        from django.apps import apps
        from django.conf import settings
        
        # Check that our apps are installed
        required_apps = [
            'changedetection.core',
            'changedetection.change_detection',
            'changedetection.data_processing',
            'changedetection.alerts'
        ]
        
        installed_apps = settings.INSTALLED_APPS
        
        for app in required_apps:
            if app in installed_apps:
                print(f"✅ Django app: {app}")
            else:
                print(f"❌ Django app not found: {app}")
                return False
        
        print("✅ Django configuration successful")
        return True
        
    except Exception as e:
        print(f"❌ Django configuration: {e}")
        return False

def test_file_structure():
    """Test that all essential files are in place."""
    print("\n📁 Testing file structure...")
    
    essential_files = [
        'src/changedetection/__init__.py',
        'src/changedetection/core/models.py',
        'src/changedetection/data_processing/temporal_analysis.py',
        'src/changedetection/data_processing/spectral_indices.py',
        'src/changedetection/change_detection/advanced_models.py',
        'src/changedetection/change_detection/explainability.py',
        'src/web/backend/settings.py',
        'src/manage.py',
        'config/requirements.txt',
        'pyproject.toml',
        'README.md'
    ]
    
    missing_files = []
    for file_path in essential_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  {len(missing_files)} essential files are missing")
        return False
    
    print("✅ All essential files present")
    return True

def main():
    """Run all verification tests."""
    print("🚀 Change Detection System - Installation Verification")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Run all tests
    tests = [
        test_file_structure,
        test_package_imports,
        test_core_functionality,
        test_django_configuration
    ]
    
    for test in tests:
        if not test():
            all_tests_passed = False
        print()  # Add spacing between tests
    
    # Final result
    print("=" * 60)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! Installation verification successful.")
        print("\n📝 Next steps:")
        print("   1. Install dependencies: pip install -e .[dev]")
        print("   2. Set up environment: cp config/.env.example .env")
        print("   3. Configure database and run migrations")
        print("   4. Start the development server: cd src && python manage.py runserver")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED! Please fix the issues above.")
        sys.exit(1)

if __name__ == '__main__':
    main() 