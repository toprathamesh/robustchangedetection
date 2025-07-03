#!/usr/bin/env python
"""
Change Detection Pipeline Test
=============================
Creates sample images and tests the complete pipeline to ensure everything works.
"""

import numpy as np
import cv2
from pathlib import Path
import sys
import subprocess
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_images():
    """Create before and after test images with clear changes"""
    
    logger.info("🎨 Creating test images...")
    
    # Create base image (512x512 RGB)
    height, width = 512, 512
    before_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add background with some texture
    before_image[:, :] = [30, 60, 120]  # Blue background
    
    # Add some noise for realism
    noise = np.random.randint(-20, 20, (height, width, 3)).astype(np.int16)
    before_image = np.clip(before_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Add some geometric shapes
    cv2.rectangle(before_image, (50, 50), (150, 150), (255, 255, 255), -1)  # White square
    cv2.circle(before_image, (300, 300), 60, (255, 0, 0), -1)  # Red circle
    cv2.rectangle(before_image, (400, 100), (480, 180), (0, 255, 0), -1)  # Green rectangle
    
    # Create after image (copy + changes)
    after_image = before_image.copy()
    
    # Add changes
    cv2.rectangle(after_image, (200, 200), (350, 350), (0, 255, 255), -1)  # Yellow square (NEW)
    cv2.circle(after_image, (300, 300), 60, (0, 0, 255), -1)  # Change red circle to blue
    # Remove green rectangle
    cv2.rectangle(after_image, (400, 100), (480, 180), (30, 60, 120), -1)  # Paint over with background
    # Add new building-like structure
    cv2.rectangle(after_image, (100, 350), (200, 450), (128, 128, 128), -1)  # Gray building
    cv2.rectangle(after_image, (110, 360), (190, 440), (64, 64, 64), -1)  # Dark windows
    
    # Save images
    before_path = "test_before.png"
    after_path = "test_after.png"
    
    cv2.imwrite(before_path, cv2.cvtColor(before_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(after_path, cv2.cvtColor(after_image, cv2.COLOR_RGB2BGR))
    
    logger.info(f"✅ Created test images:")
    logger.info(f"   📁 {before_path} ({before_image.shape})")
    logger.info(f"   📁 {after_path} ({after_image.shape})")
    
    return before_path, after_path

def test_direct_api():
    """Test the change detection API directly"""
    
    logger.info("\n🧪 Testing Direct API...")
    
    before_path, after_path = create_test_images()
    
    try:
        # Test import
        from changedetection import create_detector, quick_predict
        logger.info("✅ Successfully imported changedetection modules")
        
        # Test model creation
        detector = create_detector('siamese_unet')
        logger.info("✅ Successfully created detector")
        
        # Test prediction
        results = detector.predict(before_path, after_path)
        logger.info(f"✅ Prediction successful: {results['change_percentage']:.2f}% change detected")
        
        # Test visualization
        detector.visualize_results(before_path, after_path, results, 
                                 save_path="test_api_results.png")
        logger.info("✅ Visualization successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Direct API test failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False
    
    finally:
        # Cleanup
        for file in [before_path, after_path]:
            if Path(file).exists():
                Path(file).unlink()

def test_main_script():
    """Test the main.py script"""
    
    logger.info("\n🧪 Testing Main Script...")
    
    before_path, after_path = create_test_images()
    
    try:
        # Test basic functionality
        cmd = [sys.executable, "main.py", "--before", before_path, "--after", after_path, 
               "--model", "siamese_unet"]
        logger.info(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            logger.info("✅ Main script executed successfully")
            logger.info(f"Output: {result.stdout[-200:]}")  # Last 200 chars
            return True
        else:
            logger.error(f"❌ Main script failed with return code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Main script timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Error running main script: {e}")
        return False
    
    finally:
        # Cleanup
        for file in [before_path, after_path]:
            if Path(file).exists():
                Path(file).unlink()

def test_model_comparison():
    """Test model comparison functionality"""
    
    logger.info("\n🧪 Testing Model Comparison...")
    
    before_path, after_path = create_test_images()
    
    try:
        cmd = [sys.executable, "main.py", "--compare", "--before", before_path, "--after", after_path]
        logger.info(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        
        if result.returncode == 0:
            logger.info("✅ Model comparison executed successfully")
            logger.info(f"Output: {result.stdout[-300:]}")  # Last 300 chars
            return True
        else:
            logger.error(f"❌ Model comparison failed with return code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Model comparison timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Error running model comparison: {e}")
        return False
    
    finally:
        # Cleanup
        for file in [before_path, after_path]:
            if Path(file).exists():
                Path(file).unlink()

def test_custom_detection():
    """Test custom detection with different thresholds"""
    
    logger.info("\n🧪 Testing Custom Detection...")
    
    before_path, after_path = create_test_images()
    
    try:
        cmd = [sys.executable, "main.py", "--custom", "--before", before_path, "--after", after_path,
               "--threshold", "0.3", "--model", "siamese_unet"]
        logger.info(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            logger.info("✅ Custom detection executed successfully")
            logger.info(f"Output: {result.stdout[-200:]}")  # Last 200 chars
            return True
        else:
            logger.error(f"❌ Custom detection failed with return code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Custom detection timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Error running custom detection: {e}")
        return False
    
    finally:
        # Cleanup
        for file in [before_path, after_path]:
            if Path(file).exists():
                Path(file).unlink()

def main():
    """Run all tests"""
    
    print("🚀 Change Detection Pipeline Test Suite")
    print("=" * 50)
    
    tests = [
        ("Direct API", test_direct_api),
        ("Main Script", test_main_script),
        ("Model Comparison", test_model_comparison),
        ("Custom Detection", test_custom_detection)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n📊 TEST RESULTS SUMMARY:")
    print("=" * 30)
    
    passed = 0
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20}: {status}")
        if result:
            passed += 1
    
    total = len(results)
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Your change detection system is working!")
        return 0
    else:
        print("⚠️ Some tests failed. Check the logs above for details.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 