#!/usr/bin/env python
"""
Quick Demo Script for Change Detection
======================================
Creates synthetic images and tests change detection functionality.
"""

import numpy as np
import cv2
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def create_synthetic_images():
    """Create before and after images with clear changes"""
    
    # Create base image (256x256 RGB)
    before_image = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # Add some background pattern
    before_image[:, :] = [50, 100, 150]  # Blue-ish background
    
    # Add some shapes
    cv2.rectangle(before_image, (50, 50), (100, 100), (255, 255, 255), -1)  # White square
    cv2.circle(before_image, (180, 180), 30, (255, 0, 0), -1)  # Red circle
    
    # Create after image (copy + changes)
    after_image = before_image.copy()
    
    # Add changes
    cv2.rectangle(after_image, (120, 120), (200, 200), (0, 255, 0), -1)  # Green square (new)
    cv2.circle(after_image, (180, 180), 30, (0, 0, 255), -1)  # Change red circle to blue
    
    # Save images
    cv2.imwrite('before_test.png', before_image)
    cv2.imwrite('after_test.png', after_image)
    
    print("✅ Created synthetic test images:")
    print("   📁 before_test.png")
    print("   📁 after_test.png")
    
    return 'before_test.png', 'after_test.png'

def test_change_detection():
    """Test change detection with synthetic images"""
    
    # Create test images
    before_path, after_path = create_synthetic_images()
    
    # Test the change detection
    try:
        from changedetection import quick_predict
        
        print("\n🔬 Testing Change Detection...")
        
        # Run detection with default model
        results = quick_predict(
            before_path, 
            after_path, 
            model_type='siamese_unet',
            visualize=True
        )
        
        print(f"\n✨ RESULTS:")
        print(f"📊 Change detected: {results['change_percentage']:.2f}%")
        print(f"🔍 Changed pixels: {results['changed_pixels']:,}")
        print(f"📏 Total pixels: {results['total_pixels']:,}")
        
        if results['change_percentage'] > 0:
            print("🎉 Change detection is working!")
        else:
            print("⚠️ No changes detected - may need adjustment")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing change detection: {e}")
        return False

def test_model_comparison():
    """Test comparing different models"""
    
    try:
        from changedetection import compare_models
        
        print("\n🔬 Comparing All Models...")
        
        results = compare_models('before_test.png', 'after_test.png')
        
        print(f"\n📊 MODEL COMPARISON:")
        print("=" * 40)
        
        for model_name, result in results.items():
            if 'error' in result:
                print(f"❌ {model_name:15}: Error - {result['error']}")
            else:
                print(f"✅ {model_name:15}: {result['change_percentage']:6.2f}% change")
        
        return True
        
    except Exception as e:
        print(f"❌ Error comparing models: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Change Detection Demo Test")
    print("=" * 40)
    
    # Run tests
    detection_works = test_change_detection()
    comparison_works = test_model_comparison()
    
    print(f"\n📋 Summary:")
    print(f"✅ Change Detection: {'Working' if detection_works else 'Failed'}")
    print(f"✅ Model Comparison: {'Working' if comparison_works else 'Failed'}")
    
    # Cleanup
    # print(f"\n🧹 Cleaning up test files...")
    # for file in ['before_test.png', 'after_test.png']:
    #     if Path(file).exists():
    #         Path(file).unlink()
    #         print(f"   🗑️ Deleted {file}") 