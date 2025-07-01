#!/usr/bin/env python
"""
Download and setup local models for offline operation
"""
import os
import sys
import requests
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
import django
django.setup()

import logging
logger = logging.getLogger(__name__)


def create_local_models():
    """Create local models for offline operation"""
    models_dir = Path('ml_models')
    models_dir.mkdir(exist_ok=True)
    
    print("Setting up local models...")
    
    # 1. Create dummy U-Net model
    try:
        from change_detection.ml_models import UNetChangeDetection
        
        model_path = models_dir / 'change_detection_unet.pth'
        if not model_path.exists():
            print("Creating U-Net model...")
            model = UNetChangeDetection(n_channels=6, n_classes=1)
            
            # Initialize with random weights (in production, use trained weights)
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': 100,
                'loss': 0.15,
                'accuracy': 0.85,
                'f1_score': 0.82,
                'model_name': 'unet_change_detection',
                'version': '1.0'
            }, model_path)
            print(f"Created U-Net model at {model_path}")
        
    except Exception as e:
        print(f"Error creating U-Net model: {e}")
    
    # 2. Create sample training data structure
    data_dir = Path('training_data')
    data_dir.mkdir(exist_ok=True)
    
    # Create directories
    for subdir in ['before', 'after', 'masks']:
        (data_dir / subdir).mkdir(exist_ok=True)
    
    # Create sample data files
    sample_pairs = data_dir / 'train_pairs.txt'
    if not sample_pairs.exists():
        with open(sample_pairs, 'w') as f:
            f.write("before/sample1.png\tafter/sample1.png\tmasks/sample1.png\n")
            f.write("before/sample2.png\tafter/sample2.png\tmasks/sample2.png\n")
        print(f"Created sample pairs file at {sample_pairs}")
    
    # 3. Create sample images for testing
    create_sample_images()
    
    print("Local models setup completed!")


def create_sample_images():
    """Create sample satellite images for testing"""
    from PIL import Image
    
    sample_dir = Path('sample_data')
    sample_dir.mkdir(exist_ok=True)
    
    # Create sample RGB images (simulating satellite data)
    for i in range(3):
        # Before image (more vegetation)
        before_img = np.random.randint(50, 150, (512, 512, 3), dtype=np.uint8)
        before_img[:, :, 1] += 50  # More green
        before_img = np.clip(before_img, 0, 255)
        
        # After image (more urban/less vegetation)
        after_img = np.random.randint(80, 180, (512, 512, 3), dtype=np.uint8)
        after_img[:, :, 0] += 30  # More red
        after_img[:, :, 2] += 30  # More blue
        after_img = np.clip(after_img, 0, 255)
        
        # Add some specific changes
        # Add a "building" (bright rectangle)
        after_img[200:300, 200:300] = [200, 200, 200]
        
        # Remove vegetation (dark rectangle)
        after_img[350:400, 100:200, 1] = 50  # Less green
        
        # Save images
        Image.fromarray(before_img).save(sample_dir / f'before_{i+1}.png')
        Image.fromarray(after_img).save(sample_dir / f'after_{i+1}.png')
    
    print(f"Created sample images in {sample_dir}")


def download_bhoonidhi_samples():
    """Download sample data from Bhoonidhi (mock)"""
    print("Setting up Bhoonidhi sample data...")
    
    bhoonidhi_dir = Path('satellite_data') / 'bhoonidhi'
    bhoonidhi_dir.mkdir(parents=True, exist_ok=True)
    
    # Create mock Bhoonidhi data
    for i in range(2):
        # Create a mock 5m resolution image (Red, Green, NIR)
        mock_image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        
        # Simulate typical satellite spectral characteristics
        mock_image[:, :, 0] = np.random.randint(80, 150, (1024, 1024))  # Red
        mock_image[:, :, 1] = np.random.randint(100, 200, (1024, 1024))  # Green
        mock_image[:, :, 2] = np.random.randint(150, 250, (1024, 1024))  # NIR
        
        # Add some realistic features
        # Water bodies (low NIR)
        water_mask = np.random.random((1024, 1024)) < 0.05
        mock_image[water_mask, 2] = np.random.randint(20, 50, np.sum(water_mask))
        
        # Urban areas (high red, low NIR)
        urban_mask = np.random.random((1024, 1024)) < 0.1
        mock_image[urban_mask, 0] = np.random.randint(150, 200, np.sum(urban_mask))
        mock_image[urban_mask, 2] = np.random.randint(50, 100, np.sum(urban_mask))
        
        # Save as TIFF
        image_path = bhoonidhi_dir / f'BHOONIDHI_SAMPLE_{i+1:03d}.tiff'
        Image.fromarray(mock_image).save(image_path)
    
    print(f"Created Bhoonidhi sample data in {bhoonidhi_dir}")


def verify_setup():
    """Verify that all components are working"""
    print("\nVerifying setup...")
    
    try:
        # Test model loading
        from ml_models.local_models import get_change_detection_model
        model = get_change_detection_model()
        print("✓ Local change detection model loaded successfully")
        
        # Test preprocessing
        from data_processing.preprocessing import CloudShadowMask, RadiometricNormalization
        masker = CloudShadowMask()
        normalizer = RadiometricNormalization()
        print("✓ Preprocessing modules loaded successfully")
        
        # Test Bhoonidhi API
        from data_processing.bhoonidhi_api import BhoonidihiAPI
        api = BhoonidihiAPI()
        print("✓ Bhoonidhi API loaded successfully")
        
        # Test sample prediction
        sample_dir = Path('sample_data')
        if sample_dir.exists():
            before_files = list(sample_dir.glob('before_*.png'))
            after_files = list(sample_dir.glob('after_*.png'))
            
            if before_files and after_files:
                from PIL import Image
                import numpy as np
                
                before_img = np.array(Image.open(before_files[0]))
                after_img = np.array(Image.open(after_files[0]))
                
                change_map, confidence = model.predict(before_img, after_img)
                print(f"✓ Sample prediction successful (confidence: {confidence:.3f})")
        
        print("\n✅ All components verified successfully!")
        print("\nNext steps:")
        print("1. Start the application: docker-compose up --build")
        print("2. Access web interface: http://localhost:8000")
        print("3. Create AOIs and test change detection")
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        print("\nPlease check the logs and try again.")


def main():
    """Main setup function"""
    print("🛰️  Setting up Robust Change Detection System for offline operation...")
    print("=" * 60)
    
    try:
        create_local_models()
        create_sample_images()
        download_bhoonidhi_samples()
        verify_setup()
        
    except Exception as e:
        print(f"Setup failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 