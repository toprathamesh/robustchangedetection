#!/usr/bin/env python
"""
🚀 SOTA Change Detection - Main Demo Script
==========================================
Demonstration of state-of-the-art change detection models.

Usage:
    python main.py --before before.png --after after.png --model siamese_unet
    python main.py --compare --before before.png --after after.png
    python main.py --benchmark --dir path/to/image/pairs
"""

import argparse
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our change detection package
try:
    from changedetection import (
        create_detector, 
        quick_predict, 
        compare_models,
        UnifiedChangeDetector
    )
    MODELS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import change detection models: {e}")
    MODELS_AVAILABLE = False


def run_single_model(before_path: str, after_path: str, model_type: str = "siamese_unet"):
    """Run change detection with a single model"""
    
    if not MODELS_AVAILABLE:
        logger.error("Change detection models not available. Please install dependencies.")
        return None
    
    logger.info(f"🔬 Running {model_type.upper()} model...")
    logger.info(f"📁 Before: {before_path}")
    logger.info(f"📁 After: {after_path}")
    
    try:
        # Quick prediction with visualization
        results = quick_predict(
            before_path, 
            after_path, 
            model_type=model_type,
            visualize=True
        )
        
        # Print results
        print(f"\n✨ RESULTS:")
        print(f"📊 Change detected: {results['change_percentage']:.2f}%")
        print(f"🔍 Changed pixels: {results['changed_pixels']:,}")
        print(f"📏 Total pixels: {results['total_pixels']:,}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Error running {model_type}: {e}")
        return None


def run_model_comparison(before_path: str, after_path: str):
    """Compare all available models"""
    
    if not MODELS_AVAILABLE:
        logger.error("Change detection models not available. Please install dependencies.")
        return None
    
    logger.info("🔬 Comparing all available models...")
    
    try:
        # Compare all models
        results = compare_models(before_path, after_path)
        
        print(f"\n📊 MODEL COMPARISON RESULTS:")
        print("=" * 50)
        
        for model_name, result in results.items():
            if 'error' in result:
                print(f"❌ {model_name:15}: Error - {result['error']}")
            else:
                print(f"✅ {model_name:15}: {result['change_percentage']:6.2f}% change")
        
        # Find best model
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            best_model = max(valid_results, key=lambda x: valid_results[x]['change_percentage'])
            print(f"\n🏆 Best performing model: {best_model.upper()}")
            print(f"📊 Detected: {valid_results[best_model]['change_percentage']:.2f}% change")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Error comparing models: {e}")
        return None


def run_custom_detection(before_path: str, after_path: str, 
                        threshold: float = 0.5, model_type: str = "siamese_unet"):
    """Run custom change detection with specific parameters"""
    
    if not MODELS_AVAILABLE:
        logger.error("Change detection models not available. Please install dependencies.")
        return None
    
    logger.info(f"🔧 Running custom detection with {model_type}...")
    
    try:
        from changedetection.models import InferenceConfig
        
        # Create custom configuration
        config = InferenceConfig(
            threshold=threshold,
            apply_morphology=True,
            min_area=100,
            return_probabilities=True,
            normalize_inputs=True,
            resize_to=(512, 512)
        )
        
        # Create detector
        detector = create_detector(model_type)
        
        # Run inference
        results = detector.predict(before_path, after_path, config)
        
        # Visualize
        detector.visualize_results(
            before_path, after_path, results,
            save_path=f"custom_{model_type}_results.png"
        )
        
        print(f"\n⚙️ CUSTOM RESULTS (threshold={threshold}):")
        print(f"📊 Change detected: {results['change_percentage']:.2f}%")
        print(f"💾 Results saved: custom_{model_type}_results.png")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Error in custom detection: {e}")
        return None


def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(
        description="🚀 SOTA Change Detection - Demo Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single model prediction
    python main.py --before before.png --after after.png --model siamese_unet
    
    # Compare all models  
    python main.py --compare --before before.png --after after.png
    
    # Custom detection with specific threshold
    python main.py --before before.png --after after.png --threshold 0.3
        """
    )
    
    parser.add_argument('--before', required=True, help='Path to before image')
    parser.add_argument('--after', required=True, help='Path to after image')
    parser.add_argument('--model', default='siamese_unet', 
                       choices=['siamese_unet', 'tinycd', 'changeformer', 'baseline_unet'],
                       help='Model to use for detection')
    parser.add_argument('--compare', action='store_true', 
                       help='Compare all available models')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Detection threshold (0.0-1.0)')
    parser.add_argument('--custom', action='store_true',
                       help='Use custom configuration')
    
    args = parser.parse_args()
    
    # Validate input files
    if not Path(args.before).exists():
        logger.error(f"❌ Before image not found: {args.before}")
        sys.exit(1)
    
    if not Path(args.after).exists():
        logger.error(f"❌ After image not found: {args.after}")
        sys.exit(1)
    
    print("🚀 SOTA Change Detection - Demo")
    print("=" * 40)
    
    # Run appropriate function based on arguments
    if args.compare:
        results = run_model_comparison(args.before, args.after)
    elif args.custom:
        results = run_custom_detection(args.before, args.after, args.threshold, args.model)
    else:
        results = run_single_model(args.before, args.after, args.model)
    
    if results is None:
        logger.error("❌ Change detection failed")
        sys.exit(1)
    
    print("\n✅ Change detection completed successfully!")


if __name__ == "__main__":
    main() 