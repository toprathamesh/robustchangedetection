#!/usr/bin/env python
"""
[START] SOTA Change Detection - Main Demo Script
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# Set debug level for our modules
logging.getLogger('changedetection').setLevel(logging.INFO)

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
    
    logger.info(f"[COMPARE] Running {model_type.upper()} model...")
    logger.info(f"[FILE] Before: {before_path}")
    logger.info(f"[FILE] After: {after_path}")
    
    # Validate image files
    from PIL import Image
    try:
        logger.info("[EMOJI] Loading and validating images...")
        before_img = Image.open(before_path)
        after_img = Image.open(after_path)
        logger.info(f"[SUCCESS] Before image: {before_img.size} ({before_img.mode})")
        logger.info(f"[SUCCESS] After image: {after_img.size} ({after_img.mode})")
        before_img.close()
        after_img.close()
    except Exception as e:
        logger.error(f"[ERROR] Image loading failed: {e}")
        return None
    
    try:
        # Quick prediction with visualization
        logger.info("[MODEL] Starting model prediction...")
        results = quick_predict(
            before_path, 
            after_path, 
            model_type=model_type,
            visualize=True
        )
        
        # Print results
        print(f"\n[RESULTS] RESULTS:")
        print(f"[DATA] Change detected: {results['change_percentage']:.2f}%")
        print(f"[SEARCH] Changed pixels: {results['changed_pixels']:,}")
        print(f"[TOTAL] Total pixels: {results['total_pixels']:,}")
        print(f"[SIZE] Before image shape: {results.get('before_image_shape', 'N/A')}")
        print(f"[SIZE] After image shape: {results.get('after_image_shape', 'N/A')}")
        
        # Show additional info if available
        if 'probability_map' in results:
            print(f"[TARGET] Probability map available")
            
        return results
        
    except Exception as e:
        logger.error(f"[ERROR] Error running {model_type}: {e}")
        logger.error(f"[INFO] Error details: {type(e).__name__}: {str(e)}")
        import traceback
        logger.debug(f"Full traceback:\n{traceback.format_exc()}")
        return None


def run_model_comparison(before_path: str, after_path: str):
    """Compare all available models"""
    
    if not MODELS_AVAILABLE:
        logger.error("Change detection models not available. Please install dependencies.")
        return None
    
    logger.info("[COMPARE] Comparing all available models...")
    
    try:
        # Compare all models
        results = compare_models(before_path, after_path)
        
        print(f"\n[DATA] MODEL COMPARISON RESULTS:")
        print("=" * 50)
        
        for model_name, result in results.items():
            if 'error' in result:
                print(f"[ERROR] {model_name:15}: Error - {result['error']}")
            else:
                print(f"[SUCCESS] {model_name:15}: {result['change_percentage']:6.2f}% change")
        
        # Find best model
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            best_model = max(valid_results, key=lambda x: valid_results[x]['change_percentage'])
            print(f"\n[BEST] Best performing model: {best_model.upper()}")
            print(f"[DATA] Detected: {valid_results[best_model]['change_percentage']:.2f}% change")
        
        return results
        
    except Exception as e:
        logger.error(f"[ERROR] Error comparing models: {e}")
        return None


def run_custom_detection(before_path: str, after_path: str, 
                        threshold: float = 0.5, model_type: str = "siamese_unet"):
    """Run custom change detection with specific parameters"""
    
    if not MODELS_AVAILABLE:
        logger.error("Change detection models not available. Please install dependencies.")
        return None
    
    logger.info(f"[TOOL] Running custom detection with {model_type}...")
    logger.info(f"[CONFIG] Threshold: {threshold}")
    
    try:
        from changedetection.models.model_interface import InferenceConfig
        
        # Create custom configuration
        config = InferenceConfig(
            threshold=threshold,
            apply_morphology=True,
            min_area=100,
            return_probabilities=True,
            normalize_inputs=True,
            resize_to=(512, 512)
        )
        
        logger.info(f"[CONFIG] Config: threshold={config.threshold}, morphology={config.apply_morphology}")
        
        # Create detector
        logger.info("[BUILD] Creating detector...")
        detector = create_detector(model_type)
        
        # Run inference
        logger.info("[MODEL] Running custom inference...")
        results = detector.predict(before_path, after_path, config)
        
        # Visualize
        logger.info("[VISUAL] Creating visualization...")
        detector.visualize_results(
            before_path, after_path, results,
            save_path=f"custom_{model_type}_results.png"
        )
        
        print(f"\n[CONFIG] CUSTOM RESULTS (threshold={threshold}):")
        print(f"[DATA] Change detected: {results['change_percentage']:.2f}%")
        print(f"[SEARCH] Changed pixels: {results['changed_pixels']:,}")
        print(f"[TOTAL] Total pixels: {results['total_pixels']:,}")
        print(f"[SAVE] Results saved: custom_{model_type}_results.png")
        
        return results
        
    except Exception as e:
        logger.error(f"[ERROR] Error in custom detection: {e}")
        logger.error(f"[INFO] Error details: {type(e).__name__}: {str(e)}")
        import traceback
        logger.debug(f"Full traceback:\n{traceback.format_exc()}")
        return None


def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(
        description="=== SOTA Change Detection - Demo Script ===",
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
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger('changedetection').setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Validate input files
    logger.info(f"[INFO] Checking input files...")
    if not Path(args.before).exists():
        logger.error(f"[ERROR] Before image not found: {args.before}")
        logger.error(f"[INFO] Current working directory: {Path.cwd()}")
        sys.exit(1)
    
    if not Path(args.after).exists():
        logger.error(f"[ERROR] After image not found: {args.after}")
        logger.error(f"[INFO] Current working directory: {Path.cwd()}")
        sys.exit(1)
        
    logger.info(f"[SUCCESS] Input files validated")
    logger.info(f"[FILE] Before: {Path(args.before).absolute()}")
    logger.info(f"[FILE] After: {Path(args.after).absolute()}")
    
    print("=== SOTA Change Detection - Demo ===")
    print("=" * 40)
    
    # Run appropriate function based on arguments
    if args.compare:
        results = run_model_comparison(args.before, args.after)
    elif args.custom:
        results = run_custom_detection(args.before, args.after, args.threshold, args.model)
    else:
        results = run_single_model(args.before, args.after, args.model)
    
    if results is None:
        logger.error("[ERROR] Change detection failed")
        sys.exit(1)
    
    print("\n[SUCCESS] Change detection completed successfully!")


if __name__ == "__main__":
    main() 