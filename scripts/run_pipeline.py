import sys
import argparse
from pathlib import Path
import logging
import subprocess
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_python_command():
    """Detect the correct Python command for this system"""
    # Try to find python3 first (common on Mac/Linux)
    if shutil.which('python3'):
        return 'python3'
    # Fall back to python
    elif shutil.which('python'):
        return 'python'
    # Use the current Python interpreter as last resort
    else:
        return sys.executable


def run_pipeline(raw_dir: str = 'data/raw',
                clean_dir: str = 'data/clean',
                features_dir: str = 'data/windowed_features',
                models_dir: str = 'outputs/models',
                results_dir: str = 'outputs',
                calibration_file: str = None,
                bird_limit: int = None,
                skip_calibration: bool = False,
                skip_features: bool = False,
                skip_training: bool = False,
                skip_evaluation: bool = False):
    """
    Run complete pipeline: calibration -> feature extraction -> training -> evaluation
    
    Args:
        raw_dir: Directory with raw data
        clean_dir: Directory for calibrated data
        features_dir: Directory for extracted features
        models_dir: Directory for trained models
        results_dir: Directory for evaluation results
        calibration_file: Path to calibration file
        bird_limit: Limit number of birds to process
        skip_calibration: Skip calibration step
        skip_features: Skip feature extraction step
        skip_training: Skip model training step
        skip_evaluation: Skip model evaluation step
    """
    logger.info("="*70)
    logger.info("ACCELEROMETER BEHAVIOR CLASSIFICATION PIPELINE")
    logger.info("="*70)
    
    scripts_dir = Path(__file__).parent
    python_cmd = get_python_command()
    
    logger.info(f"\nUsing Python command: {python_cmd}")
    
    # Step 1: Calibration
    if not skip_calibration:
        logger.info("\n" + "="*70)
        logger.info("STEP 1: DATA CALIBRATION")
        logger.info("="*70)
        
        if calibration_file is None:
            calibration_file = f'{raw_dir}/calibration_recordings_6O_Apr2022.csv'
        
        cmd = [
            python_cmd, str(scripts_dir / 'calibrate_data.py'),
            '--input', raw_dir,
            '--output', clean_dir,
            '--calibration-file', calibration_file
        ]
        
        if bird_limit:
            cmd.extend(['--bird-limit', str(bird_limit)])
        
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            logger.error("Calibration failed!")
            return
        
        logger.info("✓ Calibration complete")
    else:
        logger.info("\nSkipping calibration (using existing calibrated data)")
    
    # Step 2: Feature Extraction
    if not skip_features:
        logger.info("\n" + "="*70)
        logger.info("STEP 2: FEATURE EXTRACTION")
        logger.info("="*70)
        
        cmd = [
            python_cmd, str(scripts_dir / 'extract_features.py'),
            '--input', clean_dir,
            '--output', features_dir,
            '--window-size', '1.0',
            '--overlap', '0.5'
        ]
        
        if bird_limit:
            cmd.extend(['--bird-limit', str(bird_limit)])
        
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            logger.error("Feature extraction failed!")
            return
        
        logger.info("✓ Feature extraction complete")
    else:
        logger.info("\nSkipping feature extraction (using existing features)")
    
    # Step 3: Model Training
    if not skip_training:
        logger.info("\n" + "="*70)
        logger.info("STEP 3: MODEL TRAINING")
        logger.info("="*70)
        
        features_file = f'{features_dir}/windowed_features.csv'
        
        cmd = [
            python_cmd, str(scripts_dir / 'train_models.py'),
            '--features', features_file,
            '--output', models_dir,
            '--models', 'rf', 'hmm'  # Skip NN for faster training
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            logger.error("Model training failed!")
            return
        
        logger.info("✓ Model training complete")
    else:
        logger.info("\nSkipping model training (using existing models)")
    
    # Step 4: Model Evaluation
    if not skip_evaluation:
        logger.info("\n" + "="*70)
        logger.info("STEP 4: MODEL EVALUATION")
        logger.info("="*70)
        
        features_file = f'{features_dir}/windowed_features.csv'
        
        cmd = [
            python_cmd, str(scripts_dir / 'evaluate_models.py'),
            '--features', features_file,
            '--models', models_dir,
            '--output', results_dir
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            logger.error("Model evaluation failed!")
            return
        
        logger.info("✓ Model evaluation complete")
    else:
        logger.info("\nSkipping model evaluation")
    
    # Pipeline complete
    logger.info("\n" + "="*70)
    logger.info("PIPELINE COMPLETE!")
    logger.info("="*70)
    logger.info(f"\nResults can be found in:")
    logger.info(f"  - Calibrated data: {clean_dir}")
    logger.info(f"  - Features: {features_dir}")
    logger.info(f"  - Models: {models_dir}")
    logger.info(f"  - Evaluation results: {results_dir}/results")
    logger.info(f"  - Figures: {results_dir}/figures")
    logger.info("\n✓ All done!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Run complete accelerometer behavior classification pipeline'
    )
    
    parser.add_argument(
        '--raw-dir',
        type=str,
        default='data/raw',
        help='Directory with raw data (default: data/raw)'
    )
    
    parser.add_argument(
        '--clean-dir',
        type=str,
        default='data/clean',
        help='Directory for calibrated data (default: data/clean)'
    )
    
    parser.add_argument(
        '--features-dir',
        type=str,
        default='data/windowed_features',
        help='Directory for features (default: data/windowed_features)'
    )
    
    parser.add_argument(
        '--models-dir',
        type=str,
        default='outputs/models',
        help='Directory for models (default: outputs/models)'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='outputs',
        help='Directory for results (default: outputs)'
    )
    
    parser.add_argument(
        '--calibration-file',
        type=str,
        default=None,
        help='Path to calibration file'
    )
    
    parser.add_argument(
        '--bird-limit',
        type=int,
        default=None,
        help='Limit number of birds to process'
    )
    
    parser.add_argument(
        '--skip-calibration',
        action='store_true',
        help='Skip calibration step'
    )
    
    parser.add_argument(
        '--skip-features',
        action='store_true',
        help='Skip feature extraction step'
    )
    
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip model training step'
    )
    
    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='Skip model evaluation step'
    )
    
    args = parser.parse_args()
    
    run_pipeline(
        raw_dir=args.raw_dir,
        clean_dir=args.clean_dir,
        features_dir=args.features_dir,
        models_dir=args.models_dir,
        results_dir=args.results_dir,
        calibration_file=args.calibration_file,
        bird_limit=args.bird_limit,
        skip_calibration=args.skip_calibration,
        skip_features=args.skip_features,
        skip_training=args.skip_training,
        skip_evaluation=args.skip_evaluation
    )


if __name__ == "__main__":
    main()