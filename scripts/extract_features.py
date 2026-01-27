import sys
import argparse
from pathlib import Path
import logging
import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import AccelerometerDataLoader
from src.features.extraction import FeatureExtractor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_features(input_dir: str,
                    output_dir: str,
                    window_size: float = 1.0,
                    overlap: float = 0.5,
                    sampling_rate: float = 25.0,
                    bird_limit: int = None):
    """
    Extract features from calibrated data
    
    Args:
        input_dir: Directory with calibrated data
        output_dir: Directory to save features
        window_size: Window size in seconds
        overlap: Overlap fraction (0-1)
        sampling_rate: Sampling rate in Hz
        bird_limit: Limit number of birds to process
    """
    logger.info("="*60)
    logger.info("FEATURE EXTRACTION")
    logger.info("="*60)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize
    loader = AccelerometerDataLoader(input_dir)
    extractor = FeatureExtractor(
        window_size=window_size,
        overlap=overlap,
        sampling_rate=sampling_rate
    )
    
    logger.info(f"\nWindow size: {window_size}s")
    logger.info(f"Overlap: {overlap*100}%")
    logger.info(f"Sampling rate: {sampling_rate}Hz")
    
    # Get bird list
    bird_list_file = Path(input_dir) / 'processed_birds.txt'
    
    if bird_list_file.exists():
        with open(bird_list_file, 'r') as f:
            bird_ids = [line.strip() for line in f if line.strip()]
       
    else:
        bird_ids = [f.stem.replace('_calibrated', '') 
                   for f in Path(input_dir).glob('*_calibrated.csv')]
      # Debug: print what we found
    logger.info(f"Bird list file: {bird_list_file}")
    logger.info(f"Bird list exists: {bird_list_file.exists()}")
    logger.info(f"Bird IDs found: {bird_ids}")
    
    if bird_limit:
        bird_ids = bird_ids[:bird_limit]
    
    logger.info(f"\nProcessing {len(bird_ids)} birds")
    
    # Process each bird
    all_features = []
    
    for bird_id in tqdm(bird_ids, desc="Extracting features"):
        try:
            # Load calibrated data
            data = loader.load_calibrated_data(bird_id)
            
            # Filter to required columns
            required_cols = ['acc_x', 'acc_y', 'acc_z']
            optional_cols = ['behavior', 'bird_id', 'timestamp']
            
            cols_to_keep = required_cols + [c for c in optional_cols 
                                           if c in data.columns]
            data = data[cols_to_keep]
            
            # Extract features
            features = extractor.process_dataset(data)
            
            all_features.append(features)
            
            logger.debug(f"✓ {bird_id}: {len(features)} windows, "
                        f"{len(features.columns)} features")
            
        except Exception as e:
            logger.error(f"Error processing {bird_id}: {e}")
            continue
    
    if not all_features:
        logger.error("No features extracted!")
        return
    
    # Combine all features
    logger.info("\nCombining features...")
    combined_features = pd.concat(all_features, ignore_index=True)
    
    # Save features
    output_file = output_path / 'windowed_features.csv'
    combined_features.to_csv(output_file, index=False)
    
    # Save summary
    logger.info("\n" + "="*60)
    logger.info("FEATURE EXTRACTION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total windows: {len(combined_features)}")
    logger.info(f"Total features: {len(combined_features.columns)}")
    logger.info(f"Output file: {output_file}")
    
    if 'behavior' in combined_features.columns:
        logger.info("\nBehavior distribution:")
        dist = combined_features['behavior'].value_counts()
        for behavior, count in dist.items():
            logger.info(f"  {behavior}: {count}")
    
    # Save feature names
    feature_names_file = output_path / 'feature_names.txt'
    feature_cols = [c for c in combined_features.columns 
                   if c not in ['behavior', 'bird_id', 'timestamp', 'window_id']]
    
    with open(feature_names_file, 'w') as f:
        for feat in feature_cols:
            f.write(f"{feat}\n")
    
    logger.info(f"\n✓ Feature extraction complete!")
    logger.info(f"  Feature names saved to: {feature_names_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Extract features from calibrated accelerometer data'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='data/clean',
        help='Input directory with calibrated data (default: data/clean)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/windowed_features',
        help='Output directory for features (default: data/windowed_features)'
    )
    
    parser.add_argument(
        '--window-size',
        type=float,
        default=1.0,
        help='Window size in seconds (default: 1.0)'
    )
    
    parser.add_argument(
        '--overlap',
        type=float,
        default=0.5,
        help='Overlap fraction 0-1 (default: 0.5)'
    )
    
    parser.add_argument(
        '--sampling-rate',
        type=float,
        default=25.0,
        help='Sampling rate in Hz (default: 25.0)'
    )
    
    parser.add_argument(
        '--bird-limit',
        type=int,
        default=None,
        help='Limit number of birds to process (default: all)'
    )
    
    args = parser.parse_args()
    
    extract_features(
        input_dir=args.input,
        output_dir=args.output,
        window_size=args.window_size,
        overlap=args.overlap,
        sampling_rate=args.sampling_rate,
        bird_limit=args.bird_limit
    )


if __name__ == "__main__":
    main()