import sys
import argparse
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.database import AccelerometerDB
from src.data.calibration import AccelerometerCalibrator
from src.data.sync import TimeSynchronizer
from src.utils.helpers import calculate_sample_rate

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calibrate_data(input_dir: str, 
                   output_dir: str,
                   calibration_file: str,
                   behavior_file: str = None,
                   deployment_file: str = None,
                   bird_limit: int = None,
                   samples_per_bird: int = None):
    """
    Calibrate raw accelerometer data and save to clean directory
    
    Args:
        input_dir: Directory containing raw data (with database)
        output_dir: Directory to save calibrated data
        calibration_file: Path to calibration recordings CSV
        behavior_file: Path to behavior observation CSV
        deployment_file: Path to deployment metadata CSV
        bird_limit: Limit number of birds to process (None = all)
        samples_per_bird: Limit samples per bird (None = all)
    """
    logger.info("="*60)
    logger.info("CALIBRATING ACCELEROMETER DATA")
    logger.info("="*60)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load calibration parameters
    logger.info("\nStep 1: Loading calibration parameters...")
    calibrator = AccelerometerCalibrator()
    
    try:
        cal_data = calibrator.load_calibration_data(calibration_file)
        cal_params = calibrator.estimate_calibration_parameters(cal_data)
    except FileNotFoundError:
        logger.warning(
            "Calibration file not found: %s. Using identity calibration.",
            calibration_file,
        )
        cal_params = calibrator.estimate_calibration_parameters(pd.DataFrame())

    logger.info("Calibration parameters:")
    for key, value in cal_params.items():
        logger.info(f"  {key}: {value:.4f}")

    # Save calibration parameters
    param_file = output_path / 'calibration_params.json'
    calibrator.save_calibration_params(str(param_file))
    logger.info(f"✓ Saved calibration parameters to {param_file}")
    
    # Step 2: Load raw data from database
    logger.info("\nStep 2: Loading raw data from database...")
    db_path = Path(input_dir) / 'ruff-acc.db'
    
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return
    
    db = AccelerometerDB(str(db_path))
    bird_ids = db.get_bird_ids()
    synchronizer = TimeSynchronizer()

    if behavior_file is None:
        behavior_file = str(Path(input_dir) / 'ruff_behaviour_tidy_2022-11-16.csv')
    if deployment_file is None:
        deployment_file = str(Path(input_dir) / 'logger_deployment_notes.csv')

    behavior_data = None
    if Path(behavior_file).exists():
        behavior_data = pd.read_csv(behavior_file)
        behavior_data = behavior_data.rename(
            columns={
                'behaviour': 'behavior',
                'start_dt_real': 'start_time',
                'stop_dt_real': 'end_time',
            }
        )
        behavior_data['start_time'] = pd.to_datetime(behavior_data['start_time'])
        behavior_data['end_time'] = pd.to_datetime(behavior_data['end_time'])
        behavior_data['recording_id'] = behavior_data['recording_id'].astype(str)
    else:
        logger.warning("Behavior file not found: %s", behavior_file)

    deployment_lookup = {}
    if Path(deployment_file).exists():
        deployment = pd.read_csv(deployment_file)
        deployment['recording_id'] = deployment['recording_id'].astype(str)
        deployment_lookup = (
            deployment.set_index('recording_id')['ruff_id_number'].astype(str).to_dict()
        )
    else:
        logger.warning("Deployment metadata file not found: %s", deployment_file)
    
    if bird_limit:
        bird_ids = bird_ids[:bird_limit]
        logger.info(f"Processing {len(bird_ids)} birds (limited)")
    else:
        logger.info(f"Processing all {len(bird_ids)} birds")
    
    # Step 3: Process each bird
    logger.info("\nStep 3: Calibrating data for each bird...")
    
    calibrated_files = []
    
    for bird_id in tqdm(bird_ids, desc="Processing recordings"):
        try:
            # Load raw data
            raw_data = db.query_raw_data(bird_id, limit=samples_per_bird)
            
            if len(raw_data) == 0:
                logger.warning(f"No data for {bird_id}, skipping")
                continue
            
            raw_data['timestamp'] = pd.to_datetime(raw_data['timestamp'])
            raw_data['recording_id'] = str(bird_id)
            raw_data['bird_id'] = deployment_lookup.get(str(bird_id), str(bird_id))

            if behavior_data is not None:
                recording_behavior = behavior_data[
                    behavior_data['recording_id'] == str(bird_id)
                ][['start_time', 'end_time', 'behavior']]
                if not recording_behavior.empty:
                    raw_data = synchronizer.align_timestamps(
                        raw_data,
                        recording_behavior,
                        time_offset=None,
                    )
                else:
                    raw_data['behavior'] = 'unknown'
            else:
                raw_data['behavior'] = 'unknown'
            
            # Apply calibration
            calibrated_data = calibrator.apply_calibration(raw_data)
            
            # Calculate additional metrics
            sample_rate = calculate_sample_rate(calibrated_data['timestamp'])
            static_window = int(round(sample_rate)) if sample_rate else 50
            calibrated_data = calibrator.calculate_static_acceleration(
                calibrated_data,
                window_size=static_window,
            )
            calibrated_data = calibrator.calculate_dynamic_acceleration(calibrated_data)
            calibrated_data['vedba'] = calibrator.calculate_vedba(calibrated_data)
            
            # Validate calibration
            validation = calibrator.validate_calibration(calibrated_data)
            
            # Save calibrated data
            output_file = output_path / f'{bird_id}_calibrated.csv'
            calibrated_data.to_csv(output_file, index=False)
            calibrated_files.append(output_file)
            
            logger.debug(f"✓ {bird_id}: {len(calibrated_data)} samples, "
                        f"magnitude = {validation['mean_magnitude']:.3f}g")
        
        except Exception as e:
            logger.error(f"Error processing {bird_id}: {e}")
            continue
    
    # Step 4: Create summary
    logger.info("\n" + "="*60)
    logger.info("CALIBRATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Birds processed: {len(calibrated_files)}")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Files created:")
    for f in calibrated_files[:5]:  # Show first 5
        logger.info(f"  - {f.name}")
    if len(calibrated_files) > 5:
        logger.info(f"  ... and {len(calibrated_files) - 5} more")
    
    # Save list of processed birds
    bird_list_file = output_path / 'processed_birds.txt'
    with open(bird_list_file, 'w') as f:
        for bird_file in calibrated_files:
            f.write(f"{bird_file.stem.replace('_calibrated', '')}\n")
    
    logger.info(f"\n✓ Calibration complete!")
    logger.info(f"  Bird list saved to: {bird_list_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Calibrate raw accelerometer data'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='data/raw',
        help='Input directory containing database (default: data/raw)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/clean',
        help='Output directory for calibrated data (default: data/clean)'
    )
    
    parser.add_argument(
        '--calibration-file',
        type=str,
        default='data/raw/calibration_recordings_6O_Apr2022.csv',
        help='Path to calibration file'
    )
    
    parser.add_argument(
        '--bird-limit',
        type=int,
        default=None,
        help='Limit number of recordings to process (default: all)'
    )
    
    parser.add_argument(
        '--samples-per-bird',
        type=int,
        default=None,
        help='Limit samples per recording (default: all)'
    )

    parser.add_argument(
        '--behavior-file',
        type=str,
        default='data/raw/ruff_behaviour_tidy_2022-11-16.csv',
        help='Path to behavior observation file'
    )

    parser.add_argument(
        '--deployment-file',
        type=str,
        default='data/raw/logger_deployment_notes.csv',
        help='Path to deployment metadata file'
    )
    
    args = parser.parse_args()
    
    calibrate_data(
        input_dir=args.input,
        output_dir=args.output,
        calibration_file=args.calibration_file,
        behavior_file=args.behavior_file,
        deployment_file=args.deployment_file,
        bird_limit=args.bird_limit,
        samples_per_bird=args.samples_per_bird
    )


if __name__ == "__main__":
    main()
