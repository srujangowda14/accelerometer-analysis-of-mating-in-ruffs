import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AccelerometerDataLoader:
    """Load and prepare accelerometer data for analysis"""
    
    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)
        
    def load_calibrated_data(self, bird_id: str) -> pd.DataFrame:
        """
        Load calibrated data for a specific bird
        
        Args:
            bird_id: Bird identifier
            
        Returns:
            DataFrame with calibrated accelerometer data
        """
        file_path = self.data_dir / f'{bird_id}_calibrated.csv'
        
        if not file_path.exists():
            raise FileNotFoundError(f"Calibrated data not found: {file_path}")
        
        data = pd.read_csv(file_path)
        
        # Convert timestamp to datetime if present
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        return data
    
    def load_all_birds(self, bird_list_file: Optional[str] = None) -> pd.DataFrame:
        """
        Load data for all birds
        
        Args:
            bird_list_file: Path to file with bird IDs (one per line)
            
        Returns:
            Combined DataFrame for all birds
        """
        if bird_list_file:
            with open(bird_list_file, 'r') as f:
                bird_ids = [line.strip() for line in f if line.strip()]
        else:
            # Find all calibrated files
            bird_ids = [f.stem.replace('_calibrated', '') 
                       for f in self.data_dir.glob('*_calibrated.csv')]
        
        all_data = []
        
        for bird_id in bird_ids:
            try:
                data = self.load_calibrated_data(bird_id)
                all_data.append(data)
                logger.info(f"Loaded {len(data)} samples for {bird_id}")
            except Exception as e:
                logger.warning(f"Could not load {bird_id}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No data loaded")
        
        combined = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total samples loaded: {len(combined)}")
        
        return combined
    
    def load_video_observations(self, video_file: str) -> pd.DataFrame:
        """
        Load video observation data with behavior labels
        
        Args:
            video_file: Path to video observations CSV
            
        Returns:
            DataFrame with video observations
        """
        data = pd.read_csv(video_file)
        
        # Convert time columns to datetime
        time_cols = ['start_time', 'end_time', 'timestamp']
        for col in time_cols:
            if col in data.columns:
                data[col] = pd.to_datetime(data[col])
        
        return data
    
    def split_by_bird(self, data: pd.DataFrame, 
                      test_birds: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data by bird for train/test
        
        Args:
            data: Full dataset
            test_birds: List of bird IDs for test set
            
        Returns:
            Train and test DataFrames
        """
        train_data = data[~data['bird_id'].isin(test_birds)].copy()
        test_data = data[data['bird_id'].isin(test_birds)].copy()
        
        logger.info(f"Train: {len(train_data)} samples from "
                   f"{train_data['bird_id'].nunique()} birds")
        logger.info(f"Test: {len(test_data)} samples from "
                   f"{test_data['bird_id'].nunique()} birds")
        
        return train_data, test_data
    
    def load_windowed_features(self, feature_file: str) -> pd.DataFrame:
        """
        Load pre-computed windowed features
        
        Args:
            feature_file: Path to feature file
            
        Returns:
            DataFrame with features
        """
        features = pd.read_csv(feature_file)
        
        if 'timestamp' in features.columns:
            features['timestamp'] = pd.to_datetime(features['timestamp'])
        
        return features
    
    def filter_behaviors(self, data: pd.DataFrame, 
                        behaviors: List[str]) -> pd.DataFrame:
        """
        Filter data to include only specific behaviors
        
        Args:
            data: Full dataset
            behaviors: List of behaviors to keep
            
        Returns:
            Filtered DataFrame
        """
        if 'behavior' not in data.columns:
            raise ValueError("Data must have 'behavior' column")
        
        filtered = data[data['behavior'].isin(behaviors)].copy()
        
        logger.info(f"Filtered to {len(filtered)} samples with behaviors: "
                   f"{behaviors}")
        
        return filtered
    
    def get_behavior_distribution(self, data: pd.DataFrame) -> pd.Series:
        """
        Get distribution of behaviors in dataset
        
        Args:
            data: Dataset with behavior labels
            
        Returns:
            Series with behavior counts
        """
        if 'behavior' not in data.columns:
            raise ValueError("Data must have 'behavior' column")
        
        distribution = data['behavior'].value_counts().sort_index()
        
        logger.info("\nBehavior distribution:")
        for behavior, count in distribution.items():
            percentage = 100 * count / len(data)
            logger.info(f"  {behavior}: {count} ({percentage:.1f}%)")
        
        return distribution