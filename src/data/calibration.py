import numpy as np
import pandas as pd
import json
from typing import Dict, List
import logging


class AccelerometerCalibrator:
    """Calibrate accelerometer data"""
    
    def __init__(self):
        self.calibration_params = {}
        self.logger = logging.getLogger(__name__)
    
    def load_calibration_data(self, calibration_file: str) -> pd.DataFrame:
        """Load calibration recordings"""
        return pd.read_csv(calibration_file)
    
    def estimate_calibration_parameters(self, cal_data: pd.DataFrame,
                                       axes: List[str] = ['x', 'y', 'z']) -> Dict:
        """
        Estimate calibration parameters using static positions
        Method: Assumes recordings in 6 static positions (+/- g for each axis)
        Estimates offset and scale factors for each axis
        
        Args:
            cal_data: DataFrame with calibration recordings
            axes: List of axis names
            
        Returns:
            Dictionary with calibration parameters
        """
        params = {}
        
        raw_axes = [f'acc_{axis}' for axis in axes]
        if not set(raw_axes).issubset(cal_data.columns):
            self.logger.warning(
                "Calibration file does not contain raw accelerometer columns. "
                "Using identity calibration parameters."
            )
            params = {
                f'{axis}_offset': 0.0
                for axis in axes
            }
            params.update({
                f'{axis}_scale': 1.0
                for axis in axes
            })
            self.calibration_params = params
            return params
        
        # Try to detect the orientation column
        orientation_col = None
        possible_names = ['orientation', 'position', 'axis', 'placement', 'side']
        
        for name in possible_names:
            if name in cal_data.columns:
                orientation_col = name
                self.logger.info("Using '%s' as orientation column", name)
                break
        
        if orientation_col is None:
            # Try alternative method: look for columns with axis indicators
            self.logger.warning(
                "No orientation column found in calibration file. "
                "Trying fallback calibration method."
            )
            return self._estimate_from_separate_columns(cal_data, axes)
        
        # Standard method with orientation column
        for axis in axes:
            # Try different orientation formats
            pos_patterns = [f'+{axis}', f'+{axis.upper()}', f'pos_{axis}', axis + '_up']
            neg_patterns = [f'-{axis}', f'-{axis.upper()}', f'neg_{axis}', axis + '_down']
            
            pos_g = None
            neg_g = None
            
            for pattern in pos_patterns:
                mask = cal_data[orientation_col].astype(str).str.contains(pattern, case=False, na=False)
                if mask.any():
                    pos_g = cal_data[mask][f'acc_{axis}'].values
                    self.logger.info(
                        "Found positive %s orientation with pattern '%s' (%s samples)",
                        axis,
                        pattern,
                        len(pos_g),
                    )
                    break
            
            for pattern in neg_patterns:
                mask = cal_data[orientation_col].astype(str).str.contains(pattern, case=False, na=False)
                if mask.any():
                    neg_g = cal_data[mask][f'acc_{axis}'].values
                    self.logger.info(
                        "Found negative %s orientation with pattern '%s' (%s samples)",
                        axis,
                        pattern,
                        len(neg_g),
                    )
                    break
            
            if pos_g is None or neg_g is None:
                self.logger.warning(
                    "Could not find calibration data for axis %s. "
                    "Falling back to identity for this axis.",
                    axis,
                )
                # Use default values
                params[f'{axis}_offset'] = 0.0
                params[f'{axis}_scale'] = 1.0
                continue
            
            # Calculate offset and scale
            offset = (pos_g.mean() + neg_g.mean()) / 2.0
            scale = (pos_g.mean() - neg_g.mean()) / 2.0
            
            params[f'{axis}_offset'] = offset
            params[f'{axis}_scale'] = scale
        
        self.calibration_params = params
        return params
    
    def _estimate_from_separate_columns(self, cal_data: pd.DataFrame, 
                                       axes: List[str]) -> Dict:
        """
        Alternative calibration method for data with separate columns per orientation
        """
        params = {}
        
        self.logger.info("Trying to find calibration data in separate columns...")
        
        for axis in axes:
            # Look for columns like: acc_x_up, acc_x_down, or +x, -x
            pos_cols = [col for col in cal_data.columns 
                       if axis in col.lower() and any(x in col.lower() 
                       for x in ['up', 'pos', '+', 'plus'])]
            neg_cols = [col for col in cal_data.columns 
                       if axis in col.lower() and any(x in col.lower() 
                       for x in ['down', 'neg', '-', 'minus'])]
            
            if pos_cols and neg_cols:
                pos_g = cal_data[pos_cols[0]].values
                neg_g = cal_data[neg_cols[0]].values
                
                offset = (pos_g.mean() + neg_g.mean()) / 2.0
                scale = (pos_g.mean() - neg_g.mean()) / 2.0
                
                params[f'{axis}_offset'] = offset
                params[f'{axis}_scale'] = scale
                
                self.logger.info(
                    "Found %s calibration columns: pos=%s neg=%s",
                    axis,
                    pos_cols[0],
                    neg_cols[0],
                )
            else:
                params[f'{axis}_offset'] = 0.0
                params[f'{axis}_scale'] = 1.0
                self.logger.warning(
                    "Falling back to identity calibration for axis %s",
                    axis,
                )
        
        self.calibration_params = params
        return params
    
    def apply_calibration(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply calibration to raw accelerometer data
        
        Args:
            data: DataFrame with columns acc_x, acc_y, acc_z
            
        Returns:
            Calibrated DataFrame
        """
        calibrated = data.copy()
        
        for axis in ['x', 'y', 'z']:
            col = f'acc_{axis}'
            if col not in calibrated.columns:
                self.logger.warning("Column %s not found in data", col)
                continue
                
            offset = self.calibration_params.get(f'{axis}_offset', 0.0)
            scale = self.calibration_params.get(f'{axis}_scale', 1.0)
            
            if scale != 0:
                calibrated[col] = (data[col] - offset) / scale
            else:
                calibrated[col] = data[col] - offset
        
        return calibrated
    
    def calculate_static_acceleration(self, data: pd.DataFrame, 
                                     window_size: int = 25) -> pd.DataFrame:
        """
        Calculate static acceleration using moving average
        
        Args:
            data: Calibrated accelerometer data
            window_size: Window size for smoothing
            
        Returns:
            DataFrame with static acceleration columns
        """
        result = data.copy()
        
        for axis in ['x', 'y', 'z']:
            col = f'acc_{axis}'
            if col in result.columns:
                result[f'static_{axis}'] = data[col].rolling(
                    window=window_size, center=True, min_periods=1
                ).mean()
        
        return result
    
    def calculate_dynamic_acceleration(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate dynamic acceleration (total - static)
        
        Args:
            data: DataFrame with acc and static columns
            
        Returns:
            DataFrame with dynamic acceleration columns
        """
        result = data.copy()
        
        for axis in ['x', 'y', 'z']:
            if f'static_{axis}' not in data.columns:
                # Calculate static if not present
                result = self.calculate_static_acceleration(result)
            
            if f'acc_{axis}' in data.columns and f'static_{axis}' in result.columns:
                result[f'dynamic_{axis}'] = (
                    data[f'acc_{axis}'] - result[f'static_{axis}']
                )
        
        return result
    
    def calculate_vedba(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calculate vectorial dynamic body acceleration (VeDBA)
        
        Args:
            data: DataFrame with dynamic acceleration columns
            
        Returns:
            Array of VeDBA values
        """
        if 'dynamic_x' not in data.columns:
            data = self.calculate_dynamic_acceleration(data)
        
        vedba = np.sqrt(
            data['dynamic_x']**2 + 
            data['dynamic_y']**2 + 
            data['dynamic_z']**2
        )
        
        return vedba
    
    def validate_calibration(self, data: pd.DataFrame) -> Dict:
        """
        Validate calibration by checking if magnitude ≈ 1g
        
        Args:
            data: Calibrated accelerometer data
            
        Returns:
            Dictionary with validation metrics
        """
        magnitude = np.sqrt(
            data['acc_x']**2 + 
            data['acc_y']**2 + 
            data['acc_z']**2
        )
        
        validation = {
            'mean_magnitude': magnitude.mean(),
            'std_magnitude': magnitude.std(),
            'magnitude_error': abs(magnitude.mean() - 1.0)
        }
        
        return validation
    
    def save_calibration_params(self, filepath: str):
        """Save calibration parameters to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.calibration_params, f, indent=2)
    
    def load_calibration_params(self, filepath: str):
        """Load calibration parameters from JSON file"""
        with open(filepath, 'r') as f:
            self.calibration_params = json.load(f)
