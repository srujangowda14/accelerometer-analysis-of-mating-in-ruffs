import numpy as np
import pandas as pd
from scipy import optimize
from typing import Tuple
from typing import Dict, List

class AccelerometerCalibrator:
    """Calibrate accelerometer data"""

    def __init__(self):
        self.calibration_params = {}

    def load_calibration_data(self,calibration_file: str) -> pd.DataFrame:
        """Load calibration recrodings"""
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

        for axis in axes:

            #extract positive and negative g recordings

            pos_g = cal_data[cal_data['orientation'] == f'+{axis}'][f'acc_{axis}'].values
            neg_g = cal_data[cal_data['orientation'] == f'-{axis}'][f'acc_{axis}'].values

            offset = (pos_g.mean() + neg_g.mean())
            scale = (pos_g.mean() - neg_g.mean())

            params[f'{axis}_offset'] = offset
            params[f'{axis}_scale'] = scale

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

        for axis in ['x','y','z']:
            col = f'acc_{axis}'
            offset = self.calibration_params[f'{axis}_offset']
            scale = self.calibration_params[f'{axis}_scale']

            calibrated[col] = (data[col] - offset) / scale
            
            return calibrated
        
        def calculate_vectorial_magnitude(self, data: pd.DataFrame) -> np.ndarray:
            """Calculate vectorial dynamic body acceleration (VeDBA)"""
            return np.sqrt(data['acc_x']**2 + data['acc_y']**2 + data['acc_z']**2)


