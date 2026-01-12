import numpy as np
import pandas as pd
from scipy import signal
from typing import Tuple, List


class AccelerometerTransforms:
    """Transform accelerometer data for feature extraction and preprocessing"""
    
    def __init__(self):
        pass
    
    def normalize(self, data: pd.DataFrame, 
                 columns: List[str] = None) -> pd.DataFrame:
        """
        Normalize accelerometer data to zero mean and unit variance
        
        Args:
            data: Input DataFrame
            columns: Columns to normalize (default: acc_x, acc_y, acc_z)
            
        Returns:
            Normalized DataFrame
        """
        if columns is None:
            columns = ['acc_x', 'acc_y', 'acc_z']
        
        result = data.copy()
        
        for col in columns:
            if col in result.columns:
                mean = result[col].mean()
                std = result[col].std()
                if std > 0:
                    result[col] = (result[col] - mean) / std
        
        return result
    
    def bandpass_filter(self, data: pd.DataFrame,
                       lowcut: float = 0.5,
                       highcut: float = 10.0,
                       sampling_rate: float = 25.0,
                       order: int = 4) -> pd.DataFrame:
        """
        Apply bandpass filter to accelerometer data
        
        Args:
            data: Input DataFrame
            lowcut: Low frequency cutoff (Hz)
            highcut: High frequency cutoff (Hz)
            sampling_rate: Sampling rate (Hz)
            order: Filter order
            
        Returns:
            Filtered DataFrame
        """
        result = data.copy()
        
        # Design Butterworth bandpass filter
        nyquist = sampling_rate / 2.0
        low = lowcut / nyquist
        high = highcut / nyquist
        
        b, a = signal.butter(order, [low, high], btype='band')
        
        # Apply filter to each axis
        for col in ['acc_x', 'acc_y', 'acc_z']:
            if col in result.columns:
                result[col] = signal.filtfilt(b, a, result[col])
        
        return result
    
    def lowpass_filter(self, data: pd.DataFrame,
                      cutoff: float = 5.0,
                      sampling_rate: float = 25.0,
                      order: int = 4) -> pd.DataFrame:
        """
        Apply lowpass filter to accelerometer data
        
        Args:
            data: Input DataFrame
            cutoff: Cutoff frequency (Hz)
            sampling_rate: Sampling rate (Hz)
            order: Filter order
            
        Returns:
            Filtered DataFrame
        """
        result = data.copy()
        
        # Design Butterworth lowpass filter
        nyquist = sampling_rate / 2.0
        normal_cutoff = cutoff / nyquist
        
        b, a = signal.butter(order, normal_cutoff, btype='low')
        
        # Apply filter to each axis
        for col in ['acc_x', 'acc_y', 'acc_z']:
            if col in result.columns:
                result[col] = signal.filtfilt(b, a, result[col])
        
        return result
    
    def highpass_filter(self, data: pd.DataFrame,
                       cutoff: float = 0.5,
                       sampling_rate: float = 25.0,
                       order: int = 4) -> pd.DataFrame:
        """
        Apply highpass filter to accelerometer data
        
        Args:
            data: Input DataFrame
            cutoff: Cutoff frequency (Hz)
            sampling_rate: Sampling rate (Hz)
            order: Filter order
            
        Returns:
            Filtered DataFrame
        """
        result = data.copy()
        
        # Design Butterworth highpass filter
        nyquist = sampling_rate / 2.0
        normal_cutoff = cutoff / nyquist
        
        b, a = signal.butter(order, normal_cutoff, btype='high')
        
        # Apply filter to each axis
        for col in ['acc_x', 'acc_y', 'acc_z']:
            if col in result.columns:
                result[col] = signal.filtfilt(b, a, result[col])
        
        return result
    
    def moving_average(self, data: pd.DataFrame,
                      window_size: int = 5) -> pd.DataFrame:
        """
        Apply moving average smoothing
        
        Args:
            data: Input DataFrame
            window_size: Window size for moving average
            
        Returns:
            Smoothed DataFrame
        """
        result = data.copy()
        
        for col in ['acc_x', 'acc_y', 'acc_z']:
            if col in result.columns:
                result[col] = result[col].rolling(
                    window=window_size, 
                    center=True, 
                    min_periods=1
                ).mean()
        
        return result
    
    def resample(self, data: pd.DataFrame,
                target_rate: float = 25.0,
                current_rate: float = None) -> pd.DataFrame:
        """
        Resample accelerometer data to target sampling rate
        
        Args:
            data: Input DataFrame with timestamp column
            target_rate: Target sampling rate (Hz)
            current_rate: Current sampling rate (Hz), auto-detected if None
            
        Returns:
            Resampled DataFrame
        """
        if 'timestamp' not in data.columns:
            raise ValueError("Data must have timestamp column for resampling")
        
        # Ensure timestamp is datetime
        data = data.copy()
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data = data.set_index('timestamp')
        
        # Resample to target rate
        target_period = f'{1000/target_rate:.0f}ms'
        resampled = data.resample(target_period).mean()
        
        # Interpolate missing values
        resampled = resampled.interpolate(method='linear')
        
        return resampled.reset_index()
    
    def rotate_axes(self, data: pd.DataFrame,
                   rotation_matrix: np.ndarray) -> pd.DataFrame:
        """
        Rotate accelerometer axes using rotation matrix
        
        Args:
            data: Input DataFrame
            rotation_matrix: 3x3 rotation matrix
            
        Returns:
            Rotated DataFrame
        """
        result = data.copy()
        
        # Stack accelerometer values
        acc_matrix = np.column_stack([
            data['acc_x'].values,
            data['acc_y'].values,
            data['acc_z'].values
        ])
        
        # Apply rotation
        rotated = np.dot(acc_matrix, rotation_matrix.T)
        
        # Update DataFrame
        result['acc_x'] = rotated[:, 0]
        result['acc_y'] = rotated[:, 1]
        result['acc_z'] = rotated[:, 2]
        
        return result
    
    def compute_magnitude(self, data: pd.DataFrame) -> np.ndarray:
        """
        Compute magnitude of acceleration vector
        
        Args:
            data: Input DataFrame with acc_x, acc_y, acc_z
            
        Returns:
            Array of magnitudes
        """
        magnitude = np.sqrt(
            data['acc_x']**2 + 
            data['acc_y']**2 + 
            data['acc_z']**2
        )
        return magnitude
    
    def compute_pitch_roll(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute pitch and roll angles from static acceleration
        
        Args:
            data: Input DataFrame with acc_x, acc_y, acc_z
            
        Returns:
            Tuple of (pitch, roll) arrays in degrees
        """
        # Pitch (rotation around Y axis)
        pitch = np.arctan2(
            data['acc_x'],
            np.sqrt(data['acc_y']**2 + data['acc_z']**2)
        ) * 180 / np.pi
        
        # Roll (rotation around X axis)
        roll = np.arctan2(
            data['acc_y'],
            np.sqrt(data['acc_x']**2 + data['acc_z']**2)
        ) * 180 / np.pi
        
        return pitch, roll
    
    def detrend(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove linear trend from accelerometer data
        
        Args:
            data: Input DataFrame
            
        Returns:
            Detrended DataFrame
        """
        result = data.copy()
        
        for col in ['acc_x', 'acc_y', 'acc_z']:
            if col in result.columns:
                result[col] = signal.detrend(result[col])
        
        return result
    
    def difference(self, data: pd.DataFrame, order: int = 1) -> pd.DataFrame:
        """
        Compute differences (derivatives) of acceleration
        
        Args:
            data: Input DataFrame
            order: Order of differencing
            
        Returns:
            Differenced DataFrame
        """
        result = data.copy()
        
        for col in ['acc_x', 'acc_y', 'acc_z']:
            if col in result.columns:
                result[col] = result[col].diff(order)
        
        # Remove NaN values introduced by differencing
        result = result.dropna()
        
        return result
    
    def standardize_orientation(self, data: pd.DataFrame,
                               gravity_axis: str = 'z') -> pd.DataFrame:
        """
        Standardize orientation so gravity is along specified axis
        
        Args:
            data: Input DataFrame
            gravity_axis: Axis along which gravity should point ('x', 'y', or 'z')
            
        Returns:
            Reoriented DataFrame
        """
        result = data.copy()
        
        # Calculate mean acceleration (should be mostly gravity)
        mean_acc = np.array([
            data['acc_x'].mean(),
            data['acc_y'].mean(),
            data['acc_z'].mean()
        ])
        
        # Normalize to get gravity direction
        gravity_dir = mean_acc / np.linalg.norm(mean_acc)
        
        # Create target direction based on gravity_axis
        if gravity_axis == 'x':
            target = np.array([1.0, 0.0, 0.0])
        elif gravity_axis == 'y':
            target = np.array([0.0, 1.0, 0.0])
        else:  # 'z'
            target = np.array([0.0, 0.0, 1.0])
        
        # Calculate rotation matrix (simplified version)
        # This is a basic implementation; more sophisticated methods exist
        rotation_axis = np.cross(gravity_dir, target)
        if np.linalg.norm(rotation_axis) > 0.001:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            angle = np.arccos(np.clip(np.dot(gravity_dir, target), -1.0, 1.0))
            
            # Rodrigues' rotation formula
            K = np.array([
                [0, -rotation_axis[2], rotation_axis[1]],
                [rotation_axis[2], 0, -rotation_axis[0]],
                [-rotation_axis[1], rotation_axis[0], 0]
            ])
            
            rotation_matrix = (np.eye(3) + 
                             np.sin(angle) * K + 
                             (1 - np.cos(angle)) * np.dot(K, K))
            
            result = self.rotate_axes(result, rotation_matrix)
        
        return result