import numpy as np
import pandas as pd
from scipy import signal, stats
from typing import List, Dict, Tuple


class FeatureExtractor:
    """Extract features from windowed accelerometer data"""

    def __init__(self, window_size: float = 1.0, overlap: float = 0.5,
                 sampling_rate: float = 25.0):
        """
        Args:
            window_size: Window size in seconds
            overlap: Overlap fraction (0-1)
            sampling_rate: Sampling rate in Hz
        """
        self.window_size = window_size
        self.overlap = overlap
        self.sampling_rate = sampling_rate
        self.window_samples = int(window_size * sampling_rate)
        self.step_samples = int(self.window_samples * (1 - overlap))

    def create_windows(self, data: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Create sliding windows from continuous data
        
        Args:
            data: Continuous accelerometer data
        
        Returns:
            List of windowed DataFrames
        """

        windows = []
        n_samples = len(data)

        for start_idx in range(0, n_samples - self.window_samples + 1,
                               self.step_samples):
            end_idx = start_idx + self.window_samples
            window = data.iloc[start_idx: end_idx].copy()
            window['window_id'] = len(windows)
            windows.append(window)

        return windows
    
    def extract_time_domain_features(self, window: pd.DataFrame) -> Dict:
        """
        Extract time-domain features from a single window
        
        Features include:
        - Mean, median, std, min, max
        - Quartiles, IQR
        - Skewness, kurtosis
        - Zero crossing rate
        - Energy
        """

        features = {}

        for axis in ['acc_x', 'acc_y', 'acc_z']:
            values = window[axis].values

            features[f'{axis}_mean'] = np.mean(values)
            features[f'{axis}_median'] = np.median(values)
            features[f'{axis}_std'] = np.std(values)
            features[f'{axis}_min'] = np.min(values)
            features[f'{axis}_max'] = np.max(values)
            features[f'{axis}_range'] = np.ptp(values)

            # Quartiles
            q25, q75 = np.percentile(values, [25, 75])
            features[f'{axis}_q25'] = q25
            features[f'{axis}_q75'] = q75
            features[f'{axis}_iqr'] = q75 - q25

            features[f'{axis}_skew'] = stats.skew(values)
            features[f'{axis}_kurtosis'] = stats.kurtosis(values)

            # Zero crossing rate
            features[f'{axis}_zcr'] = np.sum(np.diff(np.sign(values)) != 0)

            # Energy
            features[f'{axis}_energy'] = np.sum(values**2)

            # Root mean square
            features[f'{axis}_rms'] = np.sqrt(np.mean(values**2))

        # Cross-axis features
        features['vedba'] = np.mean(np.sqrt(
            window['acc_x']**2 + window['acc_y']**2 + window['acc_z']**2
        ))

        # Correlation between axes
        features['corr_xy'] = np.corrcoef(window['acc_x'], window['acc_y'])[0, 1]
        features['corr_xz'] = np.corrcoef(window['acc_x'], window['acc_z'])[0, 1]
        features['corr_yz'] = np.corrcoef(window['acc_y'], window['acc_z'])[0, 1]

        return features
    
    def extract_frequency_domain_features(self, window: pd.DataFrame) -> Dict:
        """
        Extract frequency-domain features using FFT
        
        Features include:
        - Dominant frequency
        - Spectral entropy
        - Power in frequency bands
        """

        features = {}

        for axis in ['acc_x', 'acc_y', 'acc_z']:
            values = window[axis].values

        # FFT
        fft_vals = np.fft.rfft(values)
        fft_freq = np.fft.rfftfreq(len(values), 1/self.sampling_rate)
        power = np.abs(fft_vals)**2

        #Dominant frequency
        dominant_idx = np.argmax(power[1:]) + 1  # Skip DC component
        features[f'{axis}_dominant_freq'] = fft_freq[dominant_idx]
        features[f'{axis}_dominant_power'] = power[dominant_idx]

        # Spectral entropy
        power_norm = power / np.sum(power)
        power_norm = power_norm[power_norm > 0]
        features[f'{axis}_spectral_entropy'] = -np.sum(
                power_norm * np.log2(power_norm)
        )

        # Power in frequency bands
        # Low: 0-2 Hz, Medium: 2-5 Hz, High: 5-12.5 Hz
        low_mask = (fft_freq >= 0) & (fft_freq < 2)
        mid_mask = (fft_freq >= 2) & (fft_freq < 5)
        high_mask = (fft_freq >= 5)
        
        features[f'{axis}_power_low'] = np.sum(power[low_mask])
        features[f'{axis}_power_mid'] = np.sum(power[mid_mask])
        features[f'{axis}_power_high'] = np.sum(power[high_mask])

        return features
    
    def extract_all_features(self, window: pd.DataFrame) -> Dict:
        """Extract both time and frequency domain features"""

        features = {}
        features.update(self.extract_time_domain_features(window))
        features.update(self.extract_frequency_domain_features(window))

        # Add window metadata
        if 'behavior' in window.columns:
            features['behavior'] = window['behavior'].mode()[0]
        if 'bird_id' in window.columns:
            features['bird_id'] = window['bird_id'].iloc[0]
        if 'timestamp' in window.columns:
            features['timestamp'] = window['timestamp'].iloc[0]

        return features
    
    def process_dataset(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process entire dataset: create windows and extract features
        
        Args:
            data: Full accelerometer dataset
        
        Returns:
            DataFrame with one row per window and features as columns
        """
        windows = self.create_windows(data)
        features_list = []
        
        for window in windows:
            features = self.extract_all_features(window)
            features_list.append(features)
        
        return pd.DataFrame(features_list)
        
        









        






