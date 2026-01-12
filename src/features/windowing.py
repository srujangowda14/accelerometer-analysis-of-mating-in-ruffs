import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


class WindowGenerator:
    """Generate sliding windows from continuous accelerometer data"""
    
    def __init__(self, window_size: float = 1.0, 
                 overlap: float = 0.5,
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
        
    def create_sliding_windows(self, data: pd.DataFrame) -> List[pd.DataFrame]:
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
            window = data.iloc[start_idx:end_idx].copy()
            window['window_id'] = len(windows)
            window['window_start_idx'] = start_idx
            window['window_end_idx'] = end_idx
            windows.append(window)
        
        return windows
    
    def create_non_overlapping_windows(self, data: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Create non-overlapping windows
        
        Args:
            data: Continuous accelerometer data
            
        Returns:
            List of windowed DataFrames
        """
        windows = []
        n_samples = len(data)
        
        for start_idx in range(0, n_samples, self.window_samples):
            end_idx = min(start_idx + self.window_samples, n_samples)
            
            # Only include full windows
            if end_idx - start_idx == self.window_samples:
                window = data.iloc[start_idx:end_idx].copy()
                window['window_id'] = len(windows)
                windows.append(window)
        
        return windows
    
    def create_event_based_windows(self, data: pd.DataFrame,
                                   event_column: str,
                                   event_values: List = None,
                                   before_samples: int = 0,
                                   after_samples: int = 0) -> List[pd.DataFrame]:
        """
        Create windows centered around specific events
        
        Args:
            data: Continuous data with event markers
            event_column: Column containing event markers
            event_values: List of event values to window around
            before_samples: Samples before event
            after_samples: Samples after event
            
        Returns:
            List of event-centered windows
        """
        windows = []
        
        # Find event indices
        if event_values is None:
            event_mask = data[event_column].notna()
        else:
            event_mask = data[event_column].isin(event_values)
        
        event_indices = data[event_mask].index.tolist()
        
        for event_idx in event_indices:
            start_idx = max(0, event_idx - before_samples)
            end_idx = min(len(data), event_idx + after_samples + 1)
            
            window = data.iloc[start_idx:end_idx].copy()
            window['window_id'] = len(windows)
            window['event_idx'] = event_idx
            windows.append(window)
        
        return windows
    
    def create_behavior_windows(self, data: pd.DataFrame,
                               behavior_column: str = 'behavior',
                               min_duration: float = None) -> List[pd.DataFrame]:
        """
        Create windows for each continuous behavior segment
        
        Args:
            data: Data with behavior labels
            behavior_column: Column containing behavior labels
            min_duration: Minimum duration in seconds (None = no filter)
            
        Returns:
            List of behavior segment windows
        """
        windows = []
        
        # Find behavior changes
        behavior_changes = data[behavior_column].ne(data[behavior_column].shift())
        segment_starts = data[behavior_changes].index.tolist()
        
        for i, start_idx in enumerate(segment_starts):
            # Determine end of segment
            if i < len(segment_starts) - 1:
                end_idx = segment_starts[i + 1]
            else:
                end_idx = len(data)
            
            # Check minimum duration
            if min_duration is not None:
                duration_samples = int(min_duration * self.sampling_rate)
                if end_idx - start_idx < duration_samples:
                    continue
            
            window = data.iloc[start_idx:end_idx].copy()
            window['window_id'] = len(windows)
            window['segment_start'] = start_idx
            window['segment_end'] = end_idx
            windows.append(window)
        
        return windows
    
    def create_fixed_count_windows(self, data: pd.DataFrame,
                                   n_windows: int) -> List[pd.DataFrame]:
        """
        Create a fixed number of evenly spaced windows
        
        Args:
            data: Continuous data
            n_windows: Number of windows to create
            
        Returns:
            List of windowed DataFrames
        """
        windows = []
        n_samples = len(data)
        
        # Calculate step size for n_windows
        available_starts = n_samples - self.window_samples + 1
        step = max(1, available_starts // n_windows)
        
        for i in range(n_windows):
            start_idx = i * step
            end_idx = start_idx + self.window_samples
            
            if end_idx > n_samples:
                break
            
            window = data.iloc[start_idx:end_idx].copy()
            window['window_id'] = len(windows)
            windows.append(window)
        
        return windows
    
    def get_window_labels(self, window: pd.DataFrame,
                         label_column: str = 'behavior',
                         method: str = 'majority') -> str:
        """
        Get label for a window
        
        Args:
            window: Window DataFrame
            label_column: Column containing labels
            method: 'majority', 'start', 'end', or 'center'
            
        Returns:
            Window label
        """
        if label_column not in window.columns:
            return None
        
        if method == 'majority':
            # Most common label in window
            return window[label_column].mode()[0]
        elif method == 'start':
            return window[label_column].iloc[0]
        elif method == 'end':
            return window[label_column].iloc[-1]
        elif method == 'center':
            center_idx = len(window) // 2
            return window[label_column].iloc[center_idx]
        else:
            raise ValueError(f"Unknown labeling method: {method}")
    
    def filter_windows_by_label_purity(self, windows: List[pd.DataFrame],
                                      label_column: str = 'behavior',
                                      min_purity: float = 0.8) -> List[pd.DataFrame]:
        """
        Filter windows to only include those with high label purity
        
        Args:
            windows: List of windows
            label_column: Column containing labels
            min_purity: Minimum fraction of samples with majority label
            
        Returns:
            Filtered list of windows
        """
        filtered_windows = []
        
        for window in windows:
            if label_column in window.columns:
                # Calculate purity (fraction with most common label)
                value_counts = window[label_column].value_counts()
                if len(value_counts) > 0:
                    purity = value_counts.iloc[0] / len(window)
                    if purity >= min_purity:
                        filtered_windows.append(window)
            else:
                filtered_windows.append(window)
        
        return filtered_windows
    
    def get_window_statistics(self, windows: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Get statistics about windows
        
        Args:
            windows: List of windows
            
        Returns:
            DataFrame with window statistics
        """
        stats = []
        
        for i, window in enumerate(windows):
            stat = {
                'window_id': i,
                'n_samples': len(window),
                'duration_sec': len(window) / self.sampling_rate
            }
            
            if 'timestamp' in window.columns:
                stat['start_time'] = window['timestamp'].iloc[0]
                stat['end_time'] = window['timestamp'].iloc[-1]
            
            if 'behavior' in window.columns:
                stat['behavior'] = self.get_window_labels(window)
                stat['n_unique_behaviors'] = window['behavior'].nunique()
            
            stats.append(stat)
        
        return pd.DataFrame(stats)
    
    def split_windows_by_bird(self, windows: List[pd.DataFrame],
                             bird_column: str = 'bird_id') -> dict:
        """
        Split windows by bird ID
        
        Args:
            windows: List of windows
            bird_column: Column containing bird IDs
            
        Returns:
            Dictionary mapping bird_id to list of windows
        """
        bird_windows = {}
        
        for window in windows:
            if bird_column in window.columns:
                bird_id = window[bird_column].iloc[0]
                if bird_id not in bird_windows:
                    bird_windows[bird_id] = []
                bird_windows[bird_id].append(window)
        
        return bird_windows
    
    def balance_windows_by_behavior(self, windows: List[pd.DataFrame],
                                   behavior_column: str = 'behavior',
                                   method: str = 'undersample') -> List[pd.DataFrame]:
        """
        Balance windows across behaviors
        
        Args:
            windows: List of windows
            behavior_column: Column containing behavior labels
            method: 'undersample' or 'oversample'
            
        Returns:
            Balanced list of windows
        """
        # Group windows by behavior
        behavior_groups = {}
        for window in windows:
            behavior = self.get_window_labels(window, behavior_column)
            if behavior not in behavior_groups:
                behavior_groups[behavior] = []
            behavior_groups[behavior].append(window)
        
        # Balance
        if method == 'undersample':
            # Sample down to smallest class
            min_count = min(len(wins) for wins in behavior_groups.values())
            balanced = []
            for behavior, wins in behavior_groups.items():
                sampled = np.random.choice(wins, min_count, replace=False).tolist()
                balanced.extend(sampled)
        
        elif method == 'oversample':
            # Sample up to largest class
            max_count = max(len(wins) for wins in behavior_groups.values())
            balanced = []
            for behavior, wins in behavior_groups.items():
                sampled = np.random.choice(wins, max_count, replace=True).tolist()
                balanced.extend(sampled)
        
        else:
            raise ValueError(f"Unknown balancing method: {method}")
        
        return balanced