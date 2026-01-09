import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

class TimeSynchronizer:
    """Synchronize accelerometer and video observation timestamps"""

    def __init__(self, sampling_rate: float = 25.0):
        """
        Args:
            sampling_rate: Accelerometer sampling rate in Hz
        """

        self.sampling_rate = sampling_rate

    def align_timestamps(self,
                         acc_data: pd.DataFrame,
                         video_data: pd.DataFrame,
                         time_offset: Optional[timedelta]):
        """
        Align accelerometer data with video observations
        Args:
            acc_data: Accelerometer data with timestamp column
            video_data: Video observations with start/end times and behavior labels
            time_offset: Manual time offset correction
        
        Returns:
            Merged DataFrame with behavior labels
        """
        if time_offset:
            acc_data['timestamp'] = acc_data['timestamp'] + time_offset

        acc_data['behavior'] = 'unknown'

        for idx, row in video_data.iterrows():
            mask = (acc_data['timestamp'] >= row['start_time']) & \
                   (acc_data['timestamp'] <= row['end_time'])
            acc_data.loc[mask, 'behavior'] = row['behavior']

        return acc_data
    
    def estimate_time_offset(self, 
                            acc_data: pd.DataFrame,
                            video_data: pd.DataFrame,
                            reference_event: str) -> timedelta:
        """
        Estimate time offset between accelerometer and video
        using a reference event visible in both data sources
        
        Args:
            acc_data: Accelerometer data
            video_data: Video observations
            reference_event: Description of reference event
        
        Returns:
            Estimated time offset
        """
        # Implementation depends on reference event type
        # Example: using a sharp movement or feeding event
        pass
    
