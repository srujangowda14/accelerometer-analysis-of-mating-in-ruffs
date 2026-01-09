import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_sample_rate(timestamps: pd.Series) -> float:
    """
    Calculate sampling rate from timestamps
    
    Args:
        timestamps: Series of timestamps
        
    Returns:
        Sampling rate in Hz
    """
    if len(timestamps) < 2:
        return None
    
    # Calculate time differences
    time_diffs = timestamps.diff().dropna()
    
    # Get median time difference in seconds
    median_diff = time_diffs.median().total_seconds()
    
    if median_diff == 0:
        return None
    
    # Sampling rate is inverse of time difference
    sample_rate = 1.0 / median_diff
    
    return sample_rate


def detect_gaps(timestamps: pd.Series, max_gap_seconds: float = 1.0) -> pd.DataFrame:
    """
    Detect gaps in time series data
    
    Args:
        timestamps: Series of timestamps
        max_gap_seconds: Maximum allowed gap in seconds
        
    Returns:
        DataFrame with gap information
    """
    time_diffs = timestamps.diff()
    gaps = time_diffs[time_diffs > pd.Timedelta(seconds=max_gap_seconds)]
    
    gap_info = pd.DataFrame({
        'gap_start': timestamps[gaps.index - 1].values,
        'gap_end': timestamps[gaps.index].values,
        'gap_duration': gaps.values
    })
    
    return gap_info


def remove_outliers(data: pd.DataFrame, 
                   columns: List[str], 
                   n_std: float = 3.0) -> pd.DataFrame:
    """
    Remove outliers using standard deviation method
    
    Args:
        data: Input DataFrame
        columns: Columns to check for outliers
        n_std: Number of standard deviations for outlier threshold
        
    Returns:
        DataFrame with outliers removed
    """
    filtered_data = data.copy()
    
    for col in columns:
        mean = filtered_data[col].mean()
        std = filtered_data[col].std()
        
        lower_bound = mean - n_std * std
        upper_bound = mean + n_std * std
        
        mask = (filtered_data[col] >= lower_bound) & (filtered_data[col] <= upper_bound)
        filtered_data = filtered_data[mask]
    
    removed = len(data) - len(filtered_data)
    if removed > 0:
        logger.info(f"Removed {removed} outliers ({100*removed/len(data):.2f}%)")
    
    return filtered_data


def smooth_signal(signal: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Apply moving average smoothing to signal
    
    Args:
        signal: Input signal
        window_size: Size of smoothing window
        
    Returns:
        Smoothed signal
    """
    if window_size < 2:
        return signal
    
    # Pad signal at edges
    pad_size = window_size // 2
    padded = np.pad(signal, pad_size, mode='edge')
    
    # Apply moving average
    smoothed = np.convolve(padded, np.ones(window_size)/window_size, mode='valid')
    
    return smoothed


def calculate_activity_index(acc_data: pd.DataFrame, 
                            window_seconds: float = 60.0,
                            sampling_rate: float = 25.0) -> pd.DataFrame:
    """
    Calculate activity index over time windows
    
    Args:
        acc_data: DataFrame with acc_x, acc_y, acc_z columns
        window_seconds: Window size in seconds
        sampling_rate: Sampling rate in Hz
        
    Returns:
        DataFrame with activity index per window
    """
    window_samples = int(window_seconds * sampling_rate)
    
    # Calculate VEDBA
    vedba = np.sqrt(
        acc_data['acc_x']**2 + 
        acc_data['acc_y']**2 + 
        acc_data['acc_z']**2
    )
    
    # Calculate activity in windows
    n_windows = len(vedba) // window_samples
    activity_indices = []
    
    for i in range(n_windows):
        start_idx = i * window_samples
        end_idx = start_idx + window_samples
        window_vedba = vedba[start_idx:end_idx]
        
        activity_indices.append({
            'window': i,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'mean_vedba': window_vedba.mean(),
            'std_vedba': window_vedba.std(),
            'max_vedba': window_vedba.max()
        })
    
    return pd.DataFrame(activity_indices)


def balance_classes(X: pd.DataFrame, y: pd.Series, 
                   method: str = 'undersample') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Balance class distribution in dataset
    
    Args:
        X: Feature matrix
        y: Labels
        method: 'undersample' or 'oversample'
        
    Returns:
        Balanced X and y
    """
    from collections import Counter
    
    class_counts = Counter(y)
    logger.info(f"Original class distribution: {class_counts}")
    
    if method == 'undersample':
        # Undersample to smallest class
        min_count = min(class_counts.values())
        
        balanced_indices = []
        for class_label in class_counts.keys():
            class_indices = y[y == class_label].index
            sampled_indices = np.random.choice(class_indices, min_count, replace=False)
            balanced_indices.extend(sampled_indices)
        
        X_balanced = X.loc[balanced_indices].reset_index(drop=True)
        y_balanced = y.loc[balanced_indices].reset_index(drop=True)
        
    elif method == 'oversample':
        # Oversample to largest class
        max_count = max(class_counts.values())
        
        balanced_indices = []
        for class_label in class_counts.keys():
            class_indices = y[y == class_label].index
            sampled_indices = np.random.choice(class_indices, max_count, replace=True)
            balanced_indices.extend(sampled_indices)
        
        X_balanced = X.loc[balanced_indices].reset_index(drop=True)
        y_balanced = y.loc[balanced_indices].reset_index(drop=True)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    new_counts = Counter(y_balanced)
    logger.info(f"Balanced class distribution: {new_counts}")
    
    return X_balanced, y_balanced


def create_cross_validation_splits(data: pd.DataFrame, 
                                   n_splits: int = 5,
                                   group_by: str = 'bird_id') -> List[Tuple]:
    """
    Create cross-validation splits grouped by bird/individual
    
    Args:
        data: Full dataset
        n_splits: Number of CV splits
        group_by: Column to group by (e.g., 'bird_id')
        
    Returns:
        List of (train_indices, test_indices) tuples
    """
    groups = data[group_by].unique()
    np.random.shuffle(groups)
    
    group_splits = np.array_split(groups, n_splits)
    
    cv_splits = []
    for test_groups in group_splits:
        test_mask = data[group_by].isin(test_groups)
        test_indices = data[test_mask].index.tolist()
        train_indices = data[~test_mask].index.tolist()
        cv_splits.append((train_indices, test_indices))
    
    return cv_splits


def calculate_overlap_duration(start1: pd.Timestamp, end1: pd.Timestamp,
                               start2: pd.Timestamp, end2: pd.Timestamp) -> float:
    """
    Calculate overlap duration between two time intervals
    
    Args:
        start1, end1: First interval
        start2, end2: Second interval
        
    Returns:
        Overlap duration in seconds
    """
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    
    if overlap_start >= overlap_end:
        return 0.0
    
    overlap = (overlap_end - overlap_start).total_seconds()
    return overlap


def aggregate_predictions(predictions: np.ndarray, 
                         timestamps: pd.Series,
                         window_seconds: float = 10.0) -> pd.DataFrame:
    """
    Aggregate predictions over time windows using majority voting
    
    Args:
        predictions: Array of predictions
        timestamps: Corresponding timestamps
        window_seconds: Aggregation window in seconds
        
    Returns:
        DataFrame with aggregated predictions
    """
    # Create time bins
    min_time = timestamps.min()
    max_time = timestamps.max()
    
    time_bins = pd.date_range(
        start=min_time,
        end=max_time,
        freq=f'{window_seconds}S'
    )
    
    # Bin predictions
    binned = pd.cut(timestamps, bins=time_bins, include_lowest=True)
    
    # Aggregate by majority vote
    aggregated = []
    for bin_label in binned.cat.categories:
        mask = binned == bin_label
        if mask.sum() == 0:
            continue
        
        bin_predictions = predictions[mask]
        majority_prediction = pd.Series(bin_predictions).mode()[0]
        
        aggregated.append({
            'time_window': bin_label,
            'prediction': majority_prediction,
            'n_samples': mask.sum()
        })
    
    return pd.DataFrame(aggregated)


def print_data_summary(data: pd.DataFrame):
    """
    Print comprehensive data summary
    
    Args:
        data: DataFrame to summarize
    """
    logger.info("\nData Summary:")
    logger.info("="*50)
    logger.info(f"Total samples: {len(data)}")
    logger.info(f"Columns: {len(data.columns)}")
    logger.info(f"Memory usage: {data.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    
    if 'bird_id' in data.columns:
        logger.info(f"Number of birds: {data['bird_id'].nunique()}")
    
    if 'behavior' in data.columns:
        logger.info(f"\nBehavior distribution:")
        dist = data['behavior'].value_counts()
        for behavior, count in dist.items():
            pct = 100 * count / len(data)
            logger.info(f"  {behavior}: {count} ({pct:.1f}%)")
    
    if 'timestamp' in data.columns:
        logger.info(f"\nTime range:")
        logger.info(f"  Start: {data['timestamp'].min()}")
        logger.info(f"  End: {data['timestamp'].max()}")
        duration = (data['timestamp'].max() - data['timestamp'].min()).total_seconds()
        logger.info(f"  Duration: {duration/3600:.2f} hours")
    
    logger.info("="*50)