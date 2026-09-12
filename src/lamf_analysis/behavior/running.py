"""Running behavior metrics.

To get a BehaviorSessionDataset object:

from comb.behavior_session_dataset import BehaviorSessionDataset
raw_data_dir = "/root/capsule/data/multiplane-ophys_687000_2023-09-26_08-56-32"
bsd = BehaviorSessionDataset(raw_data_dir)
"""

import numpy as np


def total_run_distance(bsd, time_window=None):
    ''' Calculate total run distance from BehaviorSessionDataset object
    Returns in meters
    '''
    running_df = bsd.running_speed.copy()
    if time_window is not None:
        assert len(time_window) == 2, "time_window should be a tuple of (start_time, end_time)"
        start_time, end_time = time_window
        assert start_time < end_time, "start_time should be less than end_time"
        running_df = running_df[(running_df.timestamps >= start_time) & (running_df.timestamps <= end_time)]
    total_distance = running_df.speed.sum() * np.median(np.diff(bsd.running_speed.timestamps)) / 100  # in m/s
    return total_distance


def process_running_speed(bsd,
                          mean_window_size: float = 2.0, # seconds
                          std_window_size: float = 1.0, # seconds
                          run_threshold: float = 2.0,  # speed threshold for running
                          jitter_threshold: float = 0.5,  # std threshold for jittering
                          ):
    running_df = bsd.running_speed
    recording_rate = 1 / np.median(np.diff(running_df['timestamps']))
    mean_samples = int(recording_rate * mean_window_size)
    std_samples = int(recording_rate * std_window_size)
    # Calculate rolling features
    running_df['roll_mean'] = running_df['speed'].rolling(mean_samples,
                                                        center=True).mean()
    running_df['roll_std'] = running_df['speed'].rolling(std_samples,
                                                        center=True).std()

    # Logic for Detection
    # 1. Running: Mean speed is high
    running_df['is_running'] = running_df['roll_mean'] > run_threshold

    # 2. Jittering: High variability but not moving far
    running_df['is_jittering'] = (running_df['roll_std'] > jitter_threshold) & (~running_df['is_running'])


def get_running_epochs(
    bsd,
    behavior_type: str,  # 'running' or 'stationary'
    duration_threshold_in_s: float = 5.0,
    time_window: tuple | None = None,
    running_speed_processing_args: dict | None = None,
):
    """Get running or stationary epochs from BehaviorSessionDataset object.
    Returns list of (start_time, end_time) tuples.
    """
    if running_speed_processing_args is None:
        running_speed_processing_args = {}
    process_running_speed(bsd, **running_speed_processing_args)
    running_df = bsd.running_speed.copy()
    if time_window is not None:
        assert len(time_window) == 2, "time_window should be a tuple of (start_time, end_time)"
        start_time, end_time = time_window
        assert start_time < end_time, "start_time should be less than end_time"
        running_df = running_df[(running_df.timestamps >= start_time) & (running_df.timestamps <= end_time)]
    is_running = running_df['is_running'].values
    is_jittering = running_df['is_jittering'].values
    timestamps = running_df['timestamps'].values
    epochs = []
    in_epoch = False
    for i in range(1, len(is_running)):
        if behavior_type == 'running':
            condition = is_running[i]
        elif behavior_type == 'stationary':
            condition = not is_running[i] and not is_jittering[i]
        else:
            raise ValueError("behavior_type must be 'running' or 'stationary'")
        
        if condition and not in_epoch:
            start_time = timestamps[i]
            in_epoch = True
        elif not condition and in_epoch:
            end_time = timestamps[i]
            in_epoch = False
            if (end_time - start_time) >= duration_threshold_in_s:
                epochs.append((start_time, end_time))
    # Check if we ended while still in an epoch
    if in_epoch:
        end_time = timestamps[-1]
        if (end_time - start_time) >= duration_threshold_in_s:
            epochs.append((start_time, end_time))
    return epochs

def plot_running_speed_with_epochs(bsd, epoch='stationary', color='orange', ax=None):
    import matplotlib.pyplot as plt
    ts = bsd.running_speed.timestamps
    spd = bsd.running_speed.speed
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ts, spd, label='Running Speed (cm/s)')

    epochs = get_running_epochs(bsd, behavior_type=epoch)
    for start_time, end_time in epochs:
        m = (ts >= start_time) & (ts <= end_time)
        ax.plot(ts[m], spd[m], color=color)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Running Speed (cm/s)')
    return ax