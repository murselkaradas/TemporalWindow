import numpy as np
import pynapple as nap
import pandas as pd
def calculate_dff_perievent(glom_session, odor_key, baseline_window=(-0.6, -0.1), response_window=(-1, 2)):
    """Calculate dF/F for perievent data around odor trials."""
    odor_epochs = glom_session['odor_epochs'][odor_key]
    ts = nap.Ts(t=odor_epochs.start)
    
    # Compute perievent data
    perievent = nap.compute_perievent_continuous(
        timeseries=glom_session['fluorescence_tsd'],
        tref=ts,
        minmax=response_window
    )
    
    # Get baseline data
    baseline_mask = (perievent.t >= baseline_window[0]) & (perievent.t <= baseline_window[1])
    if not baseline_mask.any():
        raise ValueError(f"No data in baseline window {baseline_window}")
    
    baseline_data = perievent.d[baseline_mask, :]
    f0_values = np.nanmean(baseline_data, axis=0)  # Baseline fluorescence
    
    # Handle 3D (time, trials, cells) or 2D (time, features) data
    if len(perievent.d.shape) == 3:
        n_timepoints, n_trials, n_cells = perievent.d.shape
        dff_data = np.zeros_like(perievent.d)
        
        for trial in range(n_trials):
            for cell in range(n_cells):
                f0 = f0_values[trial, cell]
                dff_data[:, trial, cell] = (perievent.d[:, trial, cell] - f0) / f0 if f0 > 0 else np.nan
        
        column_names = [f'trial_{t}_cell_{c}' for t in range(n_trials) for c in range(n_cells)]
        dff_perievent = nap.TsdTensor(t=perievent.t, d=dff_data)
    else:
        n_timepoints, n_features = perievent.d.shape
        n_trials = len(ts)
        n_cells = n_features // n_trials if n_features % n_trials == 0 else n_features
        n_trials = n_trials if n_features % n_trials == 0 else 1
        
        dff_data_reshaped = np.zeros_like(perievent.d)
        for feature in range(n_features):
            f0 = f0_values[feature]
            dff_data_reshaped[:, feature] = (perievent.d[:, feature] - f0) / f0 if f0 > 0 else np.nan
        
        column_names = [f'trial_{t}_cell_{c}' for t in range(n_trials) for c in range(n_cells)] if n_trials > 1 and n_cells > 1 else [f'feature_{f}' for f in range(n_features)]
        dff_perievent = nap.TsdFrame(t=perievent.t, d=dff_data_reshaped, columns=column_names)
    
    # Summary statistics
    mean_f0, std_f0 = np.nanmean(f0_values), np.nanstd(f0_values)
    n_zero_f0 = np.sum(f0_values <= 0)
    
    return {
        'dff_perievent': dff_perievent,
        'f0_values': f0_values,
        'raw_perievent': perievent,
        'trial_info': {
            'odor_key': odor_key,
            'n_trials': len(ts),
            'baseline_window': baseline_window,
            'response_window': response_window,
            'trial_starts': ts.t
        }
    }

def calculate_dff_all_odors(glom_session, baseline_window=(-0.6, -0.1), response_window=(-1, 2)):
    """Calculate dF/F for all odor conditions in the session."""
    if 'odor_epochs' not in glom_session:
        raise ValueError("No odor_epochs found in session")
    
    return {
        odor_key: calculate_dff_perievent(glom_session, odor_key, baseline_window, response_window)
        for odor_key in glom_session['odor_epochs'].keys()
    }

def detect_onset_latency_single_trial(dff_signal, time_vector, thA=5, thB=3, 
                                     min_peak_time=0.1, window_duration=0.5, 
                                     baseline_start=-0.5, baseline_end=-0.05):
    """Detect onset latency for a single trial."""
    baseline_idx = (time_vector >= baseline_start) & (time_vector < baseline_end)
    if not baseline_idx.any():
        return {'onset_time': window_duration, 'baseline_std': np.nan, 
                'max_response': np.nan, 'mean_response': np.nan, 'responsive': False}
    
    baseline_mean, baseline_std = np.nanmean(dff_signal[baseline_idx]), np.nanstd(dff_signal[baseline_idx])
    if baseline_std == 0 or np.isnan(baseline_std):
        return {'onset_time': window_duration, 'baseline_std': baseline_std, 
                'max_response': np.nan, 'mean_response': np.nan, 'responsive': False}
    
    valid_idx = (time_vector >= 0) & (time_vector <= window_duration)
    if not valid_idx.any():
        return {'onset_time': window_duration, 'baseline_std': baseline_std, 
                'max_response': np.nan, 'mean_response': np.nan, 'responsive': False}
    
    dff_windowed = dff_signal[valid_idx] - baseline_mean
    time_valid = time_vector[valid_idx]
    mean_response, max_response = np.nanmean(dff_windowed), np.nanmax(dff_windowed)
    
    if np.isnan(max_response):
        return {'onset_time': window_duration, 'baseline_std': baseline_std, 
                'max_response': max_response, 'mean_response': mean_response, 'responsive': False}
    
    max_idx = np.nanargmax(dff_windowed)
    max_time = time_valid[max_idx]
    onset_time, responsive = window_duration, False
    
    if mean_response > thB * baseline_std and max_response > thB * baseline_std and max_time >= min_peak_time:
        lat = np.where(dff_windowed > baseline_std * thA)[0]
        if len(lat) >= 4:
            onset_time = time_valid[lat[0]] + 1e-3 / max_response
            responsive = True
    elif mean_response < -thB * baseline_std and max_response > thB * baseline_std and max_time >= min_peak_time:
        lat = np.where(dff_windowed < -thB * baseline_std)[0]
        if len(lat) >= 4:
            onset_time = -time_valid[lat[0]] + 2.0 + 1e-3 / max_response
            responsive = True
    
    return {
        'onset_time': onset_time,
        'baseline_std': baseline_std,
        'max_response': max_response,
        'mean_response': mean_response,
        'responsive': responsive
    }

def calculate_onset_latencies_all_trials(dff_results, thA=4, thB=3, 
                                        min_peak_time=0.1, window_duration=0.5,
                                        baseline_start=-0.5, baseline_end=-0.05):
    """Calculate onset latencies for all trials across all odors."""
    latency_results = {}
    
    for odor_key, dff_result in dff_results.items():
        dff_perievent = dff_result['dff_perievent']
        time_vector = dff_perievent.t
        n_trials = dff_result['trial_info']['n_trials']
        
        n_timepoints, n_features = dff_perievent.d.shape
        n_cells = n_features // n_trials if n_features % n_trials == 0 else n_features
        n_trials = n_trials if n_features % n_trials == 0 else 1
        
        trial_metrics = []
        for trial_idx in range(n_trials):
            trial_dff = dff_perievent.d[:, trial_idx * n_cells:(trial_idx + 1) * n_cells] if n_trials > 1 and n_cells > 1 else dff_perievent.d[:, trial_idx:trial_idx + 1]
            trial_avg = np.nanmean(trial_dff, axis=1)
            
            result = detect_onset_latency_single_trial(
                trial_avg, time_vector, thA, thB, min_peak_time, window_duration, baseline_start, baseline_end
            )
            result['trial_idx'] = trial_idx
            result['odor'] = odor_key
            trial_metrics.append(result)
        
        responsive_trials = [m for m in trial_metrics if m['responsive']]
        responsive_latencies = [m['onset_time'] for m in responsive_trials if m['onset_time'] < window_duration]
        trials_to_process = n_trials - 1 if n_trials > 1 else n_trials
        
        latency_results[odor_key] = {
            'trial_metrics': trial_metrics,
            'onset_latencies': [m['onset_time'] for m in trial_metrics],
            'n_trials': trials_to_process,
            'n_total_trials': n_trials,
            'n_responsive': len(responsive_trials),
            'response_rate': len(responsive_trials) / trials_to_process if trials_to_process > 0 else 0,
            'mean_latency': np.mean(responsive_latencies) if responsive_latencies else np.nan,
            'median_latency': np.median(responsive_latencies) if responsive_latencies else np.nan,
            'parameters': {
                'thA': thA, 'thB': thB, 'min_peak_time': min_peak_time,
                'window_duration': window_duration, 'baseline_start': baseline_start,
                'baseline_end': baseline_end
            }
        }
    
    return latency_results


def detect_fluorescence_drops(glom_session, threshold_method='percentile', 
                            threshold_value=10, min_drop_duration=5):
    """
    Detect fluorescence drops and mark them as NaN
    
    Parameters:
    -----------
    glom_session : dict
        Your glom session data
    threshold_method : str
        'percentile' - use percentile of data
        'absolute' - use absolute threshold value
        'median_fraction' - fraction of median value
    threshold_value : float
        Threshold value (percentile if method='percentile', 
        absolute value if method='absolute',
        fraction if method='median_fraction')
    min_drop_duration : int
        Minimum number of consecutive points below threshold to consider as drop
    
    Returns:
    --------
    Modified glom_session with NaN values where drops detected
    """
    
    # Get fluorescence data
    fluorescence_data = glom_session['fluorescence_tsd'].copy()
    
    # Calculate threshold based on method
    if threshold_method == 'percentile':
        threshold = np.percentile(fluorescence_data, threshold_value)
    elif threshold_method == 'absolute':
        threshold = threshold_value
    elif threshold_method == 'median_fraction':
        threshold = np.median(fluorescence_data) * threshold_value
    else:
        raise ValueError("threshold_method must be 'percentile', 'absolute', or 'median_fraction'")
    
    print(f"Threshold calculated: {threshold:.2f}")
    print(f"Current median: {np.median(fluorescence_data):.2f}")
    
    # Find points below threshold
    below_threshold = fluorescence_data < threshold
    
    # Find consecutive sequences below threshold
    # Use numpy to find runs of True values
    diff = np.diff(np.concatenate(([False], below_threshold.values, [False])).astype(int))
    run_starts = np.where(diff == 1)[0]
    run_ends = np.where(diff == -1)[0]
    
    # Mark runs longer than min_drop_duration as NaN
    modified_data = fluorescence_data.copy()
    drop_count = 0
    
    for start, end in zip(run_starts, run_ends):
        run_length = end - start
        if run_length >= min_drop_duration:
            modified_data[start:end] = 0.5*(modified_data[start-1] + modified_data[end]) if start > 0 and end < len(modified_data) else np.nan
            drop_count += 1
            print(f"Drop detected: indices {start}-{end} ({run_length} points)")
    
    print(f"Total drops detected and marked as NaN: {drop_count}")
    print(f"Total NaN values added: {np.sum(np.isnan(modified_data)) - np.sum(np.isnan(fluorescence_data))}")
    
    # Update the session data
    glom_session_modified = glom_session.copy()
    glom_session_modified['fluorescence_tsd'] = modified_data
    
    return glom_session_modified, threshold

# Example usage with your data:
# Method 1: Use 5th percentile as threshold (catches the lowest 5% of values)
# glom_session_cleaned, threshold = detect_fluorescence_drops(
#     glom_session, 
#     threshold_method='median_fraction',  # 'percentile', 'absolute', or 'median_fraction'
#     threshold_value=0.5,  # 50% of median
#     min_drop_duration=1  # at least 3 consecutive low points
# )