from scipy.signal import butter, lfilter, iirfilter, filtfilt, iirnotch

import numpy as np


# Strategy B

# Let's answer the questions before we start coding:

#     - Is 40Hz the right cutoff for EEG?
#     Usually yes, as EEG signals of interest are typically below 40Hz. (note: if you analyze gamma waves, you might want a higher cutoff)

#     - What about high-pass filtering?
#     High-pass filtering can be useful to remove slow drifts and focus on higher frequency activity. (e.g. 0.5Hz cutoff)
#     EEG signals often occupy the 0.5-40Hz band, so a bandpass filter (0.5-40Hz) is often ideal.

#     - Should you use bandpass instead?
#     Often, yes. As we said, EEG preprocessing typically uses bandpass filters like 0.5-40 Hz.
# 	(That’s equivalent to applying a high-pass (0.5 Hz) and a low-pass (40 Hz) together.)

#     - What about notch filtering for powerline interference?
#     Notch filtering at 50Hz or 60Hz (depending on local powerline frequency) is common to remove electrical noise.
#     Europe uses 50Hz, North America uses 60Hz.

# TODO: Students may want to implement additional filtering:
# - High-pass filter to remove DC drift
# - Notch filter for 50/60 Hz powerline noise
# - Bandpass filter (e.g., 0.5-40 Hz for EEG)

# Zero-phase Butterworth filter implementation (filtfilt)
def butter_filter(data, cutoff_low, cutoff_high, fs, btype='bandpass', order=5):
    """
    EXAMPLE IMPLEMENTATION: Simple low-pass Butterworth filter.

    Students should understand this basic filter and consider:
    - Is 40Hz the right cutoff for EEG?
    - What about high-pass filtering?
    - Should you use bandpass instead?
    - What about notch filtering for powerline interference?

    Args:
        data (np.ndarray): The input signal.
        cutoff (float): The cutoff frequency of the filter.
        fs (int): The sampling frequency of the signal.
        order (int): The order of the filter.

    Returns:
        np.ndarray: The filtered signal.
    """

    nyquist = 0.5 * fs

    if btype == 'bandpass':
        low = cutoff_low / nyquist
        high = cutoff_high / nyquist
        Wn = [low, high]
    elif btype == 'highpass':
        # normalized cutoff by Nyquist frequency
        # use cutoff_low for highpass (cutoff_low holds the high-pass cutoff)
        normal_cutoff = cutoff_low / nyquist
        Wn = normal_cutoff
    elif btype == 'lowpass':
        # use cutoff_high for lowpass (cutoff_high holds the low-pass cutoff)
        normal_cutoff = cutoff_high / nyquist
        Wn = normal_cutoff
    
    # filter coefficients
    b, a = butter(order, Wn, btype=btype, analog=False)
    y = filtfilt(b, a, data)
    return y


def single_notch_filter(notch_freq, fs, Q=30):
    """
    Applies a Notch (Band-Stop) filter to remove fixed-frequency noise.
    Uses lfilter (standard IIR) as notch is typically very narrow.
    """
    # Return filter coefficients for a notch at notch_freq (normalized inside)
    nyquist = 0.5 * fs
    w0 = notch_freq / nyquist

    # Design notch using the stable iirnotch helper (normalized freq, Q)
    b, a = iirnotch(w0, Q)
    return b, a


# Task B
# Problem: 50 Hz (Europe) or 60 Hz (US) noise from electrical equipment
# Solution: Notch filter at powerline frequency + harmonics
def multi_notch_filter(data, fundamental_freq, fs, Q=30):
    """
    Applies Notch filters at the fundamental frequency and its harmonics,
    respecting the Nyquist limit (fs/2).
    """
    filtered_signal = np.copy(data)
    nyquist = 0.5 * fs
    
    # Iterate through harmonics (1st, 2nd, 3rd, ...)
    h = 1
    while True:
        notch_freq = h * fundamental_freq

        # Check Nyquist limit: Filter frequency must be < fs/2
        if notch_freq >= nyquist:
            break

        # Design and apply the single notch filter (get coefficients)
        b, a = single_notch_filter(notch_freq, fs, Q=Q)
        # Apply the filter using lfilter (standard IIR)
        filtered_signal = lfilter(b, a, filtered_signal)

        h += 1
    return filtered_signal


def standardize_signal(data):
    """
    Normalizes the signal to have zero mean and unit standard deviation.
    """
    if data.ndim == 0 or np.std(data) < 1e-6: # Avoid division by zero
        return data - np.mean(data)
    return (data - np.mean(data)) / np.std(data)


# --- NEW: PADDING FUNCTION FOR EDGE MITIGATION ---

def apply_padding(signal, fs, padding_time_s=3):
    """
    Applies reflection padding to a signal to mitigate filter edge effects.
    
    Args:
        signal (np.ndarray): The 30-second epoch signal.
        fs (int): Sampling frequency.
        padding_time_s (int): The duration of padding (in seconds) on each side.
        
    Returns:
        np.ndarray: The padded signal and the number of padding samples added/removed.
    """
    padding_samples = int(padding_time_s * fs)
    
    # Reflection padding (mode='even') is generally preferred over zero-padding
    # as it minimizes the discontinuity at the edges.
    padded_signal = np.pad(signal, (padding_samples, padding_samples), mode='reflect')
    
    return padded_signal, padding_samples


# --- 2. PREPROCESSING LOGIC ---

def preprocess_multi_channel(multi_channel_data, config):
    """
    Preprocesses multi-channel data (EEG, EOG, EMG) implementing the 
    HPF -> Notch -> LPF pipeline with zero-phase filtering and edge mitigation.
    """
    preprocessed_data = {}
    
    # --- FILTER SETTINGS ---
    POWERLINE_FREQ = getattr(config, 'POWERLINE_FREQ', 50) 
    FILTER_ORDER = 5 
    NOTCH_Q = 30 
    PADDING_TIME_S = 3 # 3 seconds of padding (3 * fs samples)

    CHANNEL_INFO = {
        'eeg': {'fs': 125, 'hpf': 0.5, 'lpf': 30, 'amp_thresh': 100}, 
        'eog': {'fs': 50,  'hpf': 0.3, 'lpf': 10, 'amp_thresh': 150},
        'emg': {'fs': 125, 'hpf': 10, 'lpf': 100, 'amp_thresh': 200},
    }

    artifact_flags = {}
    MAX_AMPLITUDE_THRESHOLD = getattr(config, 'MAX_AMPLITUDE_THRESHOLD', None)

    for ch_name, channel_set in multi_channel_data.items():
        if ch_name not in CHANNEL_INFO: continue
            
        info = CHANNEL_INFO[ch_name]
        fs = info['fs']
        num_epochs, num_channels, num_samples = channel_set.shape
        preprocessed_set = np.zeros_like(channel_set, dtype=float)
        
        if ch_name not in artifact_flags:
            artifact_flags[ch_name] = np.zeros((num_epochs, num_channels), dtype=bool)

        print(f"Applying pipeline to {ch_name} (BP: {info['hpf']}-{info['lpf']} Hz, Notch: {POWERLINE_FREQ} Hz) @ {fs} Hz")
        
        ch_threshold = MAX_AMPLITUDE_THRESHOLD if MAX_AMPLITUDE_THRESHOLD is not None else info['amp_thresh']

        for ch in range(num_channels):
            for epoch in range(num_epochs):
                signal = channel_set[epoch, ch, :]
                
                # 1. ARTIFACT REJECTION (Amplitude Check)
                if np.max(np.abs(signal)) > ch_threshold:
                    preprocessed_set[epoch, ch, :] = np.nan
                    artifact_flags[ch_name][epoch, ch] = True
                    continue 
                
                # 2. PADDING FOR EDGE MITIGATION
                padded_signal, pad_s = apply_padding(signal, fs, PADDING_TIME_S)
                
                # Filter limits check
                nyq = 0.5 * fs
                hpf_cut = max(1e-3, min(info['hpf'], nyq * 0.499))
                lpf_cut = max(1e-3, min(info['lpf'], nyq * 0.499))

                # 3. FILTERING SEQUENCE (HP -> Notch -> LP)
                
                # a. HIGH-PASS (Baseline Wander Removal) - Zero Phase
                signal_hpf = butter_filter(
                    padded_signal, hpf_cut, 0, fs, btype='highpass', order=FILTER_ORDER
                )
                
                # b. MULTI-NOTCH Filter (Powerline) - Standard Phase
                signal_notched = multi_notch_filter(
                    signal_hpf, POWERLINE_FREQ, fs, Q=NOTCH_Q
                )
                
                # c. LOW-PASS (High-Freq Noise Removal) - Zero Phase
                signal_filtered_padded = butter_filter(
                    signal_notched, 0, lpf_cut, fs, btype='lowpass', order=FILTER_ORDER
                )
                
                # 4. REMOVE PADDING (Isolate the clean 30-second segment)
                signal_filtered = signal_filtered_padded[pad_s:-pad_s]

                # 5. STANDARDIZATION (Normalization)
                signal_standardized = standardize_signal(signal_filtered)

                preprocessed_set[epoch, ch, :] = signal_standardized
        
        preprocessed_data[ch_name] = preprocessed_set

    # 6. Global Artifact Flag (for epochs where ANY channel was bad)
    all_artifact_flags = np.sum([np.sum(flags, axis=1) for flags in artifact_flags.values()], axis=0) > 0
    preprocessed_data['artifact_mask'] = all_artifact_flags
    
    print(f"Total Epochs Marked for Rejection (Global Mask): {np.sum(all_artifact_flags)}")

    return preprocessed_data


def preprocess(data, config):
    """
    STUDENT IMPLEMENTATION AREA: Preprocess data based on current iteration.
    (Entry point calling the multi-channel logic)
    """
    print(f"Preprocessing data for iteration {config.CURRENT_ITERATION}...")

    # Detect data format
    is_multi_channel = isinstance(data, dict) and any(key in data for key in ['eeg', 'eog', 'emg'])

    if is_multi_channel:
        return preprocess_multi_channel(data, config)
    else:
        # Fallback for single-channel (retains padding for completeness)
        if getattr(config, 'CURRENT_ITERATION', None) == 1:
            print("Processing single-channel data with zero-phase HP, Notch, and LP.")
            fs = 125
            
            padded_signal, pad_s = apply_padding(data, fs, 3)
            
            # 1. HP for Baseline Wander
            signal_hpf = butter_filter(padded_signal, 0.5, 0, fs, btype='highpass')
            # 2. Notch
            signal_notched = multi_notch_filter(signal_hpf, 60, fs)
            # 3. LP for high-freq noise
            signal_lp_padded = butter_filter(signal_notched, 0, 30, fs, btype='lowpass')
            
            signal_lp = signal_lp_padded[pad_s:-pad_s]
            
            return standardize_signal(signal_lp)
        else:
            return data


# Backward compatibility placeholder (for reference)
def preprocess_single_channel(data, config):
    raise NotImplementedError("Single-channel preprocessing is handled by the updated 'preprocess' function.")

# Original lowpass filter (kept for completeness but unused)
def lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y
