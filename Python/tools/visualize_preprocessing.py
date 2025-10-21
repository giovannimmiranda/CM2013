"""Visualize raw vs preprocessed signals for a sample EDF recording.

Saves images to visualizations/:
 - preprocessed_eeg_epoch0.png
 - preprocessed_eog_epoch0.png
 - preprocessed_emg_epoch0.png

Run from repo root:
    python3 Python/tools/visualize_preprocessing.py
"""
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace

# Make sure Python modules in the repo are importable when running from repo root
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from Python.src.data_loader import load_training_data
from Python.src.preprocessing import preprocess


def ensure_outdir(path='visualizations'):
    outdir = os.path.abspath(path)
    os.makedirs(outdir, exist_ok=True)
    return outdir


def main():
    outdir = ensure_outdir('visualizations')

    edf_path = os.path.abspath(os.path.join('data', 'sample', 'R1.edf'))
    xml_path = os.path.abspath(os.path.join('data', 'sample', 'R1.xml'))

    print('Loading sample EDF and XML')
    multi_channel_data, _, info = load_training_data(edf_path, xml_path)

    # Minimal config namespace for preprocessing
    config = SimpleNamespace(CURRENT_ITERATION=1, POWERLINE_FREQ=50, MAX_AMPLITUDE_THRESHOLD=None)

    print('Running preprocessing...')
    pre = preprocess(multi_channel_data, config)

    # For each signal type, plot raw vs preprocessed for epoch 0, channel 0 (if present)
    for sig in ['eeg', 'eog', 'emg']:
        if sig not in multi_channel_data or sig not in pre:
            print(f'Skipping {sig}: not present in data')
            continue

        raw = multi_channel_data[sig]
        proc = pre[sig]

        # pick epoch 0, channel 0
        epoch_idx = 0
        ch_idx = 0
        if raw.shape[0] <= epoch_idx:
            print(f'Skipping {sig}: no epoch {epoch_idx}')
            continue
        if raw.shape[1] <= ch_idx:
            print(f'Skipping {sig}: no channel {ch_idx}')
            continue

        raw_sig = raw[epoch_idx, ch_idx, :]
        proc_sig = proc[epoch_idx, ch_idx, :]

        # sampling rate
        fs = info.get(f'{sig}_fs', None)
        if fs is None:
            # fallback defaults used in preprocessing
            fs = 125 if sig in ['eeg', 'emg'] else 50

        t = np.arange(len(raw_sig)) / fs

        # Plot
        fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        axs[0].plot(t, raw_sig, color='tab:blue', linewidth=0.8)
        axs[0].set_title(f'Raw {sig.upper()} - epoch {epoch_idx}, ch {ch_idx}')
        axs[0].set_ylabel('Amplitude')
        axs[0].grid(True)

        # Handle NaNs in proc (artifact marked) by plotting masked
        proc_plot = np.copy(proc_sig)
        if np.isnan(proc_plot).any():
            axs[1].plot(t, proc_plot, color='tab:orange', linewidth=0.8)
            axs[1].set_title(f'Preprocessed {sig.upper()} (NaNs indicate rejected epochs)')
        else:
            axs[1].plot(t, proc_plot, color='tab:orange', linewidth=0.8)
            axs[1].set_title(f'Preprocessed {sig.upper()}')

        axs[1].set_ylabel('Normalized')
        axs[1].set_xlabel('Time (s)')
        axs[1].grid(True)

        plt.tight_layout()
        outname = os.path.join(outdir, f'preprocessed_{sig}_epoch{epoch_idx}.png')
        fig.savefig(outname)
        plt.close(fig)
        print('Saved', outname)

    print('Done')


if __name__ == '__main__':
    main()
