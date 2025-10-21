"""Save visualization plots from sample EDF/XML into a visualizations/ folder.

Usage:
    python3 Python/tools/save_visualizations.py

This script uses the project's visualization helpers and saves:
 - visualizations/epoch0_signals.png
 - visualizations/hypnogram.png

It runs headless (Agg backend) so it's CI-friendly and works on macOS/Linux.
"""
import os
import sys
import matplotlib
matplotlib.use('Agg')

# Ensure imports work when running from repository root
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from Python.src.visualization import plot_sample_epoch, plot_hypnogram
import matplotlib.pyplot as plt


def main():
    outdir = os.path.abspath('visualizations')
    os.makedirs(outdir, exist_ok=True)

    edf = os.path.abspath(os.path.join('data', 'sample', 'R1.edf'))
    xml = os.path.abspath(os.path.join('data', 'sample', 'R1.xml'))

    # Change to outdir so plot_sample_epoch saves to that folder by default
    cwd = os.getcwd()
    os.chdir(outdir)
    try:
        print('Plotting sample epoch...')
        plot_sample_epoch(edf, epoch_idx=0, epoch_duration=30)

        print('Plotting hypnogram...')
        plot_hypnogram(xml, edf_path=edf)
        # Save current figure explicitly for hypnogram
        fig = plt.gcf()
        fig.savefig('hypnogram.png')

    finally:
        os.chdir(cwd)

    print('Saved plots to', outdir)


if __name__ == '__main__':
    main()
