#!/usr/bin/env python3
"""Replot recorded sequential results without loading or training models."""
import argparse
import json
from pathlib import Path
from sequential import plot_summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results_dirs', nargs='+', type=Path,
                        help='Model results directory or parent containing model directories')
    args = parser.parse_args()
    for directory in args.results_dirs:
        files = ([directory / 'results.json'] if (directory / 'results.json').is_file()
                 else sorted(directory.glob('*/results.json')))
        if not files:
            raise FileNotFoundError(f'No results.json under {directory}')
        for path in files:
            plot_summary(json.loads(path.read_text()), path.parent)
            print(path.parent / 'sequential_summary.png', flush=True)
