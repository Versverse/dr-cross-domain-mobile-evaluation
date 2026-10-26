"""Run source confidence intervals using either supported folder spelling."""
from pathlib import Path
import argparse
import os
import subprocess
import sys

if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    if os.name == 'nt' and not str(root).startswith('\\\\?\\'):
        root = Path('\\\\?\\' + str(root))
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-directory', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    candidates = [root/'13_source_intervals', root/'13_source_intervals_20261019']
    if args.source_directory is not None:
        directory = args.source_directory.resolve()
    else:
        present = [p for p in candidates if p.is_dir()]
        if len(present) != 1:
            raise ValueError('Keep one source-interval directory, or select it with --source-directory.')
        directory = present[0]
    output = args.output or directory/'recomputed'
    subprocess.run([sys.executable, str(directory/'recompute_source_intervals.py'),
                    '--inputs', str(directory/'inputs'), '--output', str(output)], check=True)
