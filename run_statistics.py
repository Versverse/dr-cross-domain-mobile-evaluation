"""Recompute statistics from the retained inputs; does not train or run inference."""
from pathlib import Path
import os
import subprocess
import sys

if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    if os.name == 'nt' and not str(root).startswith('\\\\?\\'):
        root = Path('\\\\?\\' + str(root))
    required = [root/'05_verification/inputs', root/'08_figures/Figure2_original_20261015.png']
    for path in required:
        if not path.exists():
            raise FileNotFoundError(str(path))
    output = root/'07_statistics/recomputed'
    if output.exists():
        raise FileExistsError('Move or rename your previous recomputed directory before a new run; retained experiment outputs are not overwritten.')
    for name in ['extend_statistics.py', 'verify_and_prepare_manuscript.py']:
        subprocess.run([sys.executable, str(root/'07_statistics/analysis'/name)], check=True)
    print('Recomputation and checks complete: 07_statistics/recomputed')
