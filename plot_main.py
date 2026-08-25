import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PREFACTOR = 1e12   # C -> pC
DATA_FOLDER = Path(__file__).resolve().parent / 'Kiethley_data'


def pick_file():
    if len(sys.argv) > 1:
        return Path(sys.argv[1])

    csvs = sorted(DATA_FOLDER.glob('*.csv'), key=lambda p: p.stat().st_mtime)
    if not csvs:
        raise SystemExit(f'No CSV files found in {DATA_FOLDER} and none given as an argument.')
    return csvs[-1]


def main():
    filepath = pick_file()
    if not filepath.exists():
        raise SystemExit(f'File not found: {filepath}')

    data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    t = data[:, 0]
    q = data[:, 1] * PREFACTOR

    valid = ~np.isnan(q)
    n_total = len(q)
    n_valid = int(np.count_nonzero(valid))

    print(f'{filepath.name}: {n_valid}/{n_total} valid readings')

    plt.plot(t, q, 'ro-', markersize=3)
    plt.xlabel('time [s]')
    plt.ylabel('charge [pC]')
    plt.title(filepath.name)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    png_path = filepath.with_suffix('.png')
    plt.savefig(png_path, dpi=150)
    print(f'Saved plot to {png_path}')

    plt.show()


if __name__ == '__main__':
    main()