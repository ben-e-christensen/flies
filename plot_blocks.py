import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PREFACTOR = 1e12   # C -> pC
DATA_FOLDER = Path(__file__).resolve().parent / 'Kiethley_data'

# Timestamps (in seconds) that split the data into blocks.
# e.g. [1800, 2750, 3800] -> blocks: 0-1800, 1800-2750, 2750-3800, 3800-end
SPLIT_TIMES = [1800, 2750, 3800]
SPLIT_FIT=[1,1,0,1]
SPLIT_LABELS = ['Idle', 'Fan On', 'Beads In', 'Idle (Beads in, Fan Off)']

# Toggle per-block: break the block into ZOOM_WINDOW-second chunks, each saved
# as its own png in a subfolder next to the main plots.
SPLIT_ZOOM = [0, 0, 1, 0]
ZOOM_WINDOW = 10  # seconds


def pick_file():
    if len(sys.argv) > 1:
        return Path(sys.argv[1])

    csvs = sorted(DATA_FOLDER.glob('*.csv'), key=lambda p: p.stat().st_mtime)
    if not csvs:
        raise SystemExit(f'No CSV files found in {DATA_FOLDER} and none given as an argument.')
    return csvs[-1]


def make_blocks(t, split_times):
    bounds = [t.min()] + sorted(split_times) + [t.max()]
    blocks = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        blocks.append((start, end))
    return blocks


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

    blocks = make_blocks(t, SPLIT_TIMES)

    for i, (start, end) in enumerate(blocks):
        if end <= start:
            continue
        mask = (t >= start) & (t <= end) & valid
        if not np.any(mask):
            print(f'Block {start:g}-{end:g}s: no data, skipping')
            continue

        do_fit = bool(SPLIT_FIT[i]) if i < len(SPLIT_FIT) else False

        plt.figure()
        plt.plot(t[mask], q[mask], 'ro-', markersize=3)

        title = SPLIT_LABELS[i] if i < len(SPLIT_LABELS) else f'{filepath.name} ({start:g}-{end:g}s)'
        if do_fit and np.count_nonzero(mask) >= 2:
            slope, intercept = np.polyfit(t[mask], q[mask], 1)
            fit_line = slope * t[mask] + intercept
            plt.plot(t[mask], fit_line, 'b--', linewidth=1.5, label=f'fit: slope={slope:.4g} pC/s')
            plt.legend()

        plt.xlabel('time [s]')
        plt.ylabel('charge [pC]')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        png_path = filepath.with_name(f'{filepath.stem}_{start:g}-{end:g}s.png')
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f'Saved block {start:g}-{end:g}s to {png_path}')

        do_zoom = bool(SPLIT_ZOOM[i]) if i < len(SPLIT_ZOOM) else False
        if do_zoom:
            save_zoom_windows(filepath, t, q, valid, start, end, i)


def save_zoom_windows(filepath, t, q, valid, start, end, block_index):
    zoom_dir = filepath.with_name(f'{filepath.stem}_zoom_block{block_index}')
    zoom_dir.mkdir(exist_ok=True)

    window_start = start
    while window_start < end:
        window_end = min(window_start + ZOOM_WINDOW, end)
        mask = (t >= window_start) & (t < window_end) & valid
        if np.any(mask):
            plt.figure()
            plt.plot(t[mask], q[mask], 'ro-', markersize=3)
            plt.xlabel('time [s]')
            plt.ylabel('charge [pC]')
            plt.title(f'{filepath.name} ({window_start:g}-{window_end:g}s)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            png_path = zoom_dir / f'{filepath.stem}_{window_start:g}-{window_end:g}s.png'
            plt.savefig(png_path, dpi=150)
            plt.close()

        window_start = window_end

    print(f'Saved zoom windows for block {start:g}-{end:g}s to {zoom_dir}')


if __name__ == '__main__':
    main()
