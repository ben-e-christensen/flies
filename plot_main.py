"""Plot charge-mode electrometer sessions — one PNG per block.

Usage:
    python plot_charge.py path/to/session_folder
    python plot_charge.py path/to/electrometer.csv

Reads electrometer.csv (time, charge in Coulombs) and produces per-block
charge [pC] plots with mean/std-dev annotations.

Block format:
    (start_s, end_s, label, snapshot)

    start_s   — start time in seconds (None = beginning of file)
    end_s     — end time in seconds   (None = end of file)
    label     — string label (optional, auto-numbered)
    snapshot  — if True, also produce 10-second segment PNGs (optional,
                default False)

Output (per block):
    <label>.png
    <label>_seg_000.0-010.0s.png   (if snapshot=True)
    ...
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============ CONFIG ============
SEGMENT_DURATION = 10.0       # seconds per snapshot window

# (start_s, end_s, label, snapshot)
BLOCKS = [(0,100, 'baseline', False),]
# ================================


def load_session(path: Path):
    """Return (time, charge) arrays in SI from an electrometer.csv."""
    if path.is_dir():
        csv = path / 'electrometer.csv'
    else:
        csv = path
    if not csv.exists():
        sys.exit(f'Not found: {csv}')

    data = np.genfromtxt(csv, delimiter=',', skip_header=1,
                         filling_values=np.nan)
    if data.ndim != 2 or data.shape[1] < 2:
        sys.exit(f'Unexpected shape in {csv}: {data.shape}')

    t = data[:, 0]
    q = data[:, 1]
    return t, q, csv.parent


def parse_blocks(blocks, t):
    """Parse block tuples into a clean list of dicts."""
    out = []
    for i, entry in enumerate(blocks):
        t0 = entry[0]
        t1 = entry[1]
        name = entry[2] if len(entry) >= 3 else None
        snapshot = entry[3] if len(entry) >= 4 else False

        i0 = 0 if t0 is None else np.searchsorted(t, t0)
        i1 = len(t) if t1 is None else np.searchsorted(t, t1)

        t0_s = t0 if t0 is not None else 0
        t1_s = t1 if t1 is not None else t[-1]

        if name:
            label = name
            title = f'{name}  [{t0_s:.1f} \u2013 {t1_s:.1f} s]'
        else:
            label = f'block_{i}'
            title = f'block {i}  [{t0_s:.1f} \u2013 {t1_s:.1f} s]'

        out.append({
            'label': label,
            'title': title,
            'i0': i0,
            'i1': i1,
            't0': t0_s,
            't1': t1_s,
            'snapshot': snapshot,
        })
    return out


def slugify(s):
    """Turn a label into a safe filename."""
    return s.lower().replace(' ', '_').replace('-', '_')


def plot_block(tb, qb, title, out_path):
    """Single-panel charge plot. Saves to out_path."""
    q_pc = qb * 1e12

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(tb, q_pc, 'r.-', markersize=2, alpha=0.8)
    ax.axhline(0, color='k', linewidth=0.3)
    ax.set_ylabel('Charge [pC]')
    ax.set_xlabel('Time [s]')
    ax.set_title(title)
    ax.yaxis.set_major_locator(plt.MaxNLocator(20))
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  Saved {out_path.name}')


def main():
    if len(sys.argv) < 2:
        sys.exit('Usage: python plot_charge.py '
                 '<session_folder or electrometer.csv>')

    path = Path(sys.argv[1])
    t, q, out_dir = load_session(path)

    n_nan = np.isnan(q).sum()
    print(f'Loaded {len(t)} samples, {n_nan} NaN')
    print(f'Time span: {t[-1] - t[0]:.3f} s')
    print(f'Charge range: {np.nanmin(q)*1e12:.4f} \u2013 {np.nanmax(q)*1e12:.4f} pC')

    if not BLOCKS:
        blocks = [{'label': 'all', 'title': 'all', 'i0': 0, 'i1': len(t),
                    't0': 0, 't1': t[-1], 'snapshot': False}]
    else:
        blocks = parse_blocks(BLOCKS, t)

    for blk in blocks:
        tb = t[blk['i0']:blk['i1']]
        qb = q[blk['i0']:blk['i1']]

        if len(tb) == 0:
            print(f'\n[{blk["label"]}]  \u2014 no data in range, skipping')
            continue

        slug = slugify(blk['label'])

        q_std = np.nanstd(qb * 1e12)
        print(f'\n[{blk["label"]}]  {len(tb)} pts  '
              f'\u03c3_Q={q_std:.4f} pC')

        # --- main block plot ---
        plot_block(tb, qb, blk['title'],
                   out_dir / f'{slug}.png')

        # --- segment snapshots ---
        if blk['snapshot']:
            seg_start = tb[0]
            seg_end = tb[-1]
            cursor = seg_start
            seg_num = 0

            while cursor < seg_end:
                win_end = min(cursor + SEGMENT_DURATION, seg_end)
                si0 = np.searchsorted(tb, cursor)
                si1 = np.searchsorted(tb, win_end)

                if si1 <= si0:
                    cursor = win_end
                    continue

                seg_t = tb[si0:si1]
                seg_q = qb[si0:si1]

                seg_title = (f'{blk["label"]}  '
                             f'[{cursor:.1f} \u2013 {win_end:.1f} s]')
                seg_fname = f'{slug}_seg_{cursor:07.1f}-{win_end:07.1f}s.png'

                plot_block(seg_t, seg_q, seg_title,
                           out_dir / seg_fname)

                cursor = win_end
                seg_num += 1

            print(f'  \u2192 {seg_num} segment snapshots')

    print('\nDone.')


if __name__ == '__main__':
    main()