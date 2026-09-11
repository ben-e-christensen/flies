"""Plot a 1-minute charge window centered on each blip event.

Usage:
    python plot_blips.py path/to/session_folder
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============ CONFIG ============
WINDOW_S = 30.0   # seconds before and after the blip timestamp
# ================================


def main():
    if len(sys.argv) < 2:
        sys.exit('Usage: python plot_blips.py <session_folder>')

    session = Path(sys.argv[1])
    csv = session / 'electrometer.csv'
    evt = session / 'events.csv'
    if not csv.exists():
        sys.exit(f'Not found: {csv}')
    if not evt.exists():
        sys.exit(f'Not found: {evt}')

    data = np.genfromtxt(csv, delimiter=',', skip_header=1, filling_values=np.nan)
    t = data[:, 0]
    q = data[:, 1] * 1e12

    events = []
    with open(evt) as f:
        f.readline()
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 4:
                continue
            events.append({
                'event': int(parts[0]),
                'time':  float(parts[1]),
                'lamp':  parts[2].strip(),
                'frames': int(parts[3]),
            })

    print(f'Loaded {len(t)} samples, {len(events)} events')

    for ev in events:
        tc = ev['time']
        i0 = np.searchsorted(t, tc - WINDOW_S)
        i1 = np.searchsorted(t, tc + WINDOW_S)
        if i1 <= i0:
            print(f'  event {ev["event"]}: no data in window, skipping')
            continue

        tb = t[i0:i1]
        qb = q[i0:i1]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(tb, qb, 'r.-', markersize=2, alpha=0.8)
        ax.axvline(tc, color='blue', linewidth=1, linestyle='--',
                   alpha=0.6, label=f'blip @ {tc:.1f}s')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Charge [pC]')

        lamp_tag = f'lamp {ev["lamp"].upper()}' if ev['lamp'] != '-' else ''
        ax.set_title(f'Event {ev["event"]}  {lamp_tag}  '
                     f'[{tb[0]:.1f} – {tb[-1]:.1f} s]')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()

        out = session / f'blip_{ev["event"]:03d}.png'
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f'  Saved {out.name}')

    print('Done.')


if __name__ == '__main__':
    main()