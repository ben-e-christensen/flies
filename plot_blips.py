"""Plot a 1-minute charge window centered on each blip event.

Works on both session formats:
    old (charge mode)   electrometer.csv: time,charge,trigger      (C)
                        events.csv:       event,time,lamp,frames
    new (voltage mode)  electrometer.csv: time,voltage_V,trigger   (V)
                        events.csv:       event,time,source,lamp,rec_cams,frames
In voltage mode Q = C * V, with C read from meta.txt (cap_F), default 1 nF.

If the session has FLIR bursts (flir_frames.csv), a second x-axis on top
labels the FLIR frame number at its timestamp, e.g. 0, 25, 50 ... in
e001_0000.png, e001_0025.png ...  The span the burst covers is shaded.

Usage:
    python plot_blips.py path/to/session_folder
"""

import csv
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============ CONFIG ============
WINDOW_S = 30.0          # seconds before and after the blip timestamp
DEFAULT_CAP_F = 1e-9     # used if meta.txt has no cap_F
FLIR_TICK_EVERY = 25     # label FLIR frame numbers that are multiples of this
FLIR_MAX_TICKS  = 24     # ...stepping up (50, 75, 100 ...) if more would fit
# ================================


def read_meta(session):
    meta = {}
    p = session / 'meta.txt'
    if p.exists():
        for line in open(p):
            if '\t' in line:
                k, v = line.rstrip('\n').split('\t', 1)
                meta[k] = v
    return meta


def load_charge(session):
    """Returns (t, q_pC, label) whatever the session format."""
    csv_path = session / 'electrometer.csv'
    with open(csv_path) as f:
        header = f.readline().strip().split(',')
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1,
                         filling_values=np.nan)
    t = data[:, header.index('time')]

    if 'voltage_V' in header:
        cap = float(read_meta(session).get('cap_F', DEFAULT_CAP_F))
        v = data[:, header.index('voltage_V')]
        q = v * cap * 1e12
        print(f'Voltage mode, C = {cap * 1e9:g} nF  (1 mV = '
              f'{cap * 1e9:g} pC)')
    else:
        q = data[:, header.index('charge')] * 1e12
        print('Charge mode')
    return t, q


def load_events(session):
    events = []
    with open(session / 'events.csv', newline='') as f:
        for row in csv.DictReader(f):
            try:
                events.append({
                    'event':  int(row['event']),
                    'time':   float(row['time']),
                    'source': row.get('source', 'trigger') or 'trigger',
                    'lamp':   (row.get('lamp') or '-').strip(),
                    'rec':    row.get('rec_cams', ''),
                })
            except (KeyError, ValueError):
                continue
    return events


def load_flir(session):
    """{event: (times, frame_numbers)} from flir_frames.csv, or {}."""
    p = session / 'flir_frames.csv'
    if not p.exists():
        return {}
    num_re = re.compile(r'e(\d+)_(\d+)')
    by_event = {}
    with open(p, newline='') as f:
        for row in csv.DictReader(f):
            m = num_re.search(row['filename'])
            if not m:
                continue
            by_event.setdefault(int(row['event']), []).append(
                (float(row['time']), int(m.group(2))))
    out = {}
    for ev, rows in by_event.items():
        rows.sort()
        out[ev] = (np.array([r[0] for r in rows]),
                   np.array([r[1] for r in rows]))
    print(f'FLIR bursts for events: {sorted(out)}')
    return out


def add_flir_axis(ax, ft, fn, event):
    """Top x-axis: FLIR frame numbers at their timestamps."""
    # FLIR_MAX_TICKS is for a burst spanning the whole plot; a burst that
    # covers a third of the width gets a third as many labels
    x0, x1 = ax.get_xlim()
    frac = min(1.0, (ft[-1] - ft[0]) / (x1 - x0))
    max_ticks = max(4, int(FLIR_MAX_TICKS * frac))
    need = len(fn) / max_ticks                   # frames per label, at least
    nice = [FLIR_TICK_EVERY * k for k in (1, 2, 4, 6, 8, 10, 20, 40)]
    step = next((n for n in nice if n >= need), nice[-1])
    pick = (fn % step) == 0
    ax.axvspan(ft[0], ft[-1], color='tab:cyan', alpha=0.12, zorder=0,
               label=f'FLIR e{event:03d}_{fn[0]:04d}..{fn[-1]:04d}')

    top = ax.twiny()
    top.set_xlim(ax.get_xlim())
    top.set_xticks(ft[pick])
    top.set_xticklabels([str(n) for n in fn[pick]], fontsize=7)
    top.set_xlabel(f'FLIR frame #  (e{event:03d}_####, every {step})',
                   fontsize=8)
    top.tick_params(axis='x', length=3, pad=1)
    return top


def main():
    if len(sys.argv) < 2:
        sys.exit('Usage: python plot_blips.py <session_folder>')

    session = Path(sys.argv[1])
    for name in ('electrometer.csv', 'events.csv'):
        if not (session / name).exists():
            sys.exit(f'Not found: {session / name}')

    t, q = load_charge(session)
    events = load_events(session)
    flir = load_flir(session)
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
                   alpha=0.6, label=f'{ev["source"]} @ {tc:.1f}s')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Charge [pC]')
        ax.set_xlim(tb[0], tb[-1])
        if ev['event'] in flir:
            ft, fn = flir[ev['event']]
            add_flir_axis(ax, ft, fn, ev['event'])

        lamp_tag = (f'lamp {ev["lamp"].upper()}'
                    if ev['lamp'] not in ('-', '', 'o') else '')
        ax.set_title(f'Event {ev["event"]} ({ev["source"]})  {lamp_tag}  '
                     f'[{tb[0]:.1f} – {tb[-1]:.1f} s]')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
        fig.tight_layout()

        out = session / f'blip_{ev["event"]:03d}.png'
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f'  Saved {out.name}')

    print('Done.')


if __name__ == '__main__':
    main()