"""Plot charge across an entire session's electrometer.csv.

Usage:
    python plot_session.py path/to/session_folder

Marks:
    dashed lines     events from events.csv (red trigger, blue manual)
    cyan bands       spans covered by FLIR bursts (flir_frames.csv), so you
                     can see which parts of the run have video and which
                     were 'dead' periods

Opens an interactive window (use the zoom tool to dig into a stretch) and
saves session_charge.png in the session folder.  Works for both voltage-mode
(Q = C*V, C from meta.txt) and old charge-mode sessions.
"""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============ CONFIG ============
DEFAULT_CAP_F = 1e-9     # used if meta.txt has no cap_F
SHOW = True              # False = just save the PNG
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
    path = session / 'electrometer.csv'
    with open(path) as f:
        header = f.readline().strip().split(',')
    data = np.genfromtxt(path, delimiter=',', skip_header=1,
                         filling_values=np.nan)
    t = data[:, header.index('time')]
    if 'voltage_V' in header:
        cap = float(read_meta(session).get('cap_F', DEFAULT_CAP_F))
        q = data[:, header.index('voltage_V')] * cap * 1e12
        print(f'voltage mode, C = {cap * 1e9:g} nF')
    else:
        q = data[:, header.index('charge')] * 1e12
        print('charge mode')
    return t, q


def load_events(session):
    p = session / 'events.csv'
    if not p.exists():
        return []
    out = []
    with open(p, newline='') as f:
        for row in csv.DictReader(f):
            try:
                out.append((float(row['time']), int(row['event']),
                            row.get('source', 'trigger') or 'trigger'))
            except (KeyError, ValueError):
                pass
    return out


def load_bursts(session):
    """[(t_start, t_end, event)] from flir_frames.csv."""
    p = session / 'flir_frames.csv'
    if not p.exists():
        return []
    spans = {}
    with open(p, newline='') as f:
        for row in csv.DictReader(f):
            ev, t = int(row['event']), float(row['time'])
            lo, hi = spans.get(ev, (t, t))
            spans[ev] = (min(lo, t), max(hi, t))
    return [(lo, hi, ev) for ev, (lo, hi) in sorted(spans.items())]


def main():
    if len(sys.argv) < 2:
        sys.exit('Usage: python plot_session.py <session_folder>')
    session = Path(sys.argv[1])
    if not (session / 'electrometer.csv').exists():
        sys.exit(f'Not found: {session / "electrometer.csv"}')

    t, q = load_charge(session)
    events = load_events(session)
    bursts = load_bursts(session)
    ok = ~np.isnan(q)
    print(f'{len(t):,} samples over {t[-1] / 60:.1f} min '
          f'({(~ok).sum():,} NaN), {len(events)} events, '
          f'{len(bursts)} FLIR bursts')

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(t, q, color='k', lw=0.6)

    for lo, hi, ev in bursts:
        ax.axvspan(lo, hi, color='tab:cyan', alpha=0.18, zorder=0)
    colors = {'trigger': 'tab:red', 'manual': 'tab:blue'}
    for te, n, src in events:
        ax.axvline(te, color=colors.get(src, 'tab:red'), lw=0.8, ls='--',
                   alpha=0.7)
        ax.text(te, 1.0, f' {n}', transform=ax.get_xaxis_transform(),
                fontsize=7, va='bottom', color=colors.get(src, 'tab:red'))

    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel('time since Start [s]')
    ax.set_ylabel('Q [pC]')
    ax.grid(True, alpha=0.3)

    # minutes along the top, for orienting in a long run
    top = ax.secondary_xaxis('top', functions=(lambda s: s / 60,
                                               lambda m: m * 60))
    top.set_xlabel('[min]', fontsize=8)

    ax.set_title(f'{session.name}   ({t[-1] / 60:.1f} min, '
                 f'{len(events)} events; cyan = FLIR bursts, '
                 f'red trigger / blue manual)', fontsize=10, pad=22)
    fig.tight_layout()

    out = session / 'session_charge.png'
    fig.savefig(out, dpi=150)
    print(f'saved {out}')
    if SHOW:
        plt.show()


if __name__ == '__main__':
    main()