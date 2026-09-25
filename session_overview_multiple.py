"""One long charge overview across several session folders.

    python combined_overview.py                       # uses FOLDERS below
    python combined_overview.py "E:/.../SAP_good_data" "E:/.../pt2" ...

Each entry can be a session folder (has electrometer.csv) or a folder that
contains session folders - those are expanded.  Folder names don't matter,
so renamed sessions are fine.

Sessions are put in time order using wall_clock_start from meta.txt and laid
end to end ('stitched'), or at their real clock times ('wallclock').

Same conventions as session_overview.py:
    line color     lamps.csv if the session has it, else black
    triangles      events from events.csv along the top, colored by the lamp
                   logged at that event (filled = trigger, hollow = manual)
                   blue = Going into Faraday cage (A), from acrylic
                   orange = Going into acrylic (B), from Faraday cage
                   grey = lamps off

Writes <OUT>.png, <OUT>.svg (for Inkscape) and <OUT>_sessions.csv.
"""

import csv
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ============ CONFIG ============
FOLDERS = [
    r'E:\Ben Christensen\FLIES\SAP_good_data',
    r'E:\Ben Christensen\FLIES\SAP_good_data_pt2',
    r'E:\Ben Christensen\FLIES\SAP_good_data_pt3',
]
OUT = None              # output path without extension; None = next to the
                        # first folder, named combined_overview

X_MODE = 'stitched'     # 'stitched' = end to end, 'wallclock' = real times
GAP_MIN = 1.0           # stitched: grey gap between sessions [min]
SORT_BY_START = True    # False = keep the order of FOLDERS
SUBTRACT_BASELINE = False   # True = shift each session so its first minute
                            # sits at 0 (lines sessions up; changes offsets)
Y_LIMITS = None         # e.g. (-10, 5) to clip one huge spike

MIN_PER_INCH = 2.0      # figure width: this many minutes per inch
WIDTH_RANGE_IN = (16, 90)
HEIGHT_IN = 6
POINTS_PER_SESSION = 20_000   # min/max-decimated for plotting (SVG size)
SAVE_SVG = True

LAMP_LABELS = {'a': 'Going into Faraday cage (A), from acrylic',
               'b': 'Going into acrylic (B), from Faraday cage',
               'o': 'Lamps off',
               '?': 'Lamp not recorded'}
LAMP_COLORS = {'a': 'tab:blue', 'b': 'tab:orange', 'o': '0.45', '?': '0.7'}
DEFAULT_CAP_F = 1e-9
# ================================

NAME_TIME_RE = re.compile(r'(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})')


# ------------------------------------------------------------- loading

def find_sessions(paths):
    out = []
    for p in map(Path, paths):
        if (p / 'electrometer.csv').exists():
            out.append(p)
        elif p.is_dir():
            inner = sorted(q for q in p.iterdir()
                           if (q / 'electrometer.csv').exists())
            if inner:
                print(f'[expand] {p.name}: {len(inner)} sessions inside')
            else:
                print(f'[skip] {p} - no electrometer.csv here or one level in')
            out += inner
        else:
            print(f'[skip] {p} - not found')
    return out


def read_meta(d):
    meta = {}
    p = d / 'meta.txt'
    if p.exists():
        for line in open(p):
            if '\t' in line:
                k, v = line.rstrip('\n').split('\t', 1)
                meta[k] = v
    return meta


def start_time(d, meta):
    """wall_clock_start from meta, else a timestamp in the name, else mtime."""
    try:
        return datetime.fromisoformat(meta['wall_clock_start']), 'meta'
    except (KeyError, ValueError):
        pass
    m = NAME_TIME_RE.search(d.name)
    if m:
        return datetime.strptime(m.group(0), '%Y-%m-%d_%H-%M-%S'), 'name'
    return datetime.fromtimestamp((d / 'electrometer.csv').stat().st_ctime), \
        'file time'


def load(d):
    meta = read_meta(d)
    e = pd.read_csv(d / 'electrometer.csv')
    t = e['time'].to_numpy(float)
    if 'voltage_V' in e.columns:
        try:
            cap = float(meta.get('cap_F', DEFAULT_CAP_F))
        except ValueError:
            cap = DEFAULT_CAP_F
        q = e['voltage_V'].to_numpy(float) * cap * 1e12
    else:
        q = e['charge'].to_numpy(float) * 1e12

    events = []
    p = d / 'events.csv'
    if p.exists():
        with open(p, newline='') as f:
            for row in csv.DictReader(f):
                try:
                    lamp = (row.get('lamp') or '?').strip().lower()
                    events.append({
                        'event': int(row['event']), 'time': float(row['time']),
                        'source': (row.get('source') or 'trigger').strip(),
                        'lamp': lamp if lamp in LAMP_LABELS else '?'})
                except (KeyError, ValueError):
                    pass

    lamps = None
    p = d / 'lamps.csv'
    if p.exists():
        rows = pd.read_csv(p).sort_values('time')
        lamps = (rows['time'].to_numpy(float),
                 np.array([str(x).strip() for x in rows['lamp']], dtype=object))

    start, src = start_time(d, meta)
    return {'dir': d, 'name': d.name, 't': t, 'q': q, 'events': events,
            'lamps': lamps, 'start': start, 'start_src': src}


# ------------------------------------------------------------- drawing

def decimate(t, q, n_max):
    if len(t) <= n_max:
        return np.arange(len(t))
    k = int(np.ceil(len(t) / (n_max / 2)))
    keep = []
    for s in range(0, len(t), k):
        seg = q[s:s + k]
        if np.all(np.isnan(seg)):
            continue
        keep += sorted({s + int(np.nanargmin(seg)), s + int(np.nanargmax(seg))})
    return np.array(keep, dtype=int)


def draw(ax, x, q, states):
    pts = np.column_stack([x, q])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    ok = ~np.isnan(segs).any(axis=(1, 2))
    if states is None:
        cols = 'k'
    else:
        cols = [LAMP_COLORS.get(s, LAMP_COLORS['?']) for s in states[:-1][ok]]
    ax.add_collection(LineCollection(segs[ok], colors=cols, linewidths=0.7))


def lamp_states(lamps, t):
    starts, labels = lamps
    idx = np.searchsorted(starts, t, side='right') - 1
    return np.where(idx >= 0, labels[np.clip(idx, 0, None)], '?')


# ------------------------------------------------------------- main

def main():
    paths = sys.argv[1:] or FOLDERS
    sessions = [load(d) for d in find_sessions(paths)]
    if not sessions:
        sys.exit('no sessions found')
    if SORT_BY_START:
        sessions.sort(key=lambda s: s['start'])

    # --- x positions (minutes) ---
    t_zero = sessions[0]['start']
    offset = 0.0
    for s in sessions:
        if X_MODE == 'wallclock':
            s['x0'] = (s['start'] - t_zero).total_seconds() / 60
        else:
            s['x0'] = offset
            offset += (s['t'][-1] - s['t'][0]) / 60 + GAP_MIN
        if SUBTRACT_BASELINE:
            m = s['t'] <= s['t'][0] + 60
            s['q'] = s['q'] - np.nanmedian(s['q'][m])

    x_end = max(s['x0'] + (s['t'][-1] - s['t'][0]) / 60 for s in sessions)
    width = float(np.clip(x_end / MIN_PER_INCH, *WIDTH_RANGE_IN))
    fig, ax = plt.subplots(figsize=(width, HEIGHT_IN))

    rows, all_x, all_q, ev_x = [], [], [], []
    lamps_seen, any_black = set(), False
    for s in sessions:
        t, q = s['t'], s['q']
        keep = decimate(t, q, POINTS_PER_SESSION)
        x = s['x0'] + (t[keep] - t[0]) / 60
        states = lamp_states(s['lamps'], t[keep]) if s['lamps'] else None
        if states is None:
            any_black = True
        else:
            lamps_seen |= set(states)
        draw(ax, x, q[keep], states)
        all_x.append(x)
        all_q.append(q[keep])

        xs, xe = s['x0'], s['x0'] + (t[-1] - t[0]) / 60
        ax.axvspan(xe, xe + GAP_MIN if X_MODE == 'stitched' else xe,
                   color='0.88', lw=0, zorder=0)
        ax.text(xs, 1.0, f' {s["name"]}\n {s["start"]:%m-%d %H:%M}',
                transform=ax.get_xaxis_transform(), fontsize=8, va='top',
                color='0.25')

        tr = ax.get_xaxis_transform()
        for ev in s['events']:
            ex = s['x0'] + (ev['time'] - t[0]) / 60
            c = LAMP_COLORS[ev['lamp']]
            ax.plot(ex, 1.02, marker='v', ms=7, color=c,
                    mfc=c if ev['source'] == 'trigger' else 'white', mew=1.3,
                    transform=tr, clip_on=False, zorder=5)
            ev_x.append(ex)
            lamps_seen.add(ev['lamp'])

        lc = pd.Series([e['lamp'] for e in s['events']]).value_counts()
        rows.append({'order': len(rows) + 1, 'folder': s['name'],
                     'start': s['start'].isoformat(sep=' '),
                     'start_from': s['start_src'],
                     'x_start_min': round(xs, 2),
                     'duration_min': round((t[-1] - t[0]) / 60, 2),
                     'events': len(s['events']),
                     'faraday_cage': int(lc.get('a', 0)),
                     'acrylic': int(lc.get('b', 0)),
                     'lamps_off': int(lc.get('o', 0)),
                     'lamps_csv': s['lamps'] is not None,
                     'path': str(s['dir'])})
        print(f'[{len(rows)}] {s["name"]:<34} {s["start"]:%Y-%m-%d %H:%M}'
              f' ({s["start_src"]})  {rows[-1]["duration_min"]:6.1f} min  '
              f'{len(s["events"])} events')

    ax.set_xlim(0, x_end)
    allq = np.concatenate(all_q)
    if Y_LIMITS:
        ax.set_ylim(*Y_LIMITS)
    else:
        lo, hi = np.nanmin(allq), np.nanmax(allq)
        pad = 0.05 * (hi - lo or 1)
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel('minutes' + (' since first session start'
                               if X_MODE == 'wallclock'
                               else ' (sessions end to end, grey = break)'))
    ax.set_ylabel('Q [pC]' + ('  (each session zeroed)'
                              if SUBTRACT_BASELINE else ''))
    ax.grid(True, alpha=0.25)

    handles = [Patch(color=LAMP_COLORS[k], label=LAMP_LABELS[k])
               for k in ('a', 'b', 'o', '?') if k in lamps_seen]
    handles += [Line2D([], [], marker='v', color='k', ls='', label='trigger'),
                Line2D([], [], marker='v', color='k', mfc='white', ls='',
                       label='manual (M)')]
    if any_black:
        handles.append(Line2D([], [], color='k',
                              label='charge (no lamps.csv)'))
    # legend in the emptiest corner
    fig.canvas.draw()
    leg = ax.legend(handles=handles, fontsize=8, loc='upper left')
    bb = leg.get_window_extent(fig.canvas.get_renderer()).transformed(
        ax.transAxes.inverted())
    w, h = bb.width + 0.02, bb.height + 0.02
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    xf = (np.concatenate(all_x) - x0) / (x1 - x0)
    yf = (allq - y0) / (y1 - y0)
    boxes = {'lower right': (1 - w, 0, 1, h), 'lower left': (0, 0, w, h),
             'upper right': (1 - w, 1 - h, 1, 1), 'upper left': (0, 1 - h, w, 1)}
    order = list(boxes)
    best = min(order, key=lambda k: (int(((xf >= boxes[k][0]) &
                                          (xf <= boxes[k][2]) &
                                          (yf >= boxes[k][1]) &
                                          (yf <= boxes[k][3])).sum()),
                                     order.index(k)))
    leg.remove()
    ax.legend(handles=handles, fontsize=8, loc=best, framealpha=0.9)

    total = sum(r['duration_min'] for r in rows)
    ax.set_title(f'{len(sessions)} sessions, {total:.0f} min of data, '
                 f'{sum(r["events"] for r in rows)} events', pad=22)
    fig.tight_layout()

    out = Path(OUT) if OUT else Path(paths[0]).parent / 'combined_overview'
    fig.savefig(out.with_suffix('.png'), dpi=110)
    if SAVE_SVG:
        fig.savefig(out.with_suffix('.svg'))
    plt.close(fig)
    pd.DataFrame(rows).to_csv(f'{out}_sessions.csv', index=False)
    print(f'\n-> {out.with_suffix(".png")}'
          + (f'\n-> {out.with_suffix(".svg")}' if SAVE_SVG else '')
          + f'\n-> {out}_sessions.csv   ({width:.0f} in wide)')


if __name__ == '__main__':
    main()