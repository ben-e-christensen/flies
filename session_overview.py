"""Whole-session charge + one plot per episode, straight from events.csv.

    python session_overview.py "E:/Ben Christensen/FLIES/session_X"   # one
    python session_overview.py "E:/Ben Christensen/FLIES"             # all

EVENTS ARE THE GROUND TRUTH
    Every event in events.csv is plotted, labeled with the lamp that
    events.csv says was on at that moment:
        blue    Going into Faraday cage (A)
        orange  Going into acrylic (B)
        grey    Lamps off
    Nothing is detected or guessed.

EPISODES
    The trigger can fire several times on one disturbance (e.g. on the RC
    recovery tail of a big spike).  Events closer than MERGE_GAP_S to the
    previous one are grouped into one EPISODE, and each episode gets one
    plot covering all of its events, PRE_S before the first to POST_S
    after the last.

PLOTS
    Events are small numbered triangles along the top edge, colored by
    lamp (filled = trigger, hollow = manual M) - no lines through the data.
    Episode plots share one y axis (EPISODE_Y).  Legends go in whichever
    corner covers the fewest data points.

LINE COLOR
    Sessions with lamps.csv (recorded from now on) get the charge line
    colored by the logged lamp state.  Older sessions get a black line -
    their lamp state between events wasn't recorded, so it isn't drawn.

OUTPUT
    per session   <session>/analysis/overview/
                      session_overview.png
                      episode_##.png
                      episodes.csv   one row per episode
                      events.csv     every event with its episode number
    folder run    <FLIES>/overview_sessions.csv
                  <FLIES>/overview_episodes.csv
"""

import csv
import re
import sys
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
LAMP_LABELS = {'a': 'Going into Faraday cage (A), from acrylic',
               'b': 'Going into acrylic (B), from Faraday cage',
               'o': 'Lamps off',
               '?': 'Lamp not recorded'}
LAMP_COLORS = {'a': 'tab:blue', 'b': 'tab:orange', 'o': '0.45', '?': '0.7'}

MERGE_GAP_S = 20.0     # events closer than this to the previous = same episode
PRE_S       = 20.0     # episode plot starts this long before its first event
POST_S      = 30.0     # ...and ends this long after its last event
BASELINE_S  = (-6.0, -1.0)   # baseline window before the first event

EPISODE_Y = 'global'   # 'global'  = every episode plot in this run shares
                       #             one y range (all sessions)
                       # 'session' = shared within each session
                       # (lo, hi)  = fixed, e.g. (-5, 5)
Y_PAD = 0.05           # headroom above/below the data, as a fraction

LEGEND_PREFER = ('lower right', 'lower left', 'upper right', 'upper left')
                       # tie-break order when choosing the emptiest corner

MAX_PLOT_POINTS = 200_000
DEFAULT_CAP_F = 1e-9
# ================================


# ------------------------------------------------------------- loading

def read_meta(d):
    meta = {}
    p = d / 'meta.txt'
    if p.exists():
        for line in open(p):
            if '\t' in line:
                k, v = line.rstrip('\n').split('\t', 1)
                meta[k] = v
    return meta


def load_charge(d, meta):
    e = pd.read_csv(d / 'electrometer.csv')
    t = e['time'].to_numpy(float)
    if 'voltage_V' in e.columns:
        try:
            cap = float(meta.get('cap_F', DEFAULT_CAP_F))
        except ValueError:
            cap = DEFAULT_CAP_F
        return t, e['voltage_V'].to_numpy(float) * cap * 1e12
    return t, e['charge'].to_numpy(float) * 1e12


def load_events(d):
    p = d / 'events.csv'
    if not p.exists():
        return []
    out = []
    with open(p, newline='') as f:
        for row in csv.DictReader(f):
            try:
                lamp = (row.get('lamp') or '?').strip().lower()
                out.append({'event': int(row['event']),
                            'time': float(row['time']),
                            'source': (row.get('source') or 'trigger').strip(),
                            'lamp': lamp if lamp in LAMP_LABELS else '?'})
            except (KeyError, ValueError):
                pass
    return sorted(out, key=lambda e: e['time'])


def load_lamps(d):
    """[(t0, t1, lamp)] from lamps.csv, or None for older sessions."""
    p = d / 'lamps.csv'
    if not p.exists():
        return None
    rows = pd.read_csv(p).sort_values('time')
    t = rows['time'].tolist()
    lamps = [str(x).strip() for x in rows['lamp']]
    return [(t[i], t[i + 1] if i + 1 < len(t) else np.inf, lamps[i])
            for i in range(len(t))]


def load_flir_spans(d):
    p = d / 'flir_frames.csv'
    if not p.exists():
        return {}
    spans = {}
    with open(p, newline='') as f:
        for row in csv.DictReader(f):
            ev, t = int(row['event']), float(row['time'])
            lo, hi = spans.get(ev, (t, t))
            spans[ev] = (min(lo, t), max(hi, t))
    return spans


# ------------------------------------------------------------- episodes

def group_episodes(events):
    eps = []
    for ev in events:
        if eps and ev['time'] - eps[-1][-1]['time'] <= MERGE_GAP_S:
            eps[-1].append(ev)
        else:
            eps.append([ev])
    return eps


def window_mean(t, q, t0, t1):
    m = (t >= t0) & (t < t1)
    v = q[m]
    v = v[~np.isnan(v)]
    return float(np.mean(v)) if len(v) else np.nan


def episode_row(t, q, k, evs):
    t_first, t_last = evs[0]['time'], evs[-1]['time']
    base = window_mean(t, q, t_first + BASELINE_S[0], t_first + BASELINE_S[1])
    m = (t >= t_first) & (t <= t_last + POST_S)
    seg = q[m] - base
    seg = seg[~np.isnan(seg)]
    end = window_mean(t, q, t_last + POST_S - 5, t_last + POST_S)
    return {
        'episode': k,
        'events': ' '.join(str(e['event']) for e in evs),
        'n_events': len(evs),
        't_first_s': round(t_first, 2),
        't_last_s': round(t_last, 2),
        'lamps': ' '.join(e['lamp'].upper() for e in evs),
        'first_lamp': LAMP_LABELS[evs[0]['lamp']],
        'baseline_pC': round(base, 4),
        'min_dQ_pC': round(float(seg.min()), 4) if len(seg) else np.nan,
        'max_dQ_pC': round(float(seg.max()), 4) if len(seg) else np.nan,
        'dQ_at_end_pC': round(end - base, 4),
    }


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
    return np.array(keep)


def draw_charge(ax, t, q, lamps, lw):
    """Colored by lamps.csv if we have it, black otherwise."""
    if lamps is None:
        ax.plot(t, q, color='k', lw=lw)
        return
    starts = np.array([s[0] for s in lamps])
    labels = np.array([s[2] for s in lamps], dtype=object)
    idx = np.searchsorted(starts, t, side='right') - 1
    states = np.where(idx >= 0, labels[np.clip(idx, 0, None)], '?')
    pts = np.column_stack([t, q])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    ok = ~np.isnan(segs).any(axis=(1, 2))
    cols = [LAMP_COLORS.get(s, LAMP_COLORS['?']) for s in states[:-1][ok]]
    ax.add_collection(LineCollection(segs[ok], colors=cols, linewidths=lw))
    ax.autoscale_view()


def mark_events_top(ax, evs, ms=9, fontsize=9):
    """Small numbered triangles along the top edge - no line through the
    data.  Filled = trigger, hollow = manual (M)."""
    tr = ax.get_xaxis_transform()
    for ev in evs:
        c = LAMP_COLORS[ev['lamp']]
        ax.plot(ev['time'], 1.025, marker='v', ms=ms, color=c,
                mfc=c if ev['source'] == 'trigger' else 'white',
                mew=1.5, transform=tr, clip_on=False, zorder=5)
        ax.text(ev['time'], 1.055, str(ev['event']), color=c,
                fontsize=fontsize, fontweight='bold', ha='center',
                va='bottom', transform=tr)


def smart_legend(ax, handles, t, q, event_times=()):
    """Put the legend in the corner that covers the fewest data points.
    Call after the axis limits are final."""
    leg = ax.legend(handles=handles, fontsize=8, loc='upper left',
                    framealpha=0.9)
    fig = ax.figure
    bb = leg.get_window_extent(fig.canvas.get_renderer())
    bb = bb.transformed(ax.transAxes.inverted())
    w, h = bb.width + 0.02, bb.height + 0.02

    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    ok = ~np.isnan(q)
    xf = (np.asarray(t)[ok] - x0) / (x1 - x0)
    yf = (np.asarray(q)[ok] - y0) / (y1 - y0)
    # the event triangles live along the top edge - avoid covering those too
    ex = (np.asarray(event_times, float) - x0) / (x1 - x0)
    xf = np.r_[xf, np.repeat(ex, 50)]
    yf = np.r_[yf, np.full(len(ex) * 50, 0.98)]

    boxes = {'upper left':  (0, 1 - h, w, 1),
             'upper right': (1 - w, 1 - h, 1, 1),
             'lower left':  (0, 0, w, h),
             'lower right': (1 - w, 0, 1, h)}
    counts = {loc: int(((xf >= a) & (xf <= c) & (yf >= b) & (yf <= d)).sum())
              for loc, (a, b, c, d) in boxes.items()}
    best = min(LEGEND_PREFER, key=lambda loc: (counts[loc],
                                               LEGEND_PREFER.index(loc)))
    leg.remove()
    ax.legend(handles=handles, fontsize=8, loc=best, framealpha=0.9)


def set_ylim(ax, lim):
    if lim is not None and all(np.isfinite(lim)) and lim[1] > lim[0]:
        pad = Y_PAD * (lim[1] - lim[0])
        ax.set_ylim(lim[0] - pad, lim[1] + pad)


def legend_handles(evs, lamps_logged, has_flir):
    present = [s for s in ('a', 'b', 'o', '?') if any(e['lamp'] == s
                                                      for e in evs)]
    h = [Patch(color=LAMP_COLORS[s], label=LAMP_LABELS[s]) for s in present]
    if has_flir:
        h.append(Patch(color='tab:cyan', alpha=0.25, label='FLIR video'))
    if any(e['source'] != 'trigger' for e in evs):
        h += [Line2D([], [], marker='v', color='k', ls='', label='trigger'),
              Line2D([], [], marker='v', color='k', mfc='white', ls='',
                     label='manual (M)')]
    if not lamps_logged:
        h.append(Line2D([], [], color='k', label='charge'))
    return h


# ------------------------------------------------------------- y range

def episode_windows(t, q, episodes):
    for evs in episodes:
        t0, t1 = evs[0]['time'] - PRE_S, evs[-1]['time'] + POST_S
        w = (t >= t0) & (t <= t1)
        if w.sum() >= 2:
            yield t0, t1, w


def episode_yrange(t, q, episodes):
    lo, hi = np.inf, -np.inf
    for _, _, w in episode_windows(t, q, episodes):
        v = q[w]
        v = v[~np.isnan(v)]
        if len(v):
            lo, hi = min(lo, v.min()), max(hi, v.max())
    return (lo, hi) if np.isfinite(lo) else None


def session_yrange(d):
    """Cheap pre-pass for EPISODE_Y = 'global'."""
    t, q = load_charge(d, read_meta(d))
    return episode_yrange(t, q, group_episodes(load_events(d)))


# ------------------------------------------------------------- per session

def do_session(d, ylim=None):
    meta = read_meta(d)
    t, q = load_charge(d, meta)
    events = load_events(d)
    lamps = load_lamps(d)
    flir = load_flir_spans(d)
    episodes = group_episodes(events)
    if isinstance(EPISODE_Y, (tuple, list)):
        ylim = tuple(EPISODE_Y)
    elif EPISODE_Y == 'session' or ylim is None:
        ylim = episode_yrange(t, q, episodes)
    out = d / 'analysis' / 'overview'
    out.mkdir(parents=True, exist_ok=True)

    print(f'\n=== {d.name}   {(t[-1] - t[0]) / 60:.1f} min, '
          f'{len(events)} events in {len(episodes)} episodes'
          f'{"   (lamps.csv)" if lamps else ""}')

    # --- tables ---
    ep_rows, ev_rows = [], []
    for k, evs in enumerate(episodes, 1):
        r = episode_row(t, q, k, evs)
        ep_rows.append({'session': d.name, **r})
        for ev in evs:
            ev_rows.append({'session': d.name, 'episode': k,
                            'event': ev['event'], 'time_s': ev['time'],
                            'source': ev['source'], 'lamp': ev['lamp'],
                            'going_into': LAMP_LABELS[ev['lamp']],
                            'has_flir': ev['event'] in flir})
        print(f'  episode {k:2d}  {r["t_first_s"]:8.1f}-{r["t_last_s"]:8.1f} s'
              f'  events {r["events"]:<14} lamps {r["lamps"]:<10}'
              f'  dQ min {r["min_dQ_pC"]:+8.2f}  max {r["max_dQ_pC"]:+8.2f} pC')
    pd.DataFrame(ep_rows).to_csv(out / 'episodes.csv', index=False)
    pd.DataFrame(ev_rows).to_csv(out / 'events.csv', index=False)

    # --- whole session ---
    keep = decimate(t, q, MAX_PLOT_POINTS)
    fig, ax = plt.subplots(figsize=(16, 5.2))
    draw_charge(ax, t[keep], q[keep], lamps, lw=0.7)
    for k, evs in enumerate(episodes, 1):
        ax.axvspan(evs[0]['time'] - 2, evs[-1]['time'] + 2, color='gold',
                   alpha=0.18, lw=0, zorder=0)
        ax.text(evs[0]['time'], 0.02, f'E{k}',
                transform=ax.get_xaxis_transform(), fontsize=8,
                color='darkgoldenrod')
    ax.set_xlim(t[0], t[-1])
    ax.autoscale_view(scalex=False)
    mark_events_top(ax, events, ms=8, fontsize=7)
    ax.set_xlabel('time since Start [s]')
    ax.set_ylabel('Q [pC]')
    ax.grid(True, alpha=0.25)
    top = ax.secondary_xaxis('bottom', functions=(lambda s: s / 60,
                                                  lambda m: m * 60))
    top.spines['bottom'].set_position(('outward', 36))
    top.set_xlabel('[min]', fontsize=8)
    fig.canvas.draw()
    smart_legend(ax, legend_handles(events, lamps is not None, False),
                 t[keep], q[keep], [e['time'] for e in events])
    ax.set_title(f'{d.name}   {len(events)} events, {len(episodes)} episodes '
                 f'(gold = episode E#)', fontsize=10, pad=30)
    fig.tight_layout()
    fig.savefig(out / 'session_overview.png', dpi=140)
    plt.close(fig)

    # --- one plot per episode ---
    for k, evs in enumerate(episodes, 1):
        r = ep_rows[k - 1]
        t0, t1 = evs[0]['time'] - PRE_S, evs[-1]['time'] + POST_S
        w = (t >= t0) & (t <= t1)
        if w.sum() < 2:
            continue
        fig, ax = plt.subplots(figsize=(12, 4.6))
        has_flir = False
        for ev in evs:
            if ev['event'] in flir:
                has_flir = True
                lo, hi = flir[ev['event']]
                ax.axvspan(lo, hi, color='tab:cyan', alpha=0.10, lw=0,
                           zorder=0)
        draw_charge(ax, t[w], q[w], lamps, lw=1.2)
        if np.isfinite(r['baseline_pC']):
            ax.axhline(r['baseline_pC'], color='0.6', lw=0.6, ls=':')
        ax.set_xlim(t0, t1)
        ax.autoscale_view(scalex=False)
        set_ylim(ax, ylim)
        mark_events_top(ax, evs)
        ax.set_xlabel('time since Start [s]')
        ax.set_ylabel('Q [pC]')
        ax.grid(True, alpha=0.25)
        fig.canvas.draw()
        smart_legend(ax, legend_handles(evs, lamps is not None, has_flir),
                     t[w], q[w], [e['time'] for e in evs])
        runs = []                               # "1-4: acrylic | 5-6: off"
        for e in evs:
            if runs and runs[-1][2] == e['lamp']:
                runs[-1][1] = e['event']
            else:
                runs.append([e['event'], e['event'], e['lamp']])
        lamp_list = (LAMP_LABELS[evs[0]['lamp']] if len(runs) == 1 else
                     '   |   '.join(
                         f'{a if a == b else f"{a}-{b}"}: {LAMP_LABELS[l]}'
                         for a, b, l in runs))
        lo, hi = r['min_dQ_pC'], r['max_dQ_pC']
        peak = lo if abs(lo) > abs(hi) else hi
        which = (f'event {evs[0]["event"]} ({evs[0]["source"]})'
                 if len(evs) == 1 else
                 f'events {evs[0]["event"]}-{evs[-1]["event"]}')
        ax.set_title(f'{d.name}   {which}   -   {lamp_list}\n'
                     f'peak {peak:+.3f} pC', fontsize=9, pad=30)
        fig.tight_layout()
        fig.savefig(out / f'episode_{k:02d}.png', dpi=130)
        plt.close(fig)

    print(f'  -> {out}')
    lamp_counts = pd.Series([e['lamp'] for e in events]).value_counts()
    summary = {'session': d.name,
               'duration_min': round((t[-1] - t[0]) / 60, 2),
               'events': len(events), 'episodes': len(episodes),
               'events_faraday_cage': int(lamp_counts.get('a', 0)),
               'events_acrylic': int(lamp_counts.get('b', 0)),
               'events_lamps_off': int(lamp_counts.get('o', 0)),
               'lamps_csv': lamps is not None}
    return summary, ep_rows


# ------------------------------------------------------------- main

def main():
    if len(sys.argv) < 2:
        sys.exit('Usage: python session_overview.py <session or FLIES folder>')
    root = Path(sys.argv[1])
    sessions = ([root] if (root / 'electrometer.csv').exists() else
                sorted(p for p in root.glob('session_*')
                       if (p / 'electrometer.csv').exists()))
    if not sessions:
        sys.exit(f'no sessions with electrometer.csv under {root}')

    ylim = None
    if EPISODE_Y == 'global':
        lo, hi = np.inf, -np.inf
        for d in sessions:
            try:
                yr = session_yrange(d)
            except Exception:
                yr = None
            if yr:
                lo, hi = min(lo, yr[0]), max(hi, yr[1])
        if np.isfinite(lo):
            ylim = (lo, hi)
            print(f'episode plots share y range {lo:+.2f} .. {hi:+.2f} pC')

    summaries, episodes = [], []
    for d in sessions:
        try:
            s, eps = do_session(d, ylim)
            summaries.append(s)
            episodes += eps
        except Exception as e:
            print(f'\n=== {d.name}: FAILED ({type(e).__name__}: {e})')

    if len(sessions) > 1 and summaries:
        sm = pd.DataFrame(summaries)
        sm.to_csv(root / 'overview_sessions.csv', index=False)
        pd.DataFrame(episodes).to_csv(root / 'overview_episodes.csv',
                                      index=False)
        print('\n' + '=' * 78)
        print(sm.to_string(index=False))
        print(f'\n-> {root / "overview_sessions.csv"}'
              f'\n-> {root / "overview_episodes.csv"}')


if __name__ == '__main__':
    main()