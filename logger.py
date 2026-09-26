"""Blip viewer: the charge trace and the Brio frames around each known blip.

    python blip_viewer.py                      # uses SESSION below
    python blip_viewer.py "E:/.../session_X"   # or pass a session folder
    python blip_viewer.py "E:/.../session_X" --export   # summary PNGs only

SESSION can also be the FLIES folder itself - the newest session_* in it is
used.

"Known blips" are pulled from three places and merged (anything within
MERGE_S of each other becomes one event):
    events.csv      what the live script triggered on, plus your M presses
    EXTRA_TIMES     your own notes (e.g. 610 s)
    detection       optional offline pass over electrometer.csv, same
                    jump / swing rules as the live trigger

Window (interactive):
    top left        whole session, charge in pC, events marked,
                    cyan bands = where Brio frames exist
    bottom left     +-WINDOW_S around the current event, charge relative to
                    the baseline just before it
    right           Brio frame nearest the cursor (red line)

Keys:
    n / p           next / previous event
    right / left    step one Brio frame
    up / down       jump +-1 s
    space           play / pause at PLAY_FPS
    home            back to the event time
    b               jump to the nearest Brio frame
    o               open the nearest Brio frame in Explorer
    s               save a snapshot PNG of the window
    e               export summary PNGs for every event
    click           top plot: move there (and re-center the zoom)
                    bottom plot: move the cursor there

Output goes to <session>/analysis/:
    events_summary.csv      one row per event: time, sources, charge step
    event_###_t####.png     zoomed charge + a row of Brio thumbnails
    snapshot_t####.png      whatever 's' saved
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============ CONFIG ============
SESSION = Path(r'E:\Ben Christensen\FLIES')   # a session_* folder, or FLIES

EXTRA_TIMES = [610.0]      # your own blip notes, seconds since Start

WINDOW_S   = 10.0          # half-width of the zoom plot
BASELINE_S = (-3.0, -0.5)  # baseline window relative to the event
POST_S     = (0.5, 3.0)    # "after" window for the charge-step estimate

DETECT          = True     # also find blips offline in electrometer.csv
DETECT_JUMP_MV  = 1.0      # sample-to-sample step
DETECT_SWING_MV = 1.0      # swing inside DETECT_SWING_S (0 = off)
DETECT_SWING_S  = 5.0
IGNORE_BEFORE_S = 60.0     # skip the Keithley warmup
MERGE_S         = 3.0      # events closer than this become one

MAX_FRAME_GAP_S = 0.5      # no frame this close -> show "no frame"
PLAY_FPS        = 20
N_THUMBS        = 8        # thumbnails per exported event
THUMB_SPAN      = (-2.0, 5.0)   # s around the event the thumbnails cover

DEFAULT_CAP_F = 1e-9       # used if meta.txt has no cap_F
# ================================


# ------------------------------------------------------------- loading

def pick_session(p):
    p = Path(p)
    if p.name.startswith('session_'):
        return p
    sessions = sorted(p.glob('session_*'))
    if not sessions:
        sys.exit(f'no session_* folders in {p}')
    print(f'[load] newest session: {sessions[-1].name}')
    return sessions[-1]


def read_meta(d):
    meta = {}
    p = d / 'meta.txt'
    if p.exists():
        for line in open(p):
            if '\t' in line:
                k, v = line.rstrip('\n').split('\t', 1)
                meta[k] = v
    return meta


class Session:
    def __init__(self, d):
        self.dir = d
        self.meta = read_meta(d)
        self.cap_f = float(self.meta.get('cap_F', DEFAULT_CAP_F))

        # --- electrometer ---
        e = pd.read_csv(d / 'electrometer.csv')
        self.t = e['time'].to_numpy(float)
        if 'voltage_V' in e.columns:
            v = e['voltage_V'].to_numpy(float)
            self.mv = v * 1e3
            self.q = v * self.cap_f * 1e12          # pC
            print(f'[load] voltage mode, C = {self.cap_f * 1e9:g} nF '
                  f'-> 1 mV = {self.cap_f * 1e9:g} pC')
        else:                                        # older charge-mode run
            self.q = e['charge'].to_numpy(float) * 1e12
            self.mv = self.q / (self.cap_f * 1e9)
            print('[load] charge mode session')
        print(f'[load] {len(self.t):,} electrometer samples, '
              f'{self.t[-1] / 60:.1f} min')

        # --- Brio continuous frames ---
        self.ft = np.array([])
        self.fpath = []
        self.flamp = []
        idx = d / 'brio_cont_frames.csv'
        if idx.exists():
            fr = pd.read_csv(idx).sort_values('time')
            self.ft = fr['time'].to_numpy(float)
            self.fpath = [d / 'brio_cont' / f for f in fr['filename']]
            self.flamp = (fr['lamp'].astype(str).tolist()
                          if 'lamp' in fr.columns else ['?'] * len(fr))
            print(f'[load] {len(self.ft):,} Brio frames in '
                  f'{d / "brio_cont"}')
        else:
            print('[load] no brio_cont_frames.csv - trace only')

        # contiguous stretches of Brio coverage (split on gaps > 1 s)
        self.segments = []
        if len(self.ft):
            cuts = np.nonzero(np.diff(self.ft) > 1.0)[0]
            starts = np.r_[0, cuts + 1]
            ends = np.r_[cuts, len(self.ft) - 1]
            for a, b in zip(starts, ends):
                self.segments.append((self.ft[a], self.ft[b], self.flamp[a],
                                      self.fpath[a].name, self.fpath[b].name))
            print('[load] Brio coverage:')
            for t0, t1, lamp, n0, n1 in self.segments:
                print(f'         {t0:8.1f} - {t1:8.1f} s  lamp {lamp}  '
                      f'{n0} .. {n1}')

        self.events = self._build_events()

    # --- events ---

    def _detect(self):
        ok = ~np.isnan(self.mv) & (self.t >= IGNORE_BEFORE_S)
        tt, xx = self.t[ok], self.mv[ok]
        hits = []
        if len(xx) < 2:
            return hits
        d = np.diff(xx)
        for i in np.nonzero(np.abs(d) >= DETECT_JUMP_MV)[0] + 1:
            hits.append((tt[i], 'detected', f'jump {d[i - 1]:+.2f} mV'))
        if DETECT_SWING_MV:
            s = pd.Series(xx, index=pd.to_timedelta(tt, unit='s'))
            r = s.rolling(f'{DETECT_SWING_S}s')
            mask = (((s - r.min()) >= DETECT_SWING_MV)
                    | ((r.max() - s) >= DETECT_SWING_MV)).to_numpy()
            rising = mask & ~np.r_[False, mask[:-1]]
            for i in np.nonzero(rising)[0]:
                hits.append((tt[i], 'detected', 'swing'))
        return hits

    def _build_events(self):
        raw = []
        p = self.dir / 'events.csv'
        if p.exists():
            ev = pd.read_csv(p)
            for _, row in ev.iterrows():
                src = str(row.get('source', 'trigger'))
                raw.append((float(row['time']), src, f'event {row["event"]}'))
        for t in EXTRA_TIMES:
            raw.append((float(t), 'note', f'note {t:g} s'))
        if DETECT:
            raw += self._detect()

        raw.sort()
        merged = []
        for t, src, lab in raw:
            if merged and t - merged[-1]['t'] <= MERGE_S:
                merged[-1]['sources'].add(src)
                merged[-1]['labels'].append(lab)
            else:
                merged.append({'t': t, 'sources': {src}, 'labels': [lab]})

        for m in merged:
            m['step_pC'] = self.step(m['t'])
            fi, gap = self.nearest_frame(m['t'])
            m['lamp'] = self.flamp[fi] if fi is not None and gap < 1 else '-'
        return merged

    # --- helpers ---

    def baseline(self, t_ev):
        m = (self.t >= t_ev + BASELINE_S[0]) & (self.t < t_ev + BASELINE_S[1])
        return np.nanmean(self.q[m]) if m.any() else np.nan

    def step(self, t_ev):
        m = (self.t > t_ev + POST_S[0]) & (self.t <= t_ev + POST_S[1])
        post = np.nanmean(self.q[m]) if m.any() else np.nan
        return post - self.baseline(t_ev)

    def nearest_frame(self, t):
        if len(self.ft) == 0:
            return None, np.inf
        i = int(np.searchsorted(self.ft, t))
        cands = [j for j in (i - 1, i) if 0 <= j < len(self.ft)]
        j = min(cands, key=lambda k: abs(self.ft[k] - t))
        return j, abs(self.ft[j] - t)

    def frame_rgb(self, i):
        img = cv2.imread(str(self.fpath[i]), cv2.IMREAD_COLOR)
        return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def write_summary(self, out_dir):
        rows = []
        for k, m in enumerate(self.events):
            fi, gap = self.nearest_frame(m['t'])
            rows.append({
                'idx': k + 1,
                'time_s': round(m['t'], 3),
                'sources': '+'.join(sorted(m['sources'])),
                'labels': '; '.join(m['labels']),
                'step_pC': round(m['step_pC'], 4)
                           if not np.isnan(m['step_pC']) else '',
                'lamp': m['lamp'],
                'nearest_frame': (self.fpath[fi].name
                                  if fi is not None and gap < MAX_FRAME_GAP_S
                                  else ''),
            })
        out = out_dir / 'events_summary.csv'
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f'[save] {out}')

    def print_events(self):
        print(f'\n{len(self.events)} events:')
        for k, m in enumerate(self.events):
            step = (f'{m["step_pC"]:+8.3f} pC' if not np.isnan(m['step_pC'])
                    else '      -    ')
            print(f'  {k + 1:3d}  t={m["t"]:9.2f} s  {step}  '
                  f'lamp {m["lamp"]}  {"+".join(sorted(m["sources"]))}')
        print()


# ------------------------------------------------------------- export

SRC_COLOR = {'trigger': 'tab:red', 'manual': 'tab:blue',
             'note': 'tab:green', 'detected': 'tab:orange'}


def ev_color(m):
    for s in ('trigger', 'manual', 'note', 'detected'):
        if s in m['sources']:
            return SRC_COLOR[s]
    return 'k'


def export_event(S, k, out_dir):
    m = S.events[k]
    t_ev = m['t']
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(2, N_THUMBS, height_ratios=[1.3, 1])

    ax = fig.add_subplot(gs[0, :])
    w = (S.t >= t_ev - WINDOW_S) & (S.t <= t_ev + WINDOW_S)
    base = S.baseline(t_ev)
    ax.plot(S.t[w] - t_ev, S.q[w] - base, '.-', ms=3, lw=0.8, color='k')
    ax.axvline(0, color=ev_color(m), lw=1)
    ax.axvspan(*BASELINE_S, color='0.9', zorder=0)
    ax.axvspan(*POST_S, color='#e6f0ff', zorder=0)
    ax.set_xlim(-WINDOW_S, WINDOW_S)
    ax.set_xlabel('time from event [s]')
    ax.set_ylabel('dQ vs baseline [pC]')
    step = (f'{m["step_pC"]:+.3f} pC' if not np.isnan(m['step_pC'])
            else 'n/a')
    ax.set_title(f'event {k + 1}   t = {t_ev:.2f} s   step {step}   '
                 f'lamp {m["lamp"]}   {"+".join(sorted(m["sources"]))}')

    offsets = np.linspace(THUMB_SPAN[0], THUMB_SPAN[1], N_THUMBS)
    for j, dt in enumerate(offsets):
        a = fig.add_subplot(gs[1, j])
        a.set_xticks([])
        a.set_yticks([])
        fi, gap = S.nearest_frame(t_ev + dt)
        img = S.frame_rgb(fi) if fi is not None and gap < MAX_FRAME_GAP_S \
            else None
        if img is None:
            a.set_facecolor('0.85')
            a.text(0.5, 0.5, 'no frame', ha='center', va='center',
                   transform=a.transAxes)
        else:
            a.imshow(img)
        a.set_title(f'{dt:+.1f} s', fontsize=9)
        ax.axvline(dt, color='0.6', lw=0.5, ls=':')

    fig.tight_layout()
    out = out_dir / f'event_{k + 1:03d}_t{t_ev:07.1f}.png'
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def export_all(S):
    out_dir = S.dir / 'analysis'
    out_dir.mkdir(exist_ok=True)
    S.write_summary(out_dir)
    for k in range(len(S.events)):
        out = export_event(S, k, out_dir)
        print(f'[export] {k + 1}/{len(S.events)}  {out.name}')
    print(f'[export] done -> {out_dir}')


# ------------------------------------------------------------- viewer

class Viewer:
    def __init__(self, S):
        self.S = S
        self.ei = 0
        self.center = S.events[0]['t'] if S.events else S.t[0]
        self.tc = self.center
        self.fi = None
        self.playing = False

        for k in ('keymap.back', 'keymap.forward', 'keymap.save',
                  'keymap.pan', 'keymap.home'):
            plt.rcParams[k] = []

        self.fig = plt.figure(figsize=(16, 9))
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1.25, 1],
                                   height_ratios=[1, 1.3])
        self.ax_full = self.fig.add_subplot(gs[0, 0])
        self.ax_zoom = self.fig.add_subplot(gs[1, 0])
        self.ax_img = self.fig.add_subplot(gs[:, 1])

        # full trace
        self.ax_full.plot(S.t, S.q, lw=0.6, color='k')
        for m in S.events:
            self.ax_full.axvline(m['t'], color=ev_color(m), lw=0.8,
                                 alpha=0.7)
        self.ax_full.set_xlabel('time [s]')
        self.ax_full.set_ylabel('Q [pC]')
        self.ax_full.set_title('whole session   (red trigger, blue manual, '
                               'green note, orange detected, '
                               'cyan = Brio frames)', fontsize=9)
        for t0, t1, *_ in S.segments:
            self.ax_full.axvspan(t0, t1, color='tab:cyan', alpha=0.15,
                                 zorder=0)
        self.cur_full = self.ax_full.axvline(self.tc, color='r', lw=1)
        self.span_full = self.ax_full.axvspan(0, 0, color='y', alpha=0.25)

        # zoom
        self.zoom_line, = self.ax_zoom.plot([], [], '.-', ms=3, lw=0.8,
                                            color='k')
        self.ev_lines_zoom = []
        self.cur_zoom = self.ax_zoom.axvline(self.tc, color='r', lw=1)
        self.ax_zoom.set_xlabel('time [s]')
        self.ax_zoom.set_ylabel('dQ vs baseline [pC]')
        self.ax_zoom.axhline(0, color='0.7', lw=0.5)

        # image
        self.ax_img.set_xticks([])
        self.ax_img.set_yticks([])
        self.im = None

        self.info = self.fig.text(0.01, 0.005, '', fontsize=9,
                                  family='monospace')
        self.fig.text(0.99, 0.005,
                      'n/p event   <-/-> frame   up/down 1 s   space play   '
                      'home   b nearest frame   o open   s snapshot   '
                      'e export   click to move',
                      fontsize=8, ha='right', color='0.4')

        self.timer = self.fig.canvas.new_timer(interval=int(1000 / PLAY_FPS))
        self.timer.add_callback(self._tick)

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.tight_layout(rect=(0, 0.02, 1, 1))

        if S.events:
            self.goto_event(0)
        else:
            self.set_center(self.center)
            self.set_time(self.tc)

    # --- state changes ---

    def set_center(self, t):
        S = self.S
        self.center = t
        base = S.baseline(t)
        if np.isnan(base):
            base = 0.0
        w = (S.t >= t - WINDOW_S) & (S.t <= t + WINDOW_S)
        self.zoom_line.set_data(S.t[w], S.q[w] - base)
        self.ax_zoom.set_xlim(t - WINDOW_S, t + WINDOW_S)
        self.ax_zoom.relim()
        self.ax_zoom.autoscale_view(scalex=False)

        for ln in self.ev_lines_zoom:
            ln.remove()
        self.ev_lines_zoom = [
            self.ax_zoom.axvline(m['t'], color=ev_color(m), lw=1, alpha=0.8)
            for m in S.events if abs(m['t'] - t) <= WINDOW_S]

        self.span_full.remove()
        self.span_full = self.ax_full.axvspan(t - WINDOW_S, t + WINDOW_S,
                                              color='y', alpha=0.25)

    def set_time(self, t):
        S = self.S
        self.tc = t
        self.cur_full.set_xdata([t, t])
        self.cur_zoom.set_xdata([t, t])
        if abs(t - self.center) > WINDOW_S * 0.9:
            self.set_center(t)

        fi, gap = S.nearest_frame(t)
        self.fi = fi
        if fi is not None and gap < MAX_FRAME_GAP_S:
            img = S.frame_rgb(fi)
            title = (f'{S.fpath[fi].name}   t={S.ft[fi]:.2f} s   '
                     f'lamp {S.flamp[fi]}')
        elif fi is not None:
            img = None
            side = 'after' if S.ft[fi] > t else 'before'
            title = (f'no Brio frame here - nearest is {S.fpath[fi].name}, '
                     f'{gap:.1f} s {side}\n(b = jump to it, o = open in '
                     f'Explorer)')
        else:
            img = None
            title = 'no Brio frames in this session'
        if img is None:
            img = np.full((720, 1280, 3), 200, np.uint8)
        if self.im is None:
            self.im = self.ax_img.imshow(img)
        else:
            self.im.set_data(img)
        self.ax_img.set_title(title, fontsize=9)

        # readout
        i = int(np.clip(np.searchsorted(S.t, t), 0, len(S.t) - 1))
        base = S.baseline(self.center)
        dq = S.q[i] - base if not np.isnan(base) else np.nan
        ev = ''
        if S.events:
            m = S.events[self.ei]
            step = (f'{m["step_pC"]:+.3f}' if not np.isnan(m['step_pC'])
                    else 'n/a')
            ev = (f'event {self.ei + 1}/{len(S.events)} at '
                  f'{m["t"]:.2f} s ({"+".join(sorted(m["sources"]))}, '
                  f'step {step} pC)   ')
        self.info.set_text(
            f'{ev}cursor t={t:.2f} s   Q={S.q[i]:.3f} pC   '
            f'dQ={dq:+.3f} pC   V={S.mv[i]:.3f} mV')
        self.fig.canvas.draw_idle()

    def goto_event(self, k):
        if not self.S.events:
            return
        self.ei = k % len(self.S.events)
        t = self.S.events[self.ei]['t']
        self.set_center(t)
        self.set_time(t)

    def step_frame(self, d):
        S = self.S
        if len(S.ft) == 0:
            return
        fi, _ = S.nearest_frame(self.tc)
        fi = int(np.clip(fi + d, 0, len(S.ft) - 1))
        self.set_time(S.ft[fi])

    # --- playback ---

    def _tick(self):
        if not self.playing:
            return
        S = self.S
        fi, _ = S.nearest_frame(self.tc)
        if fi is None or fi >= len(S.ft) - 1:
            self.toggle_play()
            return
        self.set_time(S.ft[fi + 1])

    def toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self.timer.start()
        else:
            self.timer.stop()

    # --- input ---

    def on_key(self, ev):
        k = ev.key
        if k == 'n':
            self.goto_event(self.ei + 1)
        elif k == 'p':
            self.goto_event(self.ei - 1)
        elif k == 'right':
            self.step_frame(+1)
        elif k == 'left':
            self.step_frame(-1)
        elif k == 'up':
            self.set_time(self.tc + 1.0)
        elif k == 'down':
            self.set_time(self.tc - 1.0)
        elif k == ' ':
            self.toggle_play()
        elif k == 'home':
            self.goto_event(self.ei)
        elif k == 'b':
            fi, _ = self.S.nearest_frame(self.tc)
            if fi is not None:
                self.set_time(self.S.ft[fi])
        elif k == 'o':
            self.open_in_explorer()
        elif k == 's':
            out_dir = self.S.dir / 'analysis'
            out_dir.mkdir(exist_ok=True)
            out = out_dir / f'snapshot_t{self.tc:07.1f}.png'
            self.fig.savefig(out, dpi=110)
            print(f'[save] {out}')
        elif k == 'e':
            export_all(self.S)

    def open_in_explorer(self):
        fi, _ = self.S.nearest_frame(self.tc)
        target = self.S.fpath[fi] if fi is not None \
            else self.S.dir / 'brio_cont'
        print(f'[open] {target}')
        try:
            if os.name == 'nt':
                subprocess.Popen(['explorer', '/select,', str(target)])
            else:
                subprocess.Popen(['xdg-open', str(target.parent)])
        except Exception as e:
            print(f'[open] failed: {e}')

    def on_click(self, ev):
        tb = getattr(self.fig.canvas, 'toolbar', None)
        if tb is not None and getattr(tb, 'mode', ''):
            return                      # zoom/pan tool is active
        if ev.xdata is None:
            return
        if ev.inaxes is self.ax_full:
            self.set_center(ev.xdata)
            self.set_time(ev.xdata)
        elif ev.inaxes is self.ax_zoom:
            self.set_time(ev.xdata)


# ------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session', nargs='?', default=str(SESSION))
    ap.add_argument('--export', action='store_true',
                    help='write summary PNGs and exit')
    args = ap.parse_args()

    S = Session(pick_session(args.session))
    S.print_events()

    if args.export:
        export_all(S)
        return

    out_dir = S.dir / 'analysis'
    out_dir.mkdir(exist_ok=True)
    S.write_summary(out_dir)
    Viewer(S)
    plt.show()


if __name__ == '__main__':
    main()