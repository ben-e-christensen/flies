"""V2 - for sessions where the webcams saved CONTINUOUS frames (brio_cont/,
brio1_cont/ ...) instead of bursts.  FLIR still comes from its burst.
Everything else is the same as event_video.py.

Event video: every camera that recorded the event on top, the charge trace
drawing itself in underneath.

    python event_video.py                         # SESSION + EVENTS below
    python event_video.py "E:/.../session_X"      # a different session
    python event_video.py "E:/.../session_X" 3 5  # just events 3 and 5

Cameras are found automatically from the burst folders the acquisition
script writes, for each event:
    flir/e003_0000.png   + flir_frames.csv
    brio1/e003_0000.jpg  + brio1_frames.csv     (brio2, or a single brio,
                                                  on older sessions)

LAYOUT
    +-----------+---------+---------+
    |   FLIR    |  brio1  |  brio2  |   all scaled to one height so the row
    +-----------+---------+---------+   is OUT_W wide
    |        charge (full width)    |
    +-------------------------------+

TIME
    The master clock is the first camera in MASTER_ORDER that has frames
    (the FLIR, normally): one video frame per master frame, played at its
    measured fps x SPEED.  Every other camera shows its most recent frame at
    or before that time - never a future frame - labeled with how old it is.
    The charge trace draws up to that time.  All timestamps are seconds since
    Start, i.e. when each frame arrived in Python (USB webcams lag by tens of
    ms; put a measured lag in CAM_OFFSET_S).

    START_AT / STOP_AT are master image numbers: STOP_AT = 280 ends on
    e###_0280.

OUTPUT   <session>/analysis/videos/
    event_003_0000-0280.mp4
    event_003_0000-0280_sync.csv   per video frame: every camera's file and
                                   time, its offset from the master, and Q
"""

import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============ CONFIG ============
SESSION = Path(r'E:\Ben Christensen\FLIES\session_XXXX')
EVENTS = 'all'          # 'all', or a list like [1, 3]

START_AT = 0            # first master image number
STOP_AT = None          # last master image number (None = end of burst)

MASTER_ORDER = ('flir', 'brio1', 'brio', 'brio2')
CAM_FOLDERS = ('flir', 'brio', 'brio1', 'brio2')
CAM_OFFSET_S = {}       # e.g. {'brio_cont': -0.05} if a camera lags
NO_BURST_WINDOW = (-10.0, 10.0)
MAX_TOP_H = 720         # camera row height cap   # clip around the event when there's no
                                  # FLIR burst (webcam becomes the master)

SPEED = 1.0             # 1 = real time, 0.25 = quarter speed
OUT_W = 1920            # video width
CHARGE_H = 400          # height of the charge panel
PLOT_PAD_S = 0.5        # charge context before first / after last frame
RELATIVE = True         # plot dQ from the first frame instead of Q
FLIR_CONTRAST = 'auto'  # or (lo, hi) raw 16-bit counts

LAMP_LABELS = {'a': 'Going into Faraday cage (A), from acrylic',
               'b': 'Going into acrylic (B), from Faraday cage',
               'o': 'Lamps off', '?': 'Lamp not recorded'}
LAMP_BGR = {'a': (180, 119, 31), 'b': (14, 127, 255), 'o': (110, 110, 110),
            '?': (170, 170, 170)}          # matplotlib blue/orange/grey
LAMP_MPL = {'a': 'tab:blue', 'b': 'tab:orange', 'o': '0.45', '?': '0.7'}

FFMPEG = r'E:\Ben Christensen\ffmpeg-2026-09-21-git-a9cbcc2bbb-essentials_build\ffmpeg-2026-09-21-git-a9cbcc2bbb-essentials_build\bin\ffmpeg.exe'
DEFAULT_CAP_F = 1e-9
# ================================

NUM_RE = re.compile(r'[ep](\d+)_(\d+)', re.I)


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


def load_charge(d):
    cap = float(read_meta(d).get('cap_F', DEFAULT_CAP_F) or DEFAULT_CAP_F)
    e = pd.read_csv(d / 'electrometer.csv')
    if 'voltage_V' in e.columns:
        e['mV'] = e['voltage_V'] * 1e3
        e['Q'] = e['voltage_V'] * cap * 1e12
    else:
        e['Q'] = e['charge'] * 1e12
        e['mV'] = e['Q'] / (cap * 1e9)
    e = e[e['Q'].notna()]
    return (e['time'].to_numpy(float), e['Q'].to_numpy(float),
            e['mV'].to_numpy(float))


def load_events(d):
    p = d / 'events.csv'
    out = {}
    if p.exists():
        with open(p, newline='') as f:
            for row in csv.DictReader(f):
                try:
                    lamp = (row.get('lamp') or '?').strip().lower()
                    out[int(row['event'])] = {
                        'time': float(row['time']),
                        'source': (row.get('source') or 'trigger').strip(),
                        'lamp': lamp if lamp in LAMP_LABELS else '?'}
                except (KeyError, ValueError):
                    pass
    return out


def load_cont(d, lo, hi):
    """{name: DataFrame} for every *_cont folder, frames between lo and hi."""
    cams = {}
    for folder in sorted(p for p in d.iterdir()
                         if p.is_dir() and p.name.endswith('_cont')):
        idx = d / f'{folder.name}_frames.csv'
        if not idx.exists():
            continue
        df = pd.read_csv(idx)
        df['time'] = df['time'] + CAM_OFFSET_S.get(folder.name, 0.0)
        df = df[(df['time'] >= lo) & (df['time'] <= hi)].copy()
        if df.empty:
            continue
        df['num'] = df['filename'].str.extract(NUM_RE)[1].astype(int)
        df['path'] = [folder / f for f in df['filename']]
        cams[folder.name] = df.sort_values('time').reset_index(drop=True)
    return cams


def load_cams(d, event):
    """{name: DataFrame(time, num, filename, path)} for this event."""
    cams = {}
    for name in CAM_FOLDERS:
        idx = d / f'{name}_frames.csv'
        if not idx.exists() or not (d / name).is_dir():
            continue
        df = pd.read_csv(idx)
        df = df[df['event'] == event].copy()
        if df.empty:
            continue
        df['num'] = df['filename'].str.extract(NUM_RE)[1].astype(int)
        df['path'] = [d / name / f for f in df['filename']]
        df['time'] = df['time'] + CAM_OFFSET_S.get(name, 0.0)
        cams[name] = df.sort_values('time').reset_index(drop=True)
    return cams


# ------------------------------------------------------------- images

def read_frame(path, contrast):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.dtype == np.uint16:                  # 16-bit mono FLIR
        lo, hi = contrast
        img = np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1) * 255,
                      0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def pick_contrast(paths):
    if FLIR_CONTRAST != 'auto':
        return FLIR_CONTRAST
    vals = []
    for p in [paths[i] for i in np.linspace(0, len(paths) - 1,
                                            min(8, len(paths))).astype(int)]:
        raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if raw is not None and raw.dtype == np.uint16:
            vals.append(raw[::4, ::4].ravel())
    if not vals:
        return (0, 65535)
    lo, hi = np.percentile(np.concatenate(vals), [0.5, 99.8])
    return (float(lo), float(hi))


def label_bar(img, lines, color=(255, 255, 255), scale=0.55):
    h = int(26 * len(lines) + 8)
    cv2.rectangle(img, (0, 0), (img.shape[1], h), (0, 0, 0), -1)
    for i, txt in enumerate(lines):
        cv2.putText(img, txt, (8, 20 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 1, cv2.LINE_AA)


# ------------------------------------------------------------- charge panel

def render_charge_bg(t, y, t_first, t_last, ev, ylabel, title):
    fig = plt.figure(figsize=(OUT_W / 100, CHARGE_H / 100), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(t, y, lw=1, color='0.82')
    ax.axvspan(t_first, t_last, color='tab:cyan', alpha=0.08, lw=0)
    ax.set_xlim(t[0], t[-1])
    span = np.nanmax(y) - np.nanmin(y)
    pad = 0.08 * (span if span > 0 else 1)
    ax.set_ylim(np.nanmin(y) - pad, np.nanmax(y) + pad)
    if ev and t[0] <= ev['time'] <= t[-1]:
        c = LAMP_MPL[ev['lamp']]
        ax.plot(ev['time'], 1.03, marker='v', ms=10, color=c,
                mfc=c if ev['source'] == 'trigger' else 'white', mew=1.5,
                transform=ax.get_xaxis_transform(), clip_on=False)
    ax.set_xlabel('time since Start [s]')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, pad=14)
    ax.grid(True, alpha=0.25)
    fig.tight_layout(rect=(0, 0, 0.78, 1))     # right side for the readout
    fig.canvas.draw()
    W, H = fig.canvas.get_width_height()
    bg = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[:, :, :3],
                      cv2.COLOR_RGB2BGR).copy()
    trans, bb = ax.transData, ax.bbox
    plt.close(fig)

    def to_px(tt, yy):
        p = trans.transform(np.c_[np.atleast_1d(tt), np.atleast_1d(yy)])
        return np.c_[p[:, 0], H - p[:, 1]]
    return bg, to_px, (int(H - bb.y1), int(H - bb.y0))


# ------------------------------------------------------------- writer

class Writer:
    def __init__(self, path, w, h, fps):
        self.proc = self.cv = None
        exe = shutil.which(FFMPEG) or (FFMPEG if Path(FFMPEG).exists()
                                       else shutil.which('ffmpeg'))
        if exe:
            self.proc = subprocess.Popen(
                [exe, '-y', '-loglevel', 'error', '-f', 'rawvideo',
                 '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', f'{fps:.4f}',
                 '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                 '-crf', '18', str(path)], stdin=subprocess.PIPE)
        else:
            self.cv = cv2.VideoWriter(str(path),
                                      cv2.VideoWriter_fourcc(*'mp4v'),
                                      fps, (w, h))

    def write(self, img):
        if self.proc:
            self.proc.stdin.write(img.tobytes())
        else:
            self.cv.write(img)

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()
        else:
            self.cv.release()


# ------------------------------------------------------------- one event

def make_video(d, event, charge, events):
    ev = events.get(event)
    cams = load_cams(d, event)                 # bursts (the FLIR)
    if cams:
        master = next(n for n in MASTER_ORDER if n in cams)
        mdf = cams[master]
        stop = STOP_AT if STOP_AT is not None else int(mdf['num'].max())
        sel = mdf[(mdf['num'] >= START_AT) & (mdf['num'] <= stop)]
        sel = sel.reset_index(drop=True)
        if sel.empty:
            print(f'[event {event}] no {master} images {START_AT}..{stop} '
                  f'(has {mdf.num.min()}..{mdf.num.max()})')
            return
        t_first, t_last = sel.time.iloc[0], sel.time.iloc[-1]
        cams.update(load_cont(d, t_first - 2, t_last + 2))
    else:
        if not ev:
            print(f'[event {event}] no burst and no events.csv time - skipped')
            return
        lo = ev['time'] + NO_BURST_WINDOW[0]
        hi = ev['time'] + NO_BURST_WINDOW[1]
        cams = load_cont(d, lo, hi)
        if not cams:
            print(f'[event {event}] no FLIR burst and no continuous webcam '
                  f'frames around {ev["time"]:.1f} s - skipped')
            return
        master = sorted(cams)[0]
        mdf = sel = cams[master]
        t_first, t_last = sel.time.iloc[0], sel.time.iloc[-1]
    fps = 1.0 / np.median(np.diff(mdf.time)) if len(mdf) > 1 else 30.0
    names = [master] + [n for n in sorted(cams) if n != master]
    lamp = LAMP_LABELS[ev['lamp']] if ev else 'no events.csv entry'
    print(f'[event {event}] {lamp}  master {master} '
          f'{sel.num.iloc[0]}..{sel.num.iloc[-1]} ({len(sel)} frames, '
          f'~{fps:.0f} fps), cameras: {", ".join(names)}')

    # --- first frames -> panel widths so the row is OUT_W wide ---
    contrast = {n: pick_contrast(list(cams[n].path)) for n in names}
    first = {n: read_frame(cams[n].path.iloc[0], contrast[n]) for n in names}
    aspect = {n: (first[n].shape[1] / first[n].shape[0]
                  if first[n] is not None else 16 / 9) for n in names}
    top_h = int(OUT_W / sum(aspect.values())) // 2 * 2
    widths = [int(round(aspect[n] * top_h)) for n in names]
    widths[-1] = OUT_W - sum(widths[:-1])
    x_start = 0
    if top_h > MAX_TOP_H:                      # one camera: don't go huge
        top_h = MAX_TOP_H
        widths = [int(round(aspect[n] * top_h)) for n in names]
        x_start = (OUT_W - sum(widths)) // 2
    H = top_h + CHARGE_H

    # --- charge panel ---
    t, q, mv = charge
    w = (t >= t_first - PLOT_PAD_S) & (t <= t_last + PLOT_PAD_S)
    if w.sum() < 2:
        print(f'[event {event}] no electrometer data in the window')
        return
    tw, qw, mvw = t[w], q[w], mv[w]
    q0 = float(np.interp(t_first, tw, qw))
    y = qw - q0 if RELATIVE else qw
    title = (f'event {event}' + (f' ({ev["source"]})' if ev else '') +
             f'  -  {lamp}')
    bg, to_px, (y_top, y_bot) = render_charge_bg(
        tw, y, t_first, t_last, ev,
        'dQ from first frame [pC]' if RELATIVE else 'Q [pC]', title)
    pts = to_px(tw, y).astype(np.int32)
    lamp_col = LAMP_BGR[ev['lamp']] if ev else (0, 0, 0)

    out_dir = d / 'analysis' / 'videos'
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f'event_{event:03d}_{sel.num.iloc[0]:04d}-{sel.num.iloc[-1]:04d}'
    vw = Writer(out_dir / f'{tag}.mp4', OUT_W, H, fps * SPEED)

    times = {n: cams[n].time.to_numpy() for n in names}
    cache = {n: (None, None) for n in names}
    sync = []
    for k, row in sel.iterrows():
        tf = row.time
        frame = np.zeros((H, OUT_W, 3), np.uint8)
        x = x_start
        rec = {'master_num': row.num, 'master_t': round(tf, 6)}
        for n, pw in zip(names, widths):
            i = int(np.searchsorted(times[n], tf, side='right')) - 1
            if i < 0:
                img = np.full((top_h, pw, 3), 35, np.uint8)
                label_bar(img, [f'{n}  (no frame yet)'])
                rec.update({f'{n}_file': '', f'{n}_t': '', f'{n}_dt': ''})
            else:
                if cache[n][0] != i:
                    raw = read_frame(cams[n].path.iloc[i], contrast[n])
                    cache[n] = (i, None if raw is None else cv2.resize(
                        raw, (pw, top_h), interpolation=cv2.INTER_AREA))
                img = (cache[n][1].copy() if cache[n][1] is not None
                       else np.full((top_h, pw, 3), 60, np.uint8))
                fn, ft = cams[n].filename.iloc[i], times[n][i]
                lines = [f'{n}  {fn}', f't = {ft:.3f} s']
                if n == master:
                    lines[1] += f'   +{tf - t_first:.3f} s'
                else:
                    lines[1] += f'   ({ft - tf:+.3f} s)'
                label_bar(img, lines)
                rec.update({f'{n}_file': fn, f'{n}_t': round(ft, 6),
                            f'{n}_dt': round(ft - tf, 6)})
            frame[:top_h, x:x + pw] = img
            x += pw

        panel = bg.copy()
        m = int(np.searchsorted(tw, tf, side='right'))
        if m >= 2:
            cv2.polylines(panel, [pts[:m]], False, lamp_col, 2, cv2.LINE_AA)
        yf = float(np.interp(tf, tw, y))
        cx, cy = to_px(tf, yf)[0].astype(int)
        cv2.line(panel, (cx, y_top), (cx, y_bot), (0, 0, 220), 1, cv2.LINE_AA)
        cv2.circle(panel, (cx, cy), 5, (0, 0, 220), -1, cv2.LINE_AA)
        qf, mvf = float(np.interp(tf, tw, qw)), float(np.interp(tf, tw, mvw))
        rx = int(OUT_W * 0.79)
        for j, txt in enumerate([f't   {tf:.3f} s',
                                 f'Q   {qf:.3f} pC',
                                 f'dQ  {qf - q0:+.3f} pC',
                                 f'V   {mvf:.3f} mV',
                                 f'last sample {tw[max(m - 1, 0)] - tf:+.3f} s']):
            cv2.putText(panel, txt, (rx, 60 + 34 * j),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7 if j < 4 else 0.5,
                        (0, 0, 0), 2 if j < 4 else 1, cv2.LINE_AA)
        frame[top_h:top_h + panel.shape[0], :panel.shape[1]] = panel
        vw.write(frame)

        rec.update({'Q_pC': round(qf, 6), 'dQ_pC': round(qf - q0, 6),
                    'mV': round(mvf, 6)})
        sync.append(rec)
        if (k + 1) % 200 == 0:
            print(f'    {k + 1}/{len(sel)}')

    vw.close()
    pd.DataFrame(sync).to_csv(out_dir / f'{tag}_sync.csv', index=False)
    print(f'    -> {out_dir / (tag + ".mp4")}  '
          f'({len(sel) / (fps * SPEED):.1f} s of video, {OUT_W}x{H})')


def main():
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else SESSION
    if not (d / 'electrometer.csv').exists():
        sys.exit(f'no electrometer.csv in {d}')
    want = [int(a) for a in sys.argv[2:]] or EVENTS
    charge = load_charge(d)
    events = load_events(d)
    if want == 'all':
        nums = set(events)
        p = d / 'flir_frames.csv'
        if p.exists():
            nums |= set(pd.read_csv(p, usecols=['event'])['event'])
        want = sorted(nums)
    print(f'{d.name}: events {want}')
    for ev in want:
        make_video(d, int(ev), charge, events)


if __name__ == '__main__':
    main()