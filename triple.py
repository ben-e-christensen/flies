"""Triple video: FLIR burst | Brio burst | charge trace, side by side.

For sessions that saved blip BURSTS from both cameras:
    flir/e001_0000.png ...   + flir_frames.csv  (time,event,filename)
    brio/e001_0000.jpg ...   + brio_frames.csv  (time,event,filename)
    electrometer.csv

HOW TIME IS HANDLED
    The FLIR is the master clock: one video frame per FLIR frame.
    For each FLIR frame at time t:
      - Brio panel shows the most recent Brio frame with time <= t
        (+ BRIO_OFFSET_S).  Never a frame from the future.  Its label shows
        how old it is relative to the FLIR frame, e.g. "-0.021 s".
      - Charge panel draws the trace up to t, cursor at t.
    All three come from the same clock (seconds since Start).  The camera
    timestamps are when each frame ARRIVED in Python, so the Brio lags the
    real moment by USB/driver latency (tens of ms).  If you measure that lag
    (e.g. a lamp switching on in both), put it in BRIO_OFFSET_S.

    STOP_AT is the FLIR image number in the filename: 280 -> e001_0280 is
    the last frame in the video.

    python triple_video.py                     # uses SESSION below
    python triple_video.py "E:/.../session_X"

Writes to <session>/analysis/:
    triple_e001_0000-0280.mp4
    triple_e001_0000-0280_sync.csv   one row per video frame: which FLIR
                                     frame, which Brio frame, both times,
                                     the offset between them, and Q
"""

import argparse
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

EVENT    = 1          # which burst (e001 -> 1)
START_AT = 0          # first FLIR image number
STOP_AT  = 280        # last FLIR image number (None = end of the burst)

SPEED    = 0.25        # 1 = real time (FLIR fps), 0.25 = quarter speed
PANEL_H  = 540        # height of every panel
PLOT_W   = 900        # width of the charge panel
PLOT_PAD_S = 0.5      # trace context before the first / after the last frame
RELATIVE = True       # plot dQ from the first FLIR frame instead of Q

BRIO_OFFSET_S = 0.0   # added to Brio timestamps (negative = Brio frames
                      # happened earlier than they were stamped)

FLIR_CONTRAST = 'auto'   # 'auto' = fixed stretch from sampled frames,
                         # or (lo, hi) in raw 16-bit counts

FFMPEG = r'E:\Ben Christensen\ffmpeg-2026-09-21-git-a9cbcc2bbb-essentials_build\ffmpeg-2026-09-21-git-a9cbcc2bbb-essentials_build\bin\ffmpeg.exe'
DEFAULT_CAP_F = 1e-9
# ================================

NUM_RE = re.compile(r'e(\d+)_(\d+)', re.I)


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


def load_index(d, name, event):
    p = d / f'{name}_frames.csv'
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df = df[df['event'] == event].copy()
    nums = df['filename'].str.extract(NUM_RE)
    df['num'] = nums[1].astype(int)
    df['path'] = [d / name / f for f in df['filename']]
    return df.sort_values('time').reset_index(drop=True)


def load_charge(d):
    cap = float(read_meta(d).get('cap_F', DEFAULT_CAP_F))
    e = pd.read_csv(d / 'electrometer.csv')
    if 'voltage_V' in e.columns:
        e['mV'] = e['voltage_V'] * 1e3
        e['Q_pC'] = e['voltage_V'] * cap * 1e12
    else:                                   # charge-mode session
        e['Q_pC'] = e['charge'] * 1e12
        e['mV'] = e['Q_pC'] / (cap * 1e9)
    return e[e['Q_pC'].notna()].reset_index(drop=True)


def event_time(d, event):
    p = d / 'events.csv'
    if not p.exists():
        return None
    ev = pd.read_csv(p)
    hit = ev[ev['event'] == event]
    return float(hit['time'].iloc[0]) if len(hit) else None


# ------------------------------------------------------------- drawing

def label_bar(img, lines, color=(255, 255, 255)):
    """Black bar across the top with one or two lines of text."""
    h = 30 * len(lines) + 6
    cv2.rectangle(img, (0, 0), (img.shape[1], h), (0, 0, 0), -1)
    for i, txt in enumerate(lines):
        cv2.putText(img, txt, (10, 24 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, color, 2, cv2.LINE_AA)


def fit_h(img, h):
    w = int(round(img.shape[1] * h / img.shape[0]))
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def flir_to_bgr(raw, lo, hi):
    x = (raw.astype(np.float32) - lo) / max(hi - lo, 1)
    x = np.clip(x * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)


def pick_contrast(paths):
    if FLIR_CONTRAST != 'auto':
        return FLIR_CONTRAST
    sample = [paths[i] for i in np.linspace(0, len(paths) - 1,
                                            min(8, len(paths))).astype(int)]
    vals = []
    for p in sample:
        raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if raw is not None:
            vals.append(raw[::4, ::4].ravel())
    if not vals:
        return 0, 65535
    v = np.concatenate(vals)
    lo, hi = np.percentile(v, [0.5, 99.8])
    return float(lo), float(hi)


def render_plot_bg(t, y, t_first, t_last, t_event, ylabel, title):
    fig = plt.figure(figsize=(PLOT_W / 100, PANEL_H / 100), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(t, y, lw=1, color='0.8')
    ax.axvspan(t_first, t_last, color='tab:cyan', alpha=0.12)
    if t_event is not None and t[0] <= t_event <= t[-1]:
        ax.axvline(t_event, color='tab:orange', lw=1, ls='--')
        ax.text(t_event, 1.0, ' trigger', color='tab:orange', fontsize=8,
                transform=ax.get_xaxis_transform(), va='top')
    ax.set_xlim(t[0], t[-1])
    span = np.nanmax(y) - np.nanmin(y)
    pad = 0.08 * (span if span > 0 else 1)
    ax.set_ylim(np.nanmin(y) - pad, np.nanmax(y) + pad)
    ax.set_xlabel('time since Start [s]')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88))       # room for the label bar
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


class Writer:
    def __init__(self, path, w, h, fps):
        self.proc = self.cv = None
        exe = shutil.which(FFMPEG) or (FFMPEG if Path(FFMPEG).exists()
                                       else shutil.which('ffmpeg'))
        if exe:
            self.proc = subprocess.Popen(
                [exe, '-y', '-loglevel', 'error',
                 '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                 '-s', f'{w}x{h}', '-r', f'{fps:.4f}', '-i', '-',
                 '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
                 str(path)], stdin=subprocess.PIPE)
            print(f'[video] ffmpeg H.264, {w}x{h} at {fps:.2f} fps')
        else:
            self.cv = cv2.VideoWriter(str(path),
                                      cv2.VideoWriter_fourcc(*'mp4v'),
                                      fps, (w, h))
            print(f'[video] OpenCV mp4v, {w}x{h} at {fps:.2f} fps')

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


# ------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session', nargs='?', default=str(SESSION))
    args = ap.parse_args()
    d = Path(args.session)

    flir = load_index(d, 'flir', EVENT)
    if flir is None or flir.empty:
        sys.exit(f'[error] no FLIR frames for event {EVENT} in {d}')
    brio = load_index(d, 'brio', EVENT)
    if brio is None or brio.empty:
        print(f'[warn] no Brio burst for event {EVENT} - panel stays blank')
        brio = None
    e = load_charge(d)
    t_event = event_time(d, EVENT)

    # --- FLIR range by image number ---
    stop = STOP_AT if STOP_AT is not None else int(flir['num'].max())
    sel = flir[(flir['num'] >= START_AT) & (flir['num'] <= stop)]
    sel = sel.reset_index(drop=True)
    if sel.empty:
        sys.exit(f'[error] no FLIR images numbered {START_AT}..{stop} '
                 f'(event {EVENT} has {flir["num"].min()}..'
                 f'{flir["num"].max()})')
    if STOP_AT is not None and STOP_AT not in set(flir['num']):
        print(f'[warn] FLIR image {STOP_AT} does not exist, stopping at '
              f'{sel["num"].iloc[-1]}')
    t_first, t_last = sel['time'].iloc[0], sel['time'].iloc[-1]
    tag = (f'e{EVENT:03d}_{sel["num"].iloc[0]:04d}-'
           f'{sel["num"].iloc[-1]:04d}')

    flir_fps = 1.0 / np.median(np.diff(flir['time']))
    print(f'[flir] event {EVENT}: images {flir["num"].min()}..'
          f'{flir["num"].max()}, ~{flir_fps:.1f} fps; using '
          f'{sel["num"].iloc[0]}..{sel["num"].iloc[-1]} '
          f'({t_first:.3f} -> {t_last:.3f} s, {len(sel)} frames)')
    if brio is not None:
        bt = brio['time'].to_numpy() + BRIO_OFFSET_S
        brio_fps = 1.0 / np.median(np.diff(bt))
        print(f'[brio] event {EVENT}: {len(brio)} frames, ~{brio_fps:.1f} '
              f'fps, {bt[0]:.3f} -> {bt[-1]:.3f} s'
              + (f'  (offset {BRIO_OFFSET_S:+.3f} s applied)'
                 if BRIO_OFFSET_S else ''))
    if t_event is not None:
        print(f'[event] trigger at {t_event:.3f} s '
              f'(FLIR image ~{sel.iloc[np.argmin(abs(sel.time - t_event))].num}'
              f' in this range)' if t_first <= t_event <= t_last
              else f'[event] trigger at {t_event:.3f} s (outside this range)')

    # --- charge panel background ---
    w = e[(e['time'] >= t_first - PLOT_PAD_S) &
          (e['time'] <= t_last + PLOT_PAD_S)]
    if len(w) < 2:
        sys.exit('[error] no electrometer samples over this range')
    t = w['time'].to_numpy()
    q = w['Q_pC'].to_numpy()
    mv = w['mV'].to_numpy()
    q0 = float(np.interp(t_first, t, q))
    y = q - q0 if RELATIVE else q
    bg, to_px, (y_top, y_bot) = render_plot_bg(
        t, y, t_first, t_last, t_event,
        'dQ from first FLIR frame [pC]' if RELATIVE else 'Q [pC]',
        f'charge   ({len(w)} samples, '
        f'~{1 / np.median(np.diff(t)):.0f} Hz)')
    pts = to_px(t, y).astype(np.int32)

    # --- panel sizes ---
    lo, hi = pick_contrast(list(sel['path']))
    print(f'[flir] contrast {lo:.0f}..{hi:.0f} counts')
    first_flir = cv2.imread(str(sel['path'].iloc[0]), cv2.IMREAD_UNCHANGED)
    fw = int(round(first_flir.shape[1] * PANEL_H / first_flir.shape[0]))
    if brio is not None:
        first_brio = cv2.imread(str(brio['path'].iloc[0]))
        bw = int(round(first_brio.shape[1] * PANEL_H / first_brio.shape[0]))
    else:
        bw = int(PANEL_H * 16 / 9)
    W = fw + bw + bg.shape[1]
    W += W % 2
    H = PANEL_H + PANEL_H % 2

    out_dir = d / 'analysis'
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f'triple_{tag}.mp4'
    vw = Writer(out, W, H, flir_fps * SPEED)

    sync = []
    brio_cache = (None, None)       # (index, image) - reuse while it holds

    for k, row in sel.iterrows():
        tf = row['time']

        # --- FLIR ---
        raw = cv2.imread(str(row['path']), cv2.IMREAD_UNCHANGED)
        f_img = (flir_to_bgr(raw, lo, hi) if raw is not None
                 else np.full((PANEL_H, fw, 3), 60, np.uint8))
        f_img = cv2.resize(f_img, (fw, PANEL_H), interpolation=cv2.INTER_AREA)
        label_bar(f_img, [f'FLIR  {row["filename"]}',
                          f't = {tf:.3f} s   clip +{tf - t_first:.3f} s'])

        # --- Brio: latest frame at or before tf ---
        b_name, b_t, b_dt = '', np.nan, np.nan
        if brio is not None:
            bi = int(np.searchsorted(bt, tf, side='right')) - 1
            if bi >= 0:
                if brio_cache[0] != bi:
                    img = cv2.imread(str(brio['path'].iloc[bi]))
                    brio_cache = (bi, None if img is None
                                  else cv2.resize(img, (bw, PANEL_H),
                                                  interpolation=cv2.INTER_AREA))
                b_img = (brio_cache[1].copy() if brio_cache[1] is not None
                         else np.full((PANEL_H, bw, 3), 60, np.uint8))
                b_name = brio['filename'].iloc[bi]
                b_t = bt[bi]
                b_dt = b_t - tf
                label_bar(b_img, [f'Brio  {b_name}',
                                  f't = {b_t:.3f} s   ({b_dt:+.3f} s vs FLIR)'])
            else:
                b_img = np.full((PANEL_H, bw, 3), 40, np.uint8)
                label_bar(b_img, ['Brio  (no frame yet)',
                                  f'first at {bt[0]:.3f} s'])
        else:
            b_img = np.full((PANEL_H, bw, 3), 40, np.uint8)
            label_bar(b_img, ['Brio  (no burst for this event)'])

        # --- charge ---
        panel = bg.copy()
        n = int(np.searchsorted(t, tf, side='right'))
        if n >= 2:
            cv2.polylines(panel, [pts[:n]], False, (0, 0, 0), 2, cv2.LINE_AA)
        yf = float(np.interp(tf, t, y))
        cx, cy = to_px(tf, yf)[0].astype(int)
        cv2.line(panel, (cx, y_top), (cx, y_bot), (0, 0, 220), 1, cv2.LINE_AA)
        cv2.circle(panel, (cx, cy), 5, (0, 0, 220), -1, cv2.LINE_AA)
        qf = float(np.interp(tf, t, q))
        mvf = float(np.interp(tf, t, mv))
        t_sample = t[max(n - 1, 0)]
        label_bar(panel, [f'Q {qf:.3f} pC   dQ {qf - q0:+.3f} pC   '
                          f'V {mvf:.3f} mV',
                          f't = {tf:.3f} s   last sample {t_sample - tf:+.3f} s'])

        frame = np.zeros((H, W, 3), np.uint8)
        frame[:PANEL_H, :fw] = f_img
        frame[:PANEL_H, fw:fw + bw] = b_img
        frame[:panel.shape[0], fw + bw:fw + bw + panel.shape[1]] = panel
        vw.write(frame)

        sync.append({'flir_num': row['num'], 'flir_file': row['filename'],
                     'flir_t': round(tf, 6),
                     'brio_file': b_name,
                     'brio_t': round(b_t, 6) if b_name else '',
                     'brio_minus_flir_s': round(b_dt, 6) if b_name else '',
                     'Q_pC': round(qf, 6), 'dQ_pC': round(qf - q0, 6),
                     'mV': round(mvf, 6)})
        if (k + 1) % 100 == 0:
            print(f'[video] {k + 1}/{len(sel)}')

    vw.close()
    pd.DataFrame(sync).to_csv(out_dir / f'triple_{tag}_sync.csv', index=False)
    print(f'[done] {len(sel)} frames, {len(sel) / (flir_fps * SPEED):.1f} s '
          f'of video -> {out}')
    print(f'[done] sync table -> {out_dir / f"triple_{tag}_sync.csv"}')


if __name__ == '__main__':
    main()