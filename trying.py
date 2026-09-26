"""Side-by-side video: Brio frames on the left, the charge trace drawing
itself in on the right with a cursor at each frame's timestamp.

    CLIPS = [
        ('p002_001966', 'p002_002163'),
    ]

    python frames_video.py                    # uses SESSION below
    python frames_video.py "E:/.../session_X"

Writes <session>/analysis/video_<first>-<last>.mp4 for each clip.

The plot is rendered once with matplotlib; each video frame only draws the
trace-so-far, the cursor and the readout on top with OpenCV, so a few
thousand frames take seconds, not minutes.  Encoding goes through ffmpeg
(H.264, plays anywhere) if it's found, else OpenCV's mp4v.
"""

import argparse
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

CLIPS = [
    ('p002_001550', 'p002_001560'),
]

PLOT_PAD_S = 1.0       # trace context shown before/after the clip
SPEED      = 0.1       # 1 = real time, 0.25 = quarter speed (slow-mo)
OUT_H      = 720       # video height; the Brio frame is scaled to this
PLOT_W     = 960       # width of the plot panel
RELATIVE   = True      # plot dQ from the first frame instead of absolute Q

FFMPEG = r'E:\Ben Christensen\ffmpeg-2026-09-21-git-a9cbcc2bbb-essentials_build\ffmpeg-2026-09-21-git-a9cbcc2bbb-essentials_build\bin\ffmpeg.exe'
DEFAULT_CAP_F = 1e-9
# ================================


def read_meta(d):
    meta = {}
    p = d / 'meta.txt'
    if p.exists():
        for line in open(p):
            if '\t' in line:
                k, v = line.rstrip('\n').split('\t', 1)
                meta[k] = v
    return meta


def load(d):
    cap = float(read_meta(d).get('cap_F', DEFAULT_CAP_F))
    e = pd.read_csv(d / 'electrometer.csv')
    if 'voltage_V' in e.columns:
        e['mV'] = e['voltage_V'] * 1e3
        e['Q_pC'] = e['voltage_V'] * cap * 1e12
    else:
        e['Q_pC'] = e['charge'] * 1e12
        e['mV'] = e['Q_pC'] / (cap * 1e9)
    e = e[e['Q_pC'].notna()].reset_index(drop=True)
    fr = pd.read_csv(d / 'brio_cont_frames.csv').sort_values('time')
    fr['stem'] = fr['filename'].str.replace(r'\.[^.]+$', '', regex=True)
    return e, fr.reset_index(drop=True)


def find(fr, name):
    stem = Path(name).stem
    hit = fr.index[fr['stem'] == stem]
    if len(hit) == 0:
        phase = stem.split('_')[0]
        near = fr[fr['stem'].str.startswith(phase)]['stem']
        hint = (f'{phase} has {near.iloc[0]} .. {near.iloc[-1]}'
                if len(near) else f'no frames with prefix {phase}')
        sys.exit(f'[error] frame {stem} not in brio_cont_frames.csv ({hint})')
    return int(hit[0])


class Writer:
    """ffmpeg H.264 if available, OpenCV mp4v otherwise."""

    def __init__(self, path, w, h, fps):
        self.proc = None
        self.cv = None
        exe = shutil.which(FFMPEG) or (FFMPEG if Path(FFMPEG).exists()
                                       else shutil.which('ffmpeg'))
        if exe:
            cmd = [exe, '-y', '-loglevel', 'error',
                   '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                   '-s', f'{w}x{h}', '-r', f'{fps:.3f}', '-i', '-',
                   '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20',
                   str(path)]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            print(f'[video] encoding with ffmpeg (H.264) at {fps:.1f} fps')
        else:
            self.cv = cv2.VideoWriter(str(path),
                                      cv2.VideoWriter_fourcc(*'mp4v'),
                                      fps, (w, h))
            print(f'[video] ffmpeg not found, using OpenCV mp4v at '
                  f'{fps:.1f} fps')

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


def render_background(t, y, t0, t1, ylabel, title):
    """Static plot image + a function mapping (t, y) -> pixel coords."""
    fig = plt.figure(figsize=(PLOT_W / 100, OUT_H / 100), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(t, y, lw=1, color='0.8')                 # full trace, faded
    ax.axvspan(t0, t1, color='tab:cyan', alpha=0.12)
    ax.set_xlim(t[0], t[-1])
    pad = 0.08 * (np.nanmax(y) - np.nanmin(y) or 1)
    ax.set_ylim(np.nanmin(y) - pad, np.nanmax(y) + pad)
    ax.set_xlabel('time since Start [s]')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))           # room for the readout
    fig.canvas.draw()

    W, H = fig.canvas.get_width_height()
    bg = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    bg = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR).copy()
    trans = ax.transData
    bb = ax.bbox
    plt.close(fig)

    def to_px(tt, yy):
        p = trans.transform(np.c_[np.atleast_1d(tt), np.atleast_1d(yy)])
        return np.c_[p[:, 0], H - p[:, 1]]

    y_top, y_bot = H - bb.y1, H - bb.y0
    return bg, to_px, (int(y_top), int(y_bot))


def make_clip(d, e, fr, clip, out_dir):
    ia, ib = sorted((find(fr, clip[0]), find(fr, clip[1])))
    frames = fr.iloc[ia:ib + 1].reset_index(drop=True)
    t0, t1 = frames.time.iloc[0], frames.time.iloc[-1]
    tag = f'{frames.stem.iloc[0]}-{frames.stem.iloc[-1]}'

    w = e[(e.time >= t0 - PLOT_PAD_S) & (e.time <= t1 + PLOT_PAD_S)]
    if len(w) < 2:
        print(f'[{tag}] no electrometer data in this window')
        return
    t = w.time.to_numpy()
    q = w.Q_pC.to_numpy()
    mv = w.mV.to_numpy()
    q0 = np.interp(t0, t, q)
    y = q - q0 if RELATIVE else q
    ylabel = 'dQ from first frame [pC]' if RELATIVE else 'Q [pC]'

    bg, to_px, (y_top, y_bot) = render_background(
        t, y, t0, t1, ylabel, f'{tag}   ({t1 - t0:.1f} s)')
    pts = to_px(t, y).astype(np.int32)

    # native frame rate from the frame timestamps
    native = 1.0 / np.median(np.diff(frames.time)) if len(frames) > 1 else 20
    fps = native * SPEED

    first = cv2.imread(str(d / 'brio_cont' / frames.filename.iloc[0]))
    fh, fw = (first.shape[:2] if first is not None else (720, 1280))
    img_w = int(round(fw * OUT_H / fh))
    W = img_w + bg.shape[1]
    W += W % 2
    H = OUT_H + OUT_H % 2

    out = out_dir / f'video_{tag}.mp4'
    vw = Writer(out, W, H, fps)
    lamp_col = 'lamp' in frames.columns

    for k, row in frames.iterrows():
        tf = row.time
        # --- left: Brio frame ---
        img = cv2.imread(str(d / 'brio_cont' / row.filename))
        if img is None:
            img = np.full((OUT_H, img_w, 3), 90, np.uint8)
        else:
            img = cv2.resize(img, (img_w, OUT_H), interpolation=cv2.INTER_AREA)
        label = f'{row.stem}   t={tf:.2f} s' + \
            (f'   lamp {row.lamp}' if lamp_col else '')
        cv2.rectangle(img, (0, 0), (img_w, 34), (0, 0, 0), -1)
        cv2.putText(img, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)

        # --- right: trace so far + cursor + readout ---
        panel = bg.copy()
        n = int(np.searchsorted(t, tf, side='right'))
        if n >= 2:
            cv2.polylines(panel, [pts[:n]], False, (0, 0, 0), 2, cv2.LINE_AA)
        yf = np.interp(tf, t, y)
        cx, cy = to_px(tf, yf)[0].astype(int)
        cv2.line(panel, (cx, y_top), (cx, y_bot), (0, 0, 220), 1, cv2.LINE_AA)
        cv2.circle(panel, (cx, cy), 5, (0, 0, 220), -1, cv2.LINE_AA)
        qf, mvf = np.interp(tf, t, q), np.interp(tf, t, mv)
        txt = (f'Q {qf:.3f} pC   dQ {qf - q0:+.3f} pC   V {mvf:.3f} mV')
        cv2.putText(panel, txt, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 0, 0), 2, cv2.LINE_AA)

        frame = np.zeros((H, W, 3), np.uint8)
        frame[:OUT_H, :img_w] = img
        frame[:panel.shape[0], img_w:img_w + panel.shape[1]] = panel
        vw.write(frame)

        if (k + 1) % 200 == 0:
            print(f'[{tag}] {k + 1}/{len(frames)} frames')

    vw.close()
    print(f'[{tag}] {len(frames)} frames, {len(frames) / fps:.1f} s of video '
          f'-> {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session', nargs='?', default=str(SESSION))
    args = ap.parse_args()
    d = Path(args.session)
    e, fr = load(d)
    out_dir = d / 'analysis'
    out_dir.mkdir(exist_ok=True)
    for clip in CLIPS:
        make_clip(d, e, fr, clip, out_dir)


if __name__ == '__main__':
    main()