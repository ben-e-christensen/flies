#!/usr/bin/env python3
"""Turn a capture.py session into two MP4s laid out like main.py: camera on the
left, the 3 ADC traces on the right covering the whole clip, with a red cursor
at the current frame's time. One video shows the raw traces, the other the
same traces with 60 Hz mains hum (and harmonics) removed (see filters.py).

    python3 make_video.py                      # all frames, newest session, 0.1x speed
    python3 make_video.py 200 400              # frames 200..400 (inclusive)
    python3 make_video.py 200 400 --speed 0.05
    python3 make_video.py --session captures/session_2026-09-29_17-20-34

--speed   playback speed vs real time: 0.1 plays a 200 fps capture at 20 fps

Output goes in the session folder:
    video_f<start>-<end>_x<speed>_raw.mp4
    video_f<start>-<end>_x<speed>_filtered.mp4
The filter runs on the whole recording (noise floor included when the session
has one), not just the clip, so the clip edges don't distort it.
Timing: see sync.py. Captures made before the sync update get an approximate
(~1 ms) alignment and a warning.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scope_panel import CHANNEL_NAMES, COLORS, PLOT_ORDER
from sync import adc_times, frame_times
from filters import notch_mains

CAPTURES = Path(__file__).resolve().parent / "captures"

# ============ CONFIG ============
PANEL_H = 720         # output height; camera is scaled to this
PLOT_W = 960          # width of the ADC panel (right half)
DEFAULT_SPEED = 0.1
CRF = 18              # x264 quality (lower = better/bigger)
# ================================


def newest_session():
    sessions = sorted(CAPTURES.glob("session_*"))
    if not sessions:
        sys.exit(f"No sessions in {CAPTURES}")
    return sessions[-1]


def read_meta(session):
    meta = {}
    p = session / "meta.txt"
    if p.exists():
        for line in p.read_text().splitlines():
            k, _, v = line.partition("\t")
            meta[k] = v
    return meta


def load_times(session, fmeta, adc):
    """Synced frame and ADC times. Uses the `t` columns when capture.py wrote them."""
    if "t" in fmeta.dtype.names and "t" in adc.dtype.names:
        return fmeta["t"], adc["t"]
    print("[!] capture predates clock sync: using approximate alignment (~1 ms)")
    exposure_us = float(read_meta(session).get("exposure_us", 3000))
    return (frame_times(fmeta["cam_t"], exposure_us, host_t=fmeta["host_t"]),
            adc_times(adc["esp_t"], adc["host_t"]))


def filtered_traces(session, t_adc, v):
    """60 Hz-notched copy of v. Runs over noise floor + capture when available."""
    base_csv = session / "baseline.csv"
    if base_csv.exists():
        b = np.genfromtxt(base_csv, delimiter=",", names=True)
        if len(b) and "t" in b.dtype.names:
            n_b = len(b)
            t_all = np.concatenate([b["t"], t_adc])
            return [notch_mains(t_all, np.concatenate([b[f"v{i}"], v[i]]))[n_b:]
                    for i in range(3)]
    return [notch_mains(t_adc, y) for y in v]


def render_plot_panel(t_adc, v, t0, t1, height):
    """Draw the ADC traces for [t0, t1] once. Returns the BGR image, a function
    mapping time -> x pixel, and each axes' (top, bottom) pixel rows."""
    dpi = 100
    fig, axes = plt.subplots(3, 1, sharex=True, dpi=dpi, figsize=(PLOT_W / dpi, height / dpi))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.97, bottom=0.08, hspace=0.12)
    sel = (t_adc >= t0) & (t_adc <= t1)
    few = np.count_nonzero(sel) < 60
    for ax, i in zip(axes, PLOT_ORDER):
        ax.plot(t_adc[sel], v[i][sel] * 1e3, color=COLORS[i], lw=1, marker="." if few else None)
        ax.set_ylabel(f"{CHANNEL_NAMES[i].split(' (')[0]}\nmV", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
    axes[0].set_xlim(t0, t1)
    axes[-1].set_xlabel("time since start of recording (s)", fontsize=8)

    fig.canvas.draw()
    img = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[..., :3], cv2.COLOR_RGB2BGR)
    h = img.shape[0]
    spans = []
    for ax in axes:
        bb = ax.get_window_extent()
        spans.append((int(h - bb.y1), int(h - bb.y0)))
    tr = axes[0].transData
    x_of = lambda t: int(round(tr.transform((t, 0))[0]))
    plt.close(fig)
    return img, x_of, spans


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("start", nargs="?", type=int, help="first frame (default: 0)")
    ap.add_argument("end", nargs="?", type=int, help="last frame, inclusive (default: last)")
    ap.add_argument("--speed", type=float, default=DEFAULT_SPEED,
                    help=f"playback speed vs real time (default {DEFAULT_SPEED})")
    ap.add_argument("--session", type=Path, help="session folder (default: newest)")
    args = ap.parse_args()
    if (args.start is None) != (args.end is None):
        ap.error("give both start and end frames, or neither")

    session = args.session or newest_session()
    frames = np.load(session / "frames.npy", mmap_mode="r")
    fmeta = np.genfromtxt(session / "frames.csv", delimiter=",", names=True)
    adc = np.genfromtxt(session / "adc.csv", delimiter=",", names=True)
    n = len(frames)

    start = 0 if args.start is None else args.start
    end = n - 1 if args.end is None else args.end
    if not 0 <= start <= end < n:
        sys.exit(f"Frames must satisfy 0 <= start <= end <= {n - 1} (got {start}, {end})")

    t_frame, t_adc = load_times(session, fmeta, adc)
    v = [adc[f"v{i}"] for i in range(3)]
    cam_fps = (n - 1) / (t_frame[-1] - t_frame[0])
    out_fps = cam_fps * args.speed

    t0, t1 = t_frame[start], t_frame[end]
    pad = max((t1 - t0) * 0.02, 0.005)
    t0, t1 = t0 - pad, t1 + pad

    # ---- layout: camera left, plots right ----
    fh, fw = frames.shape[1:]
    cam_w = int(round(fw * PANEL_H / fh)) // 2 * 2
    out_w, out_h = cam_w + PLOT_W, PANEL_H
    stem = f"video_f{start:04d}-{end:04d}_x{args.speed:g}"
    versions = []  # (tag shown on video, traces, plot image, ffmpeg process, path)
    for tag, traces in [("RAW", v), ("60 Hz FILTERED", filtered_traces(session, t_adc, v))]:
        plot_bg, x_of, spans = render_plot_panel(t_adc, traces, t0, t1, PANEL_H)
        path = session / f"{stem}_{'raw' if tag == 'RAW' else 'filtered'}.mp4"
        ff = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}", "-r", f"{out_fps}",
             "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(CRF), str(path)],
            stdin=subprocess.PIPE)
        versions.append((tag, traces, plot_bg, ff, path))

    print(f"{session.name}: frames {start}-{end} ({end - start + 1}), "
          f"captured at {cam_fps:.1f} fps -> {out_fps:.2f} fps video ({args.speed:g}x)")

    font = cv2.FONT_HERSHEY_SIMPLEX
    canvas = np.empty((out_h, out_w, 3), np.uint8)
    for k, i in enumerate(range(start, end + 1)):
        t = t_frame[i]
        cam = cv2.resize(np.asarray(frames[i]), (cam_w, out_h), interpolation=cv2.INTER_AREA)
        x = cam_w + x_of(t)  # same axes layout in both versions
        for tag, traces, plot_bg, ff, _ in versions:
            canvas[:, :cam_w] = cam[..., None]
            canvas[:, cam_w:] = plot_bg[:out_h, :PLOT_W]

            label = f"frame {i}   t = {t:.4f} s   {args.speed:g}x   {tag}"
            cv2.putText(canvas, label, (10, 28), font, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(canvas, label, (10, 28), font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

            for (top, bot), y in zip(spans, [traces[j] for j in PLOT_ORDER]):
                cv2.line(canvas, (x, top), (x, bot), (0, 0, 220), 1, cv2.LINE_AA)
                val = np.interp(t, t_adc, y) * 1e3
                cv2.rectangle(canvas, (out_w - 116, top + 2), (out_w - 22, top + 22),
                              (255, 255, 255), -1)
                cv2.putText(canvas, f"{val:+.1f} mV", (out_w - 110, top + 16),
                            font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

            ff.stdin.write(canvas.tobytes())
        if k % 100 == 0:
            print(f"  {k}/{end - start + 1}", end="\r", flush=True)

    for _, _, _, ff, path in versions:
        ff.stdin.close()
        if ff.wait() != 0:
            sys.exit(f"ffmpeg failed on {path.name}")
        print(f"Wrote {path} ({(end - start + 1) / out_fps:.1f} s long)")


if __name__ == "__main__":
    main()
