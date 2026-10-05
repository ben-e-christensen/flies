#!/usr/bin/env python3
"""Burst capture: ADC (MAX1032, 4 channels, ~17.9 kHz each) + Basler frames
into RAM, then save to disk.

First records a noise floor: --baseline seconds (default 10) of ADC only,
camera idle. Then the capture itself (ADC + camera) starts; that's when to
drop. Nothing touches the disk until the end.

    python3 capture.py                     # 10 s baseline, then 10 s at 100 fps
    python3 capture.py -d 5 --fps 200 --exposure 2000
    python3 capture.py --baseline 0        # skip the noise floor
    python3 capture.py --rate 10000        # ADC samples/s per channel (default: board max)
    python3 capture.py --range 6           # ADC input range +/-6.144 V (default +/-12.288 V)

Output: captures/session_<timestamp>/
    baseline.csv  noise floor, same columns as adc.csv, t from -baseline to 0
    adc.csv       t,host_t,esp_t,raw0..raw3,v0..v3  (raw = 14-bit codes, 8192 = 0 V)
    frames.npy    (n_frames, H, W) uint8.  np.load(p, mmap_mode='r') to browse
    frames.csv    index,t,host_t,cam_t,block_id
    meta.txt      key<TAB>value settings + capture stats

All ADC data is saved raw (counts and unfiltered volts).

t        synced time (s since the capture started) -- use this to line up
         frames with ADC samples. Frame t = middle of its exposure. See sync.py.
host_t   when the host received it (includes USB/readout delay, ~8 ms for
         frames; ADC samples arrive in blocks, so it's the block's arrival)
esp_t / cam_t   each device's own clock

Close main.py first. Only one program can have the camera open.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from pypylon import pylon

from adc_serial import AdcReader
from channels import CHANNEL_NAMES, PLOT_ORDER
from sync import cam_clock_offset, adc_times, frame_times

# ============ CONFIG ============
DURATION_S = 10.0
BASELINE_S = 10.0       # ADC-only noise floor recorded before the capture
ADC_RATE = None         # samples/s per channel; None = the board's default (its max,
                        # ~17.9 kHz measured with 4 channels)
ADC_RANGE_V = 12        # +/- volts: 3, 6 or 12 (MAX1032 +/-3.072, 6.144, 12.288 V)
FPS = 100.0
EXPOSURE_US = None      # None = keep camera's value (clamped to fit the frame period)
GAIN_DB = 28.0          # same as basler_feed.py
OUT_ROOT = Path(__file__).resolve().parent / "captures"
# ================================


def open_camera(fps, exposure_us):
    tl = pylon.TlFactory.GetInstance()
    if not tl.EnumerateDevices():
        sys.exit("No Basler cameras found.")
    try:
        cam = pylon.InstantCamera(tl.CreateFirstDevice())
        cam.Open()
    except Exception as e:
        sys.exit(f"Could not open camera ({e}). Is main.py or pylon Viewer running?")

    cam.GainAuto.SetValue("Off")
    cam.Gain.SetValue(GAIN_DB)
    if "Mono8" in cam.PixelFormat.GetSymbolics():
        cam.PixelFormat.SetValue("Mono8")
    else:
        sys.exit(f"Camera has no Mono8 format ({cam.PixelFormat.GetSymbolics()}).")

    # Exposure must fit inside one frame period or the camera can't hit `fps`.
    cam.ExposureAuto.SetValue("Off")
    max_exp = 1e6 / fps * 0.9
    exp = exposure_us if exposure_us is not None else cam.ExposureTime.GetValue()
    if exp > max_exp:
        print(f"[!] exposure {exp:.0f} us is too long for {fps:g} fps, using {max_exp:.0f} us")
        exp = max_exp
    cam.ExposureTime.SetValue(exp)

    cam.AcquisitionFrameRateEnable.SetValue(True)
    cam.AcquisitionFrameRate.SetValue(fps)
    cam.MaxNumBuffer.SetValue(64)  # rides out short stalls on the host side

    got = cam.ResultingFrameRate.GetValue()
    if got < fps * 0.99:
        print(f"[!] camera can only do {got:.1f} fps with these settings "
              f"(exposure / resolution / USB bandwidth)")
    return cam


def noise_floor_stats(base, t_base, n_ch):
    """Quick summary of the baseline per channel (mV at the electrometer):
    overall std, and the 60 Hz component's amplitude."""
    fs = (len(t_base) - 1) / (t_base[-1] - t_base[0])
    std, hum = [], []
    for i in range(n_ch):
        v = base[:, 2 + n_ch + i] * 1e3
        v = v[np.isfinite(v)]
        std.append(f"v{i}={np.std(v):.2f}")
        spec = np.abs(np.fft.rfft(v - v.mean())) * 2 / len(v)
        f = np.fft.rfftfreq(len(v), 1 / fs)
        hum.append(f"v{i}={spec[np.abs(f - 60) < 1].max():.2f}")
    return {
        "baseline_s": f"{t_base[-1] - t_base[0]:.3f}",
        "baseline_samples": len(base),
        "noise_std_mV": " ".join(std),
        "hum_60hz_amp_mV": " ".join(hum),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--duration", type=float, default=DURATION_S, help="seconds")
    ap.add_argument("--fps", type=float, default=FPS)
    ap.add_argument("--exposure", type=float, default=EXPOSURE_US, help="microseconds")
    ap.add_argument("--baseline", type=float, default=BASELINE_S,
                    help=f"seconds of ADC-only noise floor before the capture "
                         f"(default {BASELINE_S:g}, 0 = skip)")
    ap.add_argument("--rate", type=float, default=ADC_RATE,
                    help="ADC samples/s per channel (default: as fast as the board "
                         "allows, ~17.9 kHz; it caps requests at its measured max)")
    ap.add_argument("--range", type=float, dest="range_v", default=ADC_RANGE_V,
                    help=f"ADC input range +/- volts: 3, 6 or 12 (default {ADC_RANGE_V:g}). "
                         "Wider avoids clipping, narrower gives finer steps "
                         "(1.5 mV / 750 uV / 375 uV per count)")
    ap.add_argument("--port", help="ESP32 serial port (default: auto-detect)")
    ap.add_argument("--no-camera", action="store_true", help="ADC only")
    args = ap.parse_args()

    # ---- set up ----
    cam = None
    n_frames = 0
    if not args.no_camera:
        cam = open_camera(args.fps, args.exposure)
        w, h = cam.Width.GetValue(), cam.Height.GetValue()
        exposure_us = cam.ExposureTime.GetValue()
        n_frames = int(round(args.fps * args.duration))
        print(f"Camera: {cam.GetDeviceInfo().GetModelName()} {w}x{h}, "
              f"{cam.ResultingFrameRate.GetValue():.1f} fps, "
              f"exposure {exposure_us:.0f} us")
        print(f"Allocating {n_frames * w * h / 1e9:.2f} GB for {n_frames} frames…")
        frames = np.empty((n_frames, h, w), np.uint8)
        frames.fill(0)  # touch the pages now so the OS doesn't allocate them mid-capture
        frame_meta = np.zeros((n_frames, 3), np.float64)  # host_t, cam_t, block_id

    # blocks: (host perf_counter, esp_t, raw codes, volts) as the reader
    # delivers them; host time made relative to t0 (capture start) when saving
    baseline_blocks, adc_blocks = [], []
    sink = [None]  # list the reader thread appends to; None = not recording

    def on_block(host_t, esp_t, raw, volts):
        blocks = sink[0]
        if blocks is not None:
            blocks.append((host_t, esp_t, raw, volts))

    reader = AdcReader(args.port, rate=args.rate, range_v=args.range_v)
    reader.on_block = on_block
    reader.start()
    print(f"Waiting for ADC data on {reader.port}…")
    deadline = time.time() + 6
    while reader.n_samples < reader.sample_hz * 0.2 or reader.sample_hz == 0:
        if time.time() > deadline:
            reader.stop()
            if cam:
                cam.Close()
            sys.exit(f"No ADC data ({reader.err or 'nothing received'}).")
        time.sleep(0.05)
    max_hz = (reader.cfg or {}).get("max_hz", 0)
    print(f"ADC: {reader.sample_hz:g} samples/s per channel x {reader.n_ch} "
          f"(board max {max_hz:g}), range +/-{reader.cfg['range_v']:g} V "
          f"({reader.cfg['lsb_v'] * 1e6:g} uV per count)")
    if args.rate and reader.sample_hz < args.rate * 0.99:
        print(f"[!] asked for {args.rate:g}, the board can only do {reader.sample_hz:g}")

    # ---- noise floor ----
    if args.baseline > 0:
        print(f"Noise floor: {args.baseline:g} s of ADC only. Don't drop yet…")
        sink[0] = baseline_blocks
        b_end = time.perf_counter() + args.baseline
        try:
            while (left := b_end - time.perf_counter()) > 0:
                print(f"  {left:4.1f} s", end="\r", flush=True)
                time.sleep(min(0.1, left))
        except KeyboardInterrupt:
            sink[0] = None
            reader.stop()
            if cam:
                cam.Close()
            sys.exit("\nCancelled during noise floor. Nothing saved.")
        print()

    # ---- record ----
    print(f">>> RECORDING {args.duration:g} s. Drop now. <<<")
    n_got = 0
    cam_sync = []  # (cam_s, offset) pairs; offset made relative to t0 below
    started = datetime.now()
    t0 = time.perf_counter()
    sink[0] = adc_blocks
    try:
        if cam:
            cam_sync.append(cam_clock_offset(cam))
            cam.StartGrabbingMax(n_frames, pylon.GrabStrategy_OneByOne)
            while cam.IsGrabbing():
                grab = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                try:
                    if grab.GrabSucceeded():
                        frame_meta[n_got] = (time.perf_counter() - t0,
                                             grab.TimeStamp, grab.BlockID)
                        frames[n_got] = grab.GetArray()
                        n_got += 1
                finally:
                    grab.Release()
        # ADC keeps going until the full duration (camera may finish a hair early)
        while time.perf_counter() - t0 < args.duration:
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Stopped early. Saving what was captured.")
    finally:
        sink[0] = None
        reader.stop()
        if cam:
            if cam.IsGrabbing():
                cam.StopGrabbing()
            try:
                cam_sync.append(cam_clock_offset(cam))
            except Exception as e:
                print(f"[!] end-of-capture clock sync failed ({e}), using start only")
            cam.Close()
    elapsed = time.perf_counter() - t0

    # ---- save ----
    out = OUT_ROOT / f"session_{started:%Y-%m-%d_%H-%M-%S}"
    out.mkdir(parents=True)
    print(f"Saving to {out} …")

    n_ch = reader.n_ch

    def to_table(blocks):
        """Blocks -> (n, 2 + 2*n_ch) array: host_t, esp_t, raw..., v..."""
        if not blocks:
            return np.empty((0, 2 + 2 * n_ch))
        return np.vstack([np.column_stack([np.full(len(e), h - t0), e, r, v])
                          for h, e, r, v in blocks])

    adc, base = to_table(adc_blocks), to_table(baseline_blocks)
    # one board->host clock alignment for both (same clock, more samples)
    both = np.vstack([base, adc])
    t_both = adc_times(both[:, 1], both[:, 0]) if len(both) else np.empty(0)
    t_base, t_adc = t_both[:len(base)], t_both[len(base):]
    adc_header = ",".join(["t", "host_t", "esp_t"] + [f"raw{i}" for i in range(n_ch)]
                          + [f"v{i}" for i in range(n_ch)])
    adc_fmt = ["%.6f", "%.6f", "%.6f"] + ["%d"] * n_ch + ["%.6f"] * n_ch
    np.savetxt(out / "adc.csv", np.column_stack([t_adc, adc]), delimiter=",",
               comments="", header=adc_header, fmt=adc_fmt)
    if len(base):
        np.savetxt(out / "baseline.csv", np.column_stack([t_base, base]), delimiter=",",
                   comments="", header=adc_header, fmt=adc_fmt)

    stats = {
        "wall_clock_start": started.isoformat(timespec="milliseconds"),
        "duration_s": f"{elapsed:.3f}",
        "adc_port": reader.port,
        "adc": "MAX1032 on XIAO ESP32-C6",
        "adc_cfg": " ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}"
                            for k, v in (reader.cfg or {}).items()),
        "adc_samples": len(adc),
        "adc_rate_hz": f"{len(adc) / elapsed:.1f}",
        "channel_names": "|".join(CHANNEL_NAMES),
        "plot_order": ",".join(str(i) for i in PLOT_ORDER),
        "adc_gaps_total": f"{reader.n_gaps} ({reader.n_missing} samples)",
    }
    if len(adc) > 1:
        gaps = np.diff(adc[:, 1])
        stats["adc_max_gap_ms"] = f"{gaps.max() * 1e3:.2f}"
    if len(base) > 100:
        stats.update(noise_floor_stats(base, t_base, n_ch))

    if cam:
        fm = frame_meta[:n_got]
        sync_rel = [(c, off - t0) for c, off in cam_sync]
        t_frame = frame_times(fm[:, 1], exposure_us, sync_rel)
        np.save(out / "frames.npy", frames[:n_got])
        np.savetxt(out / "frames.csv",
                   np.column_stack([np.arange(n_got), t_frame, fm[:, 0], fm[:, 1], fm[:, 2]]),
                   delimiter=",", comments="", header="index,t,host_t,cam_t,block_id",
                   fmt=["%d", "%.6f", "%.6f", "%d", "%d"])
        dropped = int(fm[-1, 2] - fm[0, 2] + 1 - n_got) if n_got else 0
        stats.update({
            "camera": f"{w}x{h} Mono8",
            "fps_set": args.fps,
            "exposure_us": f"{exposure_us:.0f}",
            "gain_db": GAIN_DB,
            "frames": n_got,
            "frames_dropped": dropped,
            # camera TimeStamp is in ns on USB3 ace cameras
            "fps_measured": f"{(n_got - 1) / ((fm[-1, 1] - fm[0, 1]) / 1e9):.2f}"
                            if n_got > 1 else "",
            # host_t = cam_s + offset, at the start and end of recording
            "cam_sync": " ".join(f"{c:.6f}:{off:.6f}" for c, off in sync_rel),
        })
        if n_got:
            stats["frame_arrival_delay_ms"] = f"{np.min(fm[:, 0] - t_frame) * 1e3:.2f}"
    with open(out / "meta.txt", "w") as f:
        for k, v in stats.items():
            f.write(f"{k}\t{v}\n")

    print("\n".join(f"  {k}: {v}" for k, v in stats.items()))
    print("Done.")


if __name__ == "__main__":
    main()
