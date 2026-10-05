#!/usr/bin/env python3
"""Plot the electrometer channels from a capture.py session.

    python3 view_adc.py                                  # newest session in captures/
    python3 view_adc.py captures/session_2026-09-29_16-39-46
    python3 view_adc.py --smooth 25                      # overlay a 25-sample moving average
    python3 view_adc.py --raw                            # plot ADC counts instead of volts
    python3 view_adc.py --baseline                       # the noise floor recorded before it

Each panel's title shows the channel's mean and standard deviation (noise) in mV.
Use the matplotlib toolbar to zoom and pan.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from channels import channel_info, n_channels, read_meta

CAPTURES = Path(__file__).resolve().parent / "captures"


def newest_session():
    sessions = sorted(CAPTURES.glob("session_*"))
    if not sessions:
        sys.exit(f"No sessions in {CAPTURES}")
    return sessions[-1]


def moving_average(x, n):
    return np.convolve(x, np.ones(n) / n, mode="same")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", nargs="?", type=Path, help="session folder (default: newest)")
    ap.add_argument("--raw", action="store_true", help="plot raw ADC counts")
    ap.add_argument("--baseline", action="store_true",
                    help="plot baseline.csv (noise floor) instead of adc.csv")
    ap.add_argument("--smooth", type=int, default=0, metavar="N",
                    help="overlay an N-sample moving average")
    args = ap.parse_args()

    session = args.session or newest_session()
    csv = session / ("baseline.csv" if args.baseline else "adc.csv")
    if not csv.exists():
        sys.exit(f"No {csv.name} in {session}")
    d = np.genfromtxt(csv, delimiter=",", names=True)
    t = d["t"] if "t" in d.dtype.names else d["host_t"]  # older captures lack t

    n_ch = n_channels(d.dtype.names)
    names, colors, order = channel_info(n_ch, read_meta(session))
    fig, axes = plt.subplots(n_ch, 1, sharex=True, figsize=(11, 2 + 1.6 * n_ch),
                             layout="constrained")
    fig.suptitle(f"{session.name}{'  NOISE FLOOR' if args.baseline else ''}   ({len(t)} samples, {len(t) / (t[-1] - t[0]):.0f} S/s)")
    for ax, i in zip(axes, order):
        name, color = names[i], colors[i]
        y = d[f"raw{i}"] if args.raw else d[f"v{i}"]
        ax.plot(t, y, color=color, lw=0.7, alpha=0.5 if args.smooth else 1)
        if args.smooth > 1:
            ax.plot(t, moving_average(y, args.smooth), color=color, lw=1.5)
        ax.set_ylabel(f"{name}\n{'counts' if args.raw else 'V'}")
        ax.grid(True, alpha=0.3)
        if args.raw:
            ax.set_title(f"mean {np.nanmean(y):.1f}  std {np.nanstd(y):.2f} counts",
                         loc="left", fontsize=9)
        else:
            ax.set_title(f"mean {np.nanmean(y) * 1e3:+.2f} mV  std {np.nanstd(y) * 1e3:.2f} mV",
                         loc="left", fontsize=9)
    axes[-1].set_xlabel("time since start of recording (s)")
    plt.show()


if __name__ == "__main__":
    main()
