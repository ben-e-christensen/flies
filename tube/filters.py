"""Noise filters for the ADC traces. Saved data stays raw; these are for display
and analysis only."""

import numpy as np

MAINS_HZ = 60.0
HARMONICS = 10      # fit 60, 120, ... 600 Hz (lights and supplies make harmonics)
WINDOW_S = 0.5      # hum amplitude/phase is re-fitted every half window; a blip
                    # of a few ms barely moves a fit over this long


def mains_frequency(t, y, lo=59.8, hi=60.2):
    """Find the actual mains frequency in a trace (it drifts ~0.02 Hz) by
    scanning for the best single-sine least-squares fit."""
    t = t - t[0]
    # coarse then fine scan; every 4th sample is plenty for this
    tt, yy = t[::4], y[::4] - np.mean(y[::4])
    best, best_power = None, -1.0
    for step, a, b in [(0.01, lo, hi), (0.0005, None, None)]:
        if a is None:
            a, b = best - 0.01, best + 0.01
        for f in np.arange(a, b + step / 2, step):
            w = 2 * np.pi * f * tt
            m = np.column_stack([np.cos(w), np.sin(w)])
            coef, *_ = np.linalg.lstsq(m, yy, rcond=None)
            power = coef @ coef
            if power > best_power:
                best, best_power = f, power
    return best


def remove_mains(t, y, f0=None):
    """Subtract mains hum (f0 and harmonics up to HARMONICS x f0) from a trace
    sampled at times t (seconds). Returns the cleaned trace.

    The hum is fitted by least squares in overlapping WINDOW_S windows (Hann
    cross-faded), using each sample's real time, so dropped packets and uneven
    spacing don't matter. Each window also fits an offset and slope so slow
    drifts or steps don't leak into the hum fit; only the hum is subtracted, so
    the DC level, steps and fast blips are kept. NaNs stay NaN.
    """
    t = np.asarray(t, np.float64)
    y = np.asarray(y, np.float64)
    ok = np.isfinite(y)
    if ok.sum() < 100:
        return y.copy()
    if f0 is None:
        f0 = mains_frequency(t[ok], y[ok])
    nyq = 0.5 / np.median(np.diff(t))
    harmonics = [k for k in range(1, HARMONICS + 1) if k * f0 < nyq * 0.95]

    hum = np.zeros_like(y)
    weight = np.zeros_like(y)
    half = min(WINDOW_S, t[-1] - t[0]) / 2
    # Windows sit fully inside the data (50% overlap); the first and last ones
    # cover the ends. A window hanging off the end would hold too few samples
    # for a sane fit.
    centers = np.append(np.arange(t[0] + half, t[-1] - half, half), t[-1] - half)
    for c in centers:
        eps = 1e-9  # so float rounding never drops the first/last sample
        i0, i1 = np.searchsorted(t, c - half - eps), np.searchsorted(t, c + half + eps, "right")
        sel = np.arange(i0, i1)[ok[i0:i1]]
        if len(sel) < 4 * len(harmonics) + 10:
            continue
        ts = t[sel] - c
        cols = [np.ones_like(ts), ts]
        for k in harmonics:
            w = 2 * np.pi * k * f0 * t[sel]
            cols += [np.cos(w), np.sin(w)]
        m = np.column_stack(cols)
        coef, *_ = np.linalg.lstsq(m, y[sel], rcond=None)
        # Hann (sums to 1 at 50% overlap), floored so the outermost samples,
        # covered by one window only, still get that window's fit
        win = 0.5 * (1 + np.cos(np.pi * ts / half)) + 1e-3
        hum[sel] += win * (m[:, 2:] @ coef[2:])
        weight[sel] += win
    good = weight > 1e-6
    out = y.copy()
    out[good] -= hum[good] / weight[good]
    return out
