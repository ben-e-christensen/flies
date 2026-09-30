"""Noise filters for the ADC traces. Saved data stays raw; these are for display
and analysis only."""

import numpy as np

MAINS_HZ = 60.0
HARMONICS = 4          # remove 60, 120, 180, 240 Hz
NOTCH_WIDTH_HZ = 0.5   # Gaussian notch sigma. Soft edges avoid the ringing a
                       # hard cutoff causes; mains drifts < 0.05 Hz so this is plenty


def notch_mains(t, y):
    """Remove mains hum (60 Hz and harmonics) from a uniformly sampled trace.

    Works in the frequency domain on the whole record, so longer records give
    a cleaner result (prepend the noise floor when you have it). Keeps the DC
    level and everything off the mains frequencies, including fast blips.
    NaNs (ADC not responding) are bridged for filtering and put back after.
    """
    y = np.asarray(y, np.float64)
    bad = ~np.isfinite(y)
    if bad.all() or len(y) < 16:
        return y.copy()
    yy = y.copy()
    if bad.any():
        yy[bad] = np.interp(t[bad], t[~bad], y[~bad])

    fs = 1.0 / np.median(np.diff(t))
    f = np.fft.rfftfreq(len(yy), 1 / fs)
    gain = np.ones_like(f)
    for k in range(1, HARMONICS + 1):
        h = MAINS_HZ * k
        if h < fs / 2:
            gain *= 1 - np.exp(-0.5 * ((f - h) / NOTCH_WIDTH_HZ) ** 2)
    out = np.fft.irfft(np.fft.rfft(yy) * gain, len(yy))
    out[bad] = np.nan
    return out
