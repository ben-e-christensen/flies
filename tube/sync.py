"""Put camera frames and ADC samples on one shared time axis.

Every sample gets a time in seconds since the start of recording (the `t`
column that capture.py writes), measured on the host's perf_counter clock.

Camera: we ask the camera for its clock (TimestampLatch) and time the round
trip. That aligns the camera clock to the host within ~0.2 ms. Each frame's
time is the middle of its exposure.
Note: the arrival time of a frame is NOT usable for this. Frames reach the
host ~8 ms after exposure starts (exposure + sensor readout + USB transfer).

ADC: the board's clock is aligned by assuming the fastest-arriving block had
~zero USB delay (good to ~1 ms). The MAX1032 samples at the start of each
conversion, a few microseconds after esp_t, so no further correction is
needed. CH1..3 are converted ~10 us after the channel before (tick_us in the
board header); far below the 200 us sample period.

The old ADS1115 board needed a 1/860 s correction (its value was a conversion
averaged over the previous ~1.2 ms); LEGACY_ADS_DELAY_S is kept for its
sessions.
"""

import time

import numpy as np

ADC_DELAY_S = 0.0              # MAX1032: samples at the start of the conversion
LEGACY_ADS_DELAY_S = 1 / 860   # old ADS1115 board @ 860 SPS: mean age of the value read

# Only used for sessions recorded before capture.py did the latch sync:
# measured arrival delay after exposure start = exposure + ~6 ms (acA1440-220um,
# full frame, USB3).
FALLBACK_PIPELINE_S = 6e-3


def cam_clock_offset(cam, n=20):
    """Return (cam_s, offset) with host_perf_counter = cam_s + offset.
    Uses the latch reading with the shortest round trip."""
    best = None
    for _ in range(n):
        a = time.perf_counter()
        cam.TimestampLatch.Execute()
        v = cam.TimestampLatchValue.GetValue() * 1e-9
        b = time.perf_counter()
        if best is None or b - a < best[0]:
            best = (b - a, v, (a + b) / 2 - v)
    return best[1], best[2]


def adc_times(esp_t, host_t, delay_s=ADC_DELAY_S):
    """Board clock (s) -> host time (s)."""
    return esp_t + np.min(host_t - esp_t) - delay_s


def frame_times(cam_t_ns, exposure_us, sync=None, host_t=None):
    """Camera TimeStamp (ns) -> host time (s) of the middle of each exposure.

    sync: [(cam_s, offset), ...] from cam_clock_offset (start and end of the
    recording; offset is interpolated between them to follow clock drift).
    Without sync, falls back to the arrival-time estimate using host_t.
    """
    cam_s = np.asarray(cam_t_ns, np.float64) * 1e-9
    half_exp = exposure_us * 1e-6 / 2
    if sync:
        xs, offs = zip(*sorted(sync))
        return cam_s + np.interp(cam_s, xs, offs) + half_exp
    arrival_offset = np.min(host_t - cam_s)
    return cam_s + arrival_offset - (exposure_us * 1e-6 + FALLBACK_PIPELINE_S) + half_exp
