"""Put camera frames and ADC samples on one shared time axis.

Every sample gets a time in seconds since the start of recording (the `t`
column that capture.py writes), measured on the host's perf_counter clock.

Camera: we ask the camera for its clock (TimestampLatch) and time the round
trip. That aligns the camera clock to the host within ~0.2 ms. Each frame's
time is the middle of its exposure.
Note: the arrival time of a frame is NOT usable for this. Frames reach the
host ~8 ms after exposure starts (exposure + sensor readout + USB transfer).

ADC: the ESP's clock is aligned by assuming the fastest-arriving line had
~zero USB delay (good to ~1 ms), then shifted back ADC_DELAY_S because the
value read at esp_t is a conversion that averaged over the previous ~1.2 ms.
"""

import time

import numpy as np

ADC_DELAY_S = 1 / 860  # ADS1115 continuous @ 860 SPS: mean age of the value read

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


def adc_times(esp_t, host_t):
    """ESP clock (s) -> host time (s)."""
    return esp_t + np.min(host_t - esp_t) - ADC_DELAY_S


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
