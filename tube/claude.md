# Tube rig: charged-ball drop past induction rings, with Basler video

State as of 2026-10-02. This file is meant to stand on its own: a reader who
can't see the code should still be able to follow the setup, the data and the
findings so far. The original Basler notes are kept in the appendix at the end.

**Status:** the ADC was switched on 2026-10-02 from 3× ADS1115 (500 SPS used,
860 max) to a single MAX1032: 4 channels at about **17,800 samples/s per
channel**, measured on the real board. Streaming, timing and saving have been
verified on the hardware. At that first test the inputs looked unconnected
(see Next steps), so there's no real electrometer data from the new board yet.
All experimental findings below come from the old ADS1115 board.

## The experiment

A charged ball is dropped down a vertical clear tube. The tube has sensor rings
(top, middle, bottom). Each ring feeds an electrometer, so a charge passing
through a ring should show up as a blip on that channel. A high-speed Basler
camera films the tube so the blips can be matched to the ball's position.

## Hardware

- **DAQ board:** Seeed XIAO ESP32-C6 plus a MAX1032 (14-bit SAR ADC, SPI,
  internal 4.096 V reference, 115 ksps total across channels). Powered from
  USB 5 V straight to DVDD (no protection diode on the XIAO), ferrite-filtered
  5 V to AVDD, XIAO 3V3 to DVDDO for 3.3 V logic. USB-Serial/JTAG at
  `/dev/ttyACM0` (USB VID 0x303A).
- **SPI pins:** D8/GPIO19 SCLK, D9/GPIO20 DOUT (MISO), D10/GPIO18 DIN (MOSI),
  D0/GPIO0 CS (10 k pull-up). SSTRB on D1 is unused.
- **Inputs:** four electrometer outputs arrive on a DB9 from a separate
  electrometer board. MAX1032 CH0–CH3 = Electro1–4, single-ended, ±12.288 V
  range by default (selectable, see below). CH4–CH7 are grounded.

  | MAX1032 | electrometer | ring | data column |
  |---|---|---|---|
  | CH0 | Electro1 | **top** | `raw0` / `v0` |
  | CH1 | Electro2 | none: **not connected, floating** | `raw1` / `v1` |
  | CH2 | Electro3 | **middle** | `raw2` / `v2` |
  | CH3 | Electro4 | **bottom** | `raw3` / `v3` |

  Names, colors and plot order (top, middle, bottom, then the floating
  channel) are set in `channels.py`.
- **Gotchas from the board's bring-up:** an unplugged input reads a steady
  +2.2 V, not 0 (the input is a ~17 kΩ network biased internally), so a loose
  DB9 shows up as a fake +2.2 V. Input resistance ~17 kΩ, so sources must be
  low impedance (op-amp outputs are fine). Measure noise with inputs shorted,
  not floating.
- **Relays/mains:** two relays switching a mains lamp sit on a separate XIAO
  board, to keep mains noise away from the ADC.
- **Camera:** Basler acA1440-220um (USB3, mono), 1440×1080 Mono8, gain 28 dB,
  exposure usually 2000 µs. Verified at 200 fps full frame with 0 dropped
  frames. 10 s at 100 fps is about 1.6 GB in RAM; the PC has about 11 GB free.

## Converting counts to volts

`V = (code − 8192) × LSB`. Codes are 14-bit offset binary (0x2000 = 0 V).
The electrometer output goes straight into the MAX1032, so there's no
pedestal or gain to undo. Codes 0 and 16383 mean the input is at or beyond
the range (railed). The PC takes the zero code and LSB from the board's
header rather than hard-coding them.

The input range is set at runtime (`--range` on `main.py`, `capture.py`,
`adc_serial.py`), the same for all channels:

| `--range` | MAX1032 code R[2:0] | span | LSB |
|---|---|---|---|
| 3 | 001 | ±3.072 V | 375 µV |
| 6 | 100 | ±6.144 V | 750 µV |
| **12 (default)** | 111 | ±12.288 V | 1.5 mV |

The inputs tolerate ±16.5 V. The default was ±3.072 V until 2026-10-05,
when the electrometer outputs were found to clip it. A live check across the
three ranges that day: the middle ring clipped at +3.07 and +6.14 V and read
+9.12 V on ±12.288 V; the top ring clipped at both narrower ranges and swung
−9.0 to −3.4 V on ±12.288 V; unclipped channels read the same in every range
(bottom ring ≈ −2.4 V, floating channel ≈ +0.27–0.30 V), which confirms the
codes and LSBs. Range codes came from memory of the datasheet (it couldn't be
downloaded at the time); that check is what verifies them.

## Firmware (`xiao_max1032/xiao_max1032.ino`)

- Every sample period it converts CH0–CH3 back to back (external clock mode,
  SPI at 3.64 MHz, just under the 3.67 MHz limit). The SPI peripheral drives
  CS itself, one 32-bit transfer per conversion.
- **Rate, measured on the board:** converting all four channels takes 45.9 µs
  (the ideal is 35 µs: 4 × 32 SPI clocks; the rest is Arduino SPI overhead).
  With 20% slack for packet/USB work the firmware allows up to ~18.2 kHz and
  runs at 17,857 Hz (56 µs period) by default; it delivers ~17,750 samples/s
  per channel (99.4%), longest gap between samples 116 µs, no lost packets.
  At `--rate 10000` it delivers exactly 10,000/s. Going beyond ~18 kHz would
  need register-level SPI code instead of the Arduino SPI library.
- **Stream format:** packets of 64 samples:
  `A5 5A | seq (u32) | t0_us (u32) | 64 × [dt_us (u16), code0..code3 (u16)] | xor8`
  (651 bytes, ~10.2 bytes per sample, ~180 KB/s at 17.8 kHz).
  `seq` counts samples since streaming started, so a jump means dropped
  packets. **Every sample carries its real conversion time** (`t0_us + dt_us`),
  so timing is exact even when the loop runs a little late; an earlier
  version labeled samples with their scheduled time and was off by ~12% when
  the board couldn't keep up.
- **Commands:** `?` prints the text header (ends with `# end`; the
  `# cfg key=value …` line has the rate, period, max rate, LSB, zero code,
  packet layout); `r<hz>\n` sets the rate (capped at `max_hz`); `g<1|4|7>\n`
  sets the input range (±3.072 / ±6.144 / ±12.288 V, default 7); `s` starts
  streaming; `x` stops.
- **Never blocks:** if the PC isn't reading fast enough, whole packets are
  dropped and show up as `seq` gaps.
- **ESP32-C6 gotcha:** on single-core chips the Arduino core pauses the main
  loop for 5 ms every 2 s (`yieldIfNecessary()` → `vTaskDelay(5)`), which put
  a 5 ms hole in the data every 2 s. The firmware now stays inside `loop()`
  for the whole stream, and disables the idle-task watchdog.
- Re-sends the channel config once a second, because the MAX1032 config can't
  be read back and a supply glitch would reset it.
- Flash from the `tube/` folder:
  `arduino-cli compile --upload --fqbn esp32:esp32:XIAO_ESP32C6:CDCOnBoot=cdc -p /dev/ttyACM0 xiao_max1032`
  (or `python3 main.py --flash`). If the port appears then vanishes: hold
  BOOT, tap RESET, flash again.
- The old ADS1115 firmware is kept in `legacy/esp32_ads/`.

## Software (all run from the `tube/` folder)

| file | what it does |
|---|---|
| `main.py` | Live GUI. Basler feed on the left, one scrolling trace per channel on the right (Pause, Clear, Autoscale Y, time window). Traces are reduced to min/max per screen pixel so short spikes stay visible at ~18 kHz. `--rate`, `--flash`, `--no-camera`, `--port`. |
| `capture.py` | Records into RAM and saves when done. First a noise floor (`--baseline`, default 10 s, ADC only), then the capture (`-d` seconds, `--fps`, `--exposure` µs, `--rate` ADC samples/s per channel, `--range` ADC input range in ±V: 3, 6 or 12 (default 12), `--no-camera`). Prints `>>> RECORDING … Drop now. <<<` when it's time to drop. |
| `view_adc.py` | Plots a session's charge data (newest by default). `--baseline` shows the noise floor, `--raw` shows codes, `--smooth N` overlays a moving average. Titles show mean and std per channel. |
| `make_video.py` | Makes two MP4s per run, `…_raw.mp4` and `…_filtered.mp4` (mains hum removed). Camera on the left; all channels for the clip on the right with a red cursor at the current frame. Arguments: `start end` (inclusive, optional), `--speed` (default 0.1 × real time), `--session`. |
| `filters.py` | `remove_mains(t, y)`: finds the actual mains frequency (it drifts ~0.02 Hz), then fits 60 Hz and harmonics up to 600 Hz by least squares in overlapping 0.5 s windows and subtracts only the hum. Uses real sample times, so dropped packets don't matter. Tested on synthetic data: 15 mV RMS of hum down to the 0.3 mV noise floor; a 2 ms, 30 mV blip comes through at 29.2 mV; an 85 mV step is kept exactly (up to ~3 mV error right at the step). For display only; saved data stays raw. |
| `vid.py` | Quick frame-by-frame viewer for `frames.npy` (a/d or arrow keys step, space plays, q quits). |
| `adc_serial.py` | Reads the board's stream in the background (used by everything else). Run directly for a once-a-second summary per channel: mean, rms noise, lost samples, railed codes. `--rate` sets the rate. |
| `channels.py` | Channel names, colors and plot order, in one place (top ring, middle, bottom, floating). |
| `sync.py` | Puts camera and ADC times on one clock (see below). |
| `basler_feed.py`, `scope_panel.py` | The camera and plot panels used by `main.py`. |

Only one program can have the camera or the serial port open at a time. The
port is opened exclusively, so a second program gets a "busy" error instead
of both silently receiving scrambled data. Close `main.py` before running
`capture.py`.

The scripts run from the `particle-electrostatics-exp/.venv` Python
environment (it has `pypylon`); the system Python doesn't.

## Session folder: `captures/session_<YYYY-MM-DD_HH-MM-SS>/`

| file | contents |
|---|---|
| `baseline.csv` | Noise floor, same columns as `adc.csv`. `t` runs from −baseline to 0. |
| `adc.csv` | `t,host_t,esp_t,raw0..raw3,v0..v3`. About 17,800 rows per second (about 1.5 MB per second of CSV). Raw and unfiltered. |
| `frames.npy` | `(n_frames, 1080, 1440)` uint8. Load with `np.load(p, mmap_mode='r')`. |
| `frames.csv` | `index,t,host_t,cam_t,block_id`. Row `i` is frame `i` in `frames.npy`. |
| `meta.txt` | `key<TAB>value`: settings, the board's `adc_cfg`, `channel_names`, `plot_order`, sample counts and lost samples, dropped frames, `cam_sync`, and from the baseline `noise_std_mV` and `hum_60hz_amp_mV` per channel. |
| `video_*.mp4` | Output of `make_video.py`. |

- `t`: synced time in seconds since the capture started. **Use this to match
  frames to ADC samples.** Frame `t` is the middle of its exposure.
- `host_t`: when the PC received the data. ADC samples arrive in packets, so
  it's the packet's arrival time. Don't use it for alignment.
- `esp_t`, `cam_t`: each device's own clock (`cam_t` in ns).

Sessions from the old ADS1115 board (before 2026-10) have 3 channels
(`raw0..2`, `v0..2` = bottom, middle, top ring), 500 rows per second, and a
different conversion (see the legacy section below). The viewers detect this
and label them correctly. Sessions before about 17:30 on 2026-09-29 also lack
the `t` column and baseline; the scripts fall back to an approximate alignment
for those and print a warning.

## Timing and sync (`sync.py`)

- **Camera:** at the start and end of each capture, the PC asks the camera
  for its clock (`TimestampLatch`) and times the round trip (0.35 ms). That
  gives the camera→PC offset to about 0.2 ms; the offset is interpolated
  between the two readings. Measured drift is about 13 ppm, which is
  negligible. Each frame gets `cam_t` + offset + half the exposure.
- **Why not arrival time:** frames reach the PC 5.5–7.5 ms after mid-exposure
  (exposure, 4.4 ms sensor readout, USB transfer). The first version aligned
  on arrival and labeled every frame about 8 ms late.
- **ADC:** each sample's board time is its real conversion time. The board
  clock is aligned to the PC by assuming the fastest-arriving packet had
  essentially no USB delay, which is good to about 1 ms. The MAX1032 samples
  at the start of each conversion, so no further correction is applied. CH1–3
  are converted ~11 µs after the channel before; small next to the 56 µs
  sample period. (The old ADS1115 board
  needed a 1/860 s correction; it's still applied to its sessions.)
- **Overall:** the software alignment is good to a few ms at worst. Any delay
  inside the electrometers is *not* corrected: the timestamp marks when the
  ADC sampled, not when the charge was at the ring.

## Findings so far (old ADS1115 board, 500 SPS per channel)


**60 Hz mains pickup dominates the noise.**
- Early on it was about ±20 mV at the electrometer scale (about ±4 mV at the
  ADC pin), in phase on all three channels (correlation 0.97), which points to
  one shared source.
- The latest noise-floor measurement (about 18:31) showed 60 Hz amplitudes of
  bottom ≈ 6 mV, middle ≈ 18 mV, top ≈ 85 mV. The top ring had become 4–5×
  noisier than earlier, probably because of a setup change.
- Suspected source: room wiring and lights coupling onto the rings. A ground
  loop through USB is also possible. Checks to try: lights off; zero-check or
  disconnect a ring at its electrometer (if the 60 Hz stays, it enters after
  the electrometer); PC on battery.

**Drop analysis, session `17-38-31` (200 fps, 2000 µs exposure).**

The ball was tracked in the frames by subtracting the background. It's a few
pixels across.

| t (s) | ball position (image y, 0 = top of frame, 1080 = bottom) |
|---|---|
| 2.630 | enters at y ≈ 40 |
| 2.665–2.680 | y ≈ 274–387 |
| 2.715–2.725 | y ≈ 701–788 |
| 2.750 | leaves the bottom of the frame (y ≈ 1060) |
| 2.77–2.90 | bouncing/settling at y ≈ 994–1079 |

It falls at roughly 8,500 px/s, so it crosses a ring band in about 15 ms.

With the 60 Hz removed (FFT notch at 60/120/180/240 Hz, the first filter version):
- **Top and middle rings:** no visible response as the ball passes (under
  ±5 mV).
- **Bottom ring:** a step from about −31 to +55 mV starting at 2.78 s, right
  as the ball arrives at the bottom (10–90% rise 2.781 → 2.907 s, 126 ms).
  It stays high with large slow swings afterwards. The bottom ring also shows
  a 4–5 Hz wobble through the whole recording, before the drop too.

Interpretation: the timing looks consistent. The one large event lines up
with the ball arriving at the bottom within about 20–30 ms. The open problem
is that **the fly-by through a ring barely registers.** Two candidate causes:
1. The electrometers' output is too slow (integration time/NPLC, digital
   filtering, averaging) and smears a 15 ms pass into almost nothing, while a
   ball sitting near a ring shows up at full size.
2. The passage signal is really below about 5 mV and is hidden in the noise.

A test to tell them apart: hold the charged ball inside one ring for about a
second, then pull it out quickly. If that gives a clean, large step, the rings
work and the fly-by is being lost to bandwidth.

## Next steps

- **Check the inputs on the new board.** At the first real test (2026-10-02,
  firmware flashed and streaming fine) the channels read: CH0 (top ring) and
  CH3 (bottom ring) swinging rail to rail (≈2.3 V RMS, mostly 60 Hz at
  2.4–3 V amplitude), CH2 (middle ring) pinned at +3.07 V (top of range).
  That looks like the electrometers weren't connected or powered. CH1 sat
  steady at +2.16 V, which is expected: Electro2 isn't connected, and a
  floating MAX1032 input reads about +2.2 V.
  Then do the scale check against a known voltage and record the
  shorted-input noise floor per channel (`python3 adc_serial.py`).
- Run the hold-and-withdraw test above. Check the electrometers' integration
  time and filter settings: at ~18 kHz the ADC is no longer the bottleneck, so
  if fly-bys are still missing, the electrometers are the next suspect.
- Reduce the 60 Hz at the source (shielding, grounding, twisted or shielded
  cable). The passage signals were 5–35 mV at most on the old board.
- Analysis ideas: add the mains filter to `view_adc.py` too; detect blips
  automatically against the noise floor; at ~18 kHz, look at blip shape, not
  just whether one exists.

---

## Legacy: ADS1115 raw → voltage (old board, sessions before 2026-10)

Each electrometer signal goes through an **INA159** level-shifting amplifier
and then into an **ADS1115** (16-bit, PGA set to ±4.096 V). On the old board the
raw ADS1115 count arrives over serial as a signed integer (`raw0..2`).

**The math:**

1. Count → voltage at the ADC pin: the ADS1115 LSB is `4.096 / 32768 = 125 µV`, so
   `V_pin = raw * 4.096 / 32768`.
2. Undo the INA159 (gain 0.2, output centered at 1.25 V):
   `V_in = (V_pin - 1.25) / 0.2 = 5 * (V_pin - 1.25)`.

Combined: `V_in = 5 * (raw * 4.096 / 32768 - 1.25)`

```python
ADS_LSB = 4.096 / 32768.0   # 125 uV per count (ADS1115 @ +/-4.096 V PGA)

def convert_to_voltage(raw: int) -> float:
    """ADS1115 @ +/-4.096V PGA through INA159: V = 5*(Vpin - 1.25)"""
    return 5.0 * (raw * ADS_LSB - 1.25)
```

**Sanity checks:**

| raw   | V_pin   | V_in (electrometer) |
|-------|---------|---------------------|
| 0     | 0.000 V | −6.25 V             |
| 10000 | 1.250 V | 0.00 V              |
| 20000 | 2.500 V | +6.25 V             |

- Resolution at the input is `5 * 125 µV = 625 µV` per count.
- If the PGA gain setting on the ADS1115 changes, update `ADS_LSB`
  (full-scale / 32768: ±2.048 V → 62.5 µV, ±6.144 V → 187.5 µV).

**From the older `particle-electrostatics-exp` project, not this rig:** that
project's `gui/motor_controls_gui.py` used the same conversion with the serial
format `seq,ms,spare,electro1_raw,electro2_raw` (five integers per line,
about 100 Hz, third field unused) and computed `dV/dt = (V - V_prev) / 0.01`
assuming 100 Hz. Its `server/server.py` (older ESP32 + BLE setup) used a
different 14-bit SPI ADC with range codes, `ratio = raw / 2**14`; range 4, the
default, gives `V = ratio * 3 * 4.096 - 1.5 * 4.096` (±6.144 V). Don't mix the
two formulas.

---

# Appendix: Basler camera + live Tkinter feed (original notes)

`basler_feed.py` in this folder is built from these notes: the same grabber
thread, turned into an embeddable `CameraPanel` frame, with a clean shutdown
that waits for the grabber to release the camera. `capture.py` changes a few
settings for recording: Mono8 set explicitly, exposure set, a fixed frame rate,
`GrabStrategy_OneByOne` so no frames are skipped, and 64 buffers.

How to open a Basler camera with `pypylon` and show a live feed in a Tkinter window.
Adapted from `gui/camera_feed_gui.py` in `particle-electrostatics-exp`, with the contour
tracking, frame saving, and shared-state stuff removed.

## Dependencies

```
pip install pypylon==4.2.0 opencv-python numpy pillow
```

- `pypylon` wheels bundle the pylon runtime, so the full Basler pylon SDK is not
  strictly required. On Linux, USB3 cameras still need udev permissions. If
  `EnumerateDevices()` returns nothing but `lsusb` shows the camera, install the pylon
  SDK (it installs the udev rules) or run once with sudo to confirm it's a permissions problem.
- Tkinter comes with Python (`sudo apt install python3-tk` if it's missing on Linux).
- Only one process can open the camera at a time. Close pylon Viewer before running.

## Architecture

- **Grabber thread** (`CameraGrabber`): opens the camera, grabs frames in a loop,
  converts each one to a numpy array, and pushes it into a small `queue.Queue(maxsize=2)`.
  When the queue is full it drops frames (`put_nowait`), so the UI never falls behind.
- **Tk UI loop**: `root.after(...)` runs about 30 times per second, drains the queue
  down to the newest frame, and draws it on a `tk.Label`.
- **Shutdown**: a `threading.Event` tells the grabber to stop. The grabber's `finally`
  block always calls `StopGrabbing()` and `Close()` so the camera is released.
- Never call Tk from the grabber thread. Tk is not thread-safe, so only the main thread
  touches widgets.

## Camera settings used in the original project

- `GainAuto = "Off"` and `Gain = 28.0`.
- `PixelFormat` is set to the first supported format (`GetSymbolics()[0]`), which is
  usually `Mono8` on mono cameras. Set it explicitly (for example `"Mono8"`) if you need
  a specific format.
- `GrabStrategy_LatestImageOnly`: always hands over the newest frame and skips any backlog.
- `ImageFormatConverter` outputs `Mono8` for mono formats and `BGR8packed` otherwise,
  so frames can go straight into OpenCV.
- Exposure is not set (the camera default is used). To set it:
  `cam.ExposureAuto.SetValue("Off")`, then `cam.ExposureTime.SetValue(us)` on
  USB/ace2 cameras, or `cam.ExposureTimeAbs.SetValue(us)` on older GigE models.

## Reference implementation (drop-in module)

```python
#!/usr/bin/env python3
"""Minimal Basler live feed in Tkinter (no processing)."""

import tkinter as tk
import threading, queue
from PIL import Image, ImageTk
import cv2
from pypylon import pylon

TARGET_UI_FPS = 30


class CameraGrabber(threading.Thread):
    def __init__(self, frame_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.q = frame_queue
        self.stop_event = stop_event
        self.cam = None
        self.err = None

    def run(self):
        try:
            tl = pylon.TlFactory.GetInstance()
            if not tl.EnumerateDevices():
                self.err = "No Basler cameras found."
                return

            self.cam = pylon.InstantCamera(tl.CreateFirstDevice())
            self.cam.Open()
            print("Opened:", self.cam.GetDeviceInfo().GetModelName())

            self.cam.GainAuto.SetValue("Off")
            self.cam.Gain.SetValue(28.0)
            self.cam.PixelFormat.SetValue(self.cam.PixelFormat.GetSymbolics()[0])

            converter = pylon.ImageFormatConverter()
            if "Mono" in self.cam.PixelFormat.GetValue():
                converter.OutputPixelFormat = pylon.PixelType_Mono8
            else:
                converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            while not self.stop_event.is_set() and self.cam.IsGrabbing():
                grab = self.cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if grab.GrabSucceeded():
                    img = converter.Convert(grab).GetArray()
                    try:
                        self.q.put_nowait(img)
                    except queue.Full:
                        pass
                grab.Release()
        except Exception as e:
            self.err = f"{type(e).__name__}: {e}"
        finally:
            try:
                if self.cam:
                    if self.cam.IsGrabbing():
                        self.cam.StopGrabbing()
                    if self.cam.IsOpen():
                        self.cam.Close()
            except Exception:
                pass


def build_camera_window(parent: tk.Misc) -> tk.Toplevel:
    top = tk.Toplevel(parent)
    top.title("Basler Camera Feed")

    label = tk.Label(top)
    label.pack(fill="both", expand=True)
    info = tk.Label(top, text="Initializing camera…")
    info.pack(anchor="w")

    frame_queue = queue.Queue(maxsize=2)
    stop_event = threading.Event()
    grabber = CameraGrabber(frame_queue, stop_event)
    grabber.start()

    tk_image = None  # keep a reference or Tk garbage-collects the image

    def update_ui():
        nonlocal tk_image
        if stop_event.is_set():
            return

        frame = None
        try:
            while True:  # drain to newest frame
                frame = frame_queue.get_nowait()
        except queue.Empty:
            pass

        if frame is not None:
            if frame.ndim == 2:
                pil = Image.fromarray(frame)  # mono8
            else:
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            tk_image = ImageTk.PhotoImage(pil)
            label.configure(image=tk_image)

        info.config(text=f"[!] {grabber.err}" if grabber.err else "Streaming…")
        top.after(int(1000 / TARGET_UI_FPS), update_ui)

    def on_close():
        stop_event.set()
        top.destroy()

    top.protocol("WM_DELETE_WINDOW", on_close)
    top.after(100, update_ui)
    return top


def attach_camera_feed(parent: tk.Misc | None = None):
    """Open the camera window. Pass an existing Tk root to embed it in an app,
    or None to run it standalone."""
    if parent is None:
        root = tk.Tk()
        root.withdraw()  # hide empty root; only the camera window shows
        top = build_camera_window(root)
        top.bind("<Destroy>", lambda e: root.quit() if e.widget is top else None)
        root.mainloop()
    else:
        build_camera_window(parent)


if __name__ == "__main__":
    attach_camera_feed(None)
```

## Using it from another app

```python
import tkinter as tk
from basler_feed import attach_camera_feed   # whatever you name the module above

root = tk.Tk()
# ... build your own controls on root ...
attach_camera_feed(parent=root)   # opens the camera in its own Toplevel window
root.mainloop()
```

## Adding processing later

Do per-frame work (OpenCV, etc.) inside `update_ui` on the `frame` array before it is
converted to PIL. If the processing is heavy, move it into a separate worker thread with
its own queue so the UI stays responsive. Use `frame.copy()` before modifying a frame
you also want to save or pass on.

## Gotchas

- `RetrieveResult(5000, ...)` raises an exception after 5 s with no frame (for example
  in hardware-trigger mode). The exception ends up in `grabber.err` and is shown in the window.
- Always call `grab.Release()`. Otherwise pylon runs out of buffers.
- High-resolution frames may be bigger than the screen. Resize before display with
  `pil.thumbnail((w, h))` or `cv2.resize`.
- `CreateFirstDevice()` picks whichever camera it finds first. With several cameras,
  filter `EnumerateDevices()` by `GetSerialNumber()` and use `tl.CreateDevice(dev)`.
