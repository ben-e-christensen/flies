# Tube rig: charged-ball drop past 3 induction rings, with Basler video

State as of 2026-09-29. This file is meant to stand on its own: a reader who
can't see the code should still be able to follow the setup, the data and the
findings so far. The original Basler notes are kept in the appendix at the end.

## The experiment

A charged ball is dropped down a vertical clear tube. The tube has three
sensor rings (top, middle, bottom). Each ring feeds an electrometer, so a
charge passing through a ring should show up as a blip on that channel. A
high-speed Basler camera films the tube so the blips can be matched to the
ball's position.

## Hardware

- **Microcontroller:** Adafruit Feather ESP32-S3 (2MB PSRAM), USB native CDC
  at `/dev/ttyACM0` (USB VID 0x239A, PID 0x811B). I2C on SDA=GPIO3, SCL=GPIO4
  at 400 kHz. The board powers its I2C/STEMMA port through GPIO7 at boot.
- **ADCs:** 3× ADS1115 (16-bit). Each channel's signal chain is
  ring → electrometer (analog output) → INA159 level shifter (gain 0.2,
  1.25 V reference) → ADS1115 input A0, single-ended.

  | ADS1115 ADDR pin | I2C address | ring | data column |
  |---|---|---|---|
  | GND | 0x48 | **bottom** | `raw0` / `v0` |
  | VDD (3.3 V) | 0x49 | **middle** | `raw1` / `v1` |
  | SDA | 0x4A | **top** | `raw2` / `v2` |

  A1–A3 are unused and float at about 0.59 V. A0 sits at about 1.245 V when
  the electrometer output is zero, which is the INA159's 1.25 V pedestal.
- **Camera:** Basler acA1440-220um (USB3, mono), 1440×1080 Mono8, gain 28 dB,
  exposure usually 2000 µs. It has been verified at 200 fps full frame with 0
  dropped frames. 10 s at 100 fps is about 1.6 GB in RAM; the PC has about
  11 GB free.

## Converting counts to volts

`V_in = 5 × (raw × 4.096/32768 − 1.25)`, which undoes the ADS1115 LSB
(125 µV at ±4.096 V range) and the INA159 (gain 0.2, 1.25 V offset).
raw 0 → −6.25 V, raw 10000 → 0 V, raw 20000 → +6.25 V. Resolution at the
electrometer output is 625 µV per count. The full derivation is in the
"ADC raw → voltage" section below.

If the ADS1115 range is changed, update three places: `PGA` and `FSR_V` in
`esp32_ads.ino`, and `ADS_FSR_V` in `adc_serial.py`. Switching to ±2.048 V
would halve the step to about 312 µV and still cover −6.25 V to +4.0 V at the
input, because the pin sits at 1.25 V.

## Firmware (`esp32_ads/esp32_ads.ino`)

- Each ADS1115 runs in continuous mode at 860 SPS. The ESP reads all three
  every 2 ms (`SAMPLE_HZ = 500`) and prints one CSV line:
  `t_us,raw0,raw1,raw2`, where `t_us` is the ESP's `micros()` at the read and
  the values are signed counts (`nan` if a chip isn't answering). Conversion
  to volts happens on the PC.
- Lines starting with `#` are info. Commands the PC can send: `?` re-prints
  the config header; `a` reads A0–A3 on every chip once and prints the pin
  voltages (useful for finding which input a signal is on).
- **Self-check:** an ADS1115 that loses power comes back powered down with its
  output register at 0, and it still answers on I2C. Before this was handled,
  the data went to raw 0 (−6.25 V) on every channel until the ESP was reset
  (session `16-56-31`). Now the firmware reads back one chip's config every
  20 ms, re-applies it if it doesn't match, and prints
  `# ADCn 0x.. was reset (config 0x8583), reconfigured`.
- Flash from the `tube/` folder (no Arduino IDE needed):
  `arduino-cli compile --upload --fqbn esp32:esp32:adafruit_feather_esp32s3 -p /dev/ttyACM0 esp32_ads`

## Software (all run from the `tube/` folder)

| file | what it does |
|---|---|
| `main.py` | Live GUI. Basler feed on the left, 3 scrolling traces on the right (Pause, Clear, Autoscale Y, time window). `--flash` uploads the firmware first, `--no-camera` shows traces only, `--port` overrides the port. |
| `capture.py` | Records into RAM and saves when done. First a noise floor (`--baseline`, default 10 s, ADC only), then the capture (`-d` seconds, `--fps`, `--exposure` µs, `--no-camera`). Prints `>>> RECORDING … Drop now. <<<` when it's time to drop. |
| `view_adc.py` | Plots a session's charge data. Defaults to the newest session. `--baseline` shows the noise floor, `--raw` shows counts, `--smooth N` overlays a moving average. Titles show mean and std per ring. |
| `make_video.py` | Makes two MP4s of a session, `…_raw.mp4` and `…_filtered.mp4` (60 Hz removed). Camera on the left; all three traces for the clip on the right with a red cursor at the current frame. Arguments: `start end` (inclusive, optional), `--speed` (default 0.1 × real time, so 200 fps plays at 20 fps), `--session`. Encodes with ffmpeg/libx264. |
| `filters.py` | `notch_mains(t, y)`: removes 60, 120, 180 and 240 Hz with soft-edged notches (0.5 Hz sigma) over the whole record, keeping the DC level and fast blips. On synthetic data it cut 60+120 Hz hum from 14.6 to 0.9 mV RMS and passed a 15 ms, 10 mV blip at 9.96 mV. `make_video.py` runs it over noise floor + capture when a baseline exists. For display only; saved data stays raw. |
| `vid.py` | Quick frame-by-frame viewer for `frames.npy` (a/d or arrow keys step, space plays, q quits). |
| `adc_serial.py` | Serial reader thread used by everything else. Run directly to print live samples; `--pins` runs the A0–A3 pin check. |
| `sync.py` | Puts camera and ADC times on one clock (see below). |
| `basler_feed.py`, `scope_panel.py` | The camera and plot panels used by `main.py`. Channel names, colors and plot order (top ring first) live in `scope_panel.py`. |

Only one program can have the camera or the serial port open at a time. The
port is opened exclusively, so a second program gets a "busy" error instead
of both silently receiving scrambled data (which happened once during
debugging). Close `main.py` before running `capture.py`.

## Session folder: `captures/session_<YYYY-MM-DD_HH-MM-SS>/`

| file | contents |
|---|---|
| `baseline.csv` | Noise floor, same columns as `adc.csv`. `t` runs from −baseline to 0. |
| `adc.csv` | `t,host_t,esp_t,raw0,raw1,raw2,v0,v1,v2`. About 500 rows per second. Raw and unfiltered. |
| `frames.npy` | `(n_frames, 1080, 1440)` uint8. Load with `np.load(p, mmap_mode='r')`. |
| `frames.csv` | `index,t,host_t,cam_t,block_id`. Row `i` is frame `i` in `frames.npy`. |
| `meta.txt` | `key<TAB>value`: settings, rates, dropped frames, `cam_sync`, and from the baseline `noise_std_mV` and `hum_60hz_amp_mV` per ring. |
| `video_*.mp4` | Output of `make_video.py`. |

- `t`: synced time in seconds since the capture started. **Use this to match
  frames to ADC samples.** Frame `t` is the middle of its exposure.
- `host_t`: when the PC received the sample. It includes transfer delay, so
  don't use it for alignment.
- `esp_t`, `cam_t`: each device's own clock (`cam_t` in ns).

Captures made before about 17:30 on 2026-09-29 have no `t` column or baseline.
The scripts fall back to an approximate alignment for those and print a
warning.

## Timing and sync (`sync.py`)

- **Camera:** at the start and end of each capture, the PC asks the camera
  for its clock (`TimestampLatch`) and times the round trip (0.35 ms). That
  gives the camera→PC offset to about 0.2 ms; the offset is interpolated
  between the two readings. Measured drift is about 13 ppm, which is
  negligible. Each frame gets `cam_t` + offset + half the exposure.
- **Why not arrival time:** frames reach the PC 5.5–7.5 ms after mid-exposure
  (exposure, 4.4 ms sensor readout, USB transfer). The first version aligned
  on arrival and labeled every frame about 8 ms late.
- **ADC:** the ESP clock is aligned by assuming the fastest-arriving line had
  essentially no USB delay. Lines arrive 1.2–3.2 ms after their synced time,
  so this is good to about 1 ms. Then 1/860 s is subtracted, because the value
  read at `esp_t` comes from a conversion that averaged over the previous
  ~1.2 ms.
- **Overall:** the software alignment is good to a few ms at worst. Any delay
  inside the electrometers or INA159s is *not* corrected: the timestamp marks
  when the ESP read the voltage, not when the charge was at the ring.

## Findings so far

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

With the 60 Hz removed (FFT notch at 60/120/180/240 Hz):
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

- Run the hold-and-withdraw test above. Check the electrometers' integration
  time and filter settings.
- Reduce the 60 Hz at the source (shielding, grounding, twisted or shielded
  cable). The passage signals are 5–35 mV at most, so this matters more than
  anything on the software side.
- Analysis ideas: add the 60 Hz filter to `view_adc.py` too (it's in
  `make_video.py` already); fit and subtract the exact hum from the noise
  floor; detect blips automatically against the noise floor.
- Hardware headroom if needed: ADS1115 up to 860 SPS (`SAMPLE_HZ`), ±2.048 V
  range for 2× resolution, camera faster than 200 fps with a shorter exposure.

---

## ADC raw → voltage (derivation)

Each electrometer signal goes through an **INA159** level-shifting amplifier
and then into an **ADS1115** (16-bit, PGA set to ±4.096 V). In this rig the
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
