# FLIES — electrometer + camera acquisition and analysis

Scripts for the FLIES charging experiments in the Burton Lab. A Keithley 6514
electrometer logs charge (or current/voltage) over serial while FLIR
Grasshopper3 and/or Logitech Brio 101 cameras record the sample. An Arduino
relay switches two lamps (A = "going into Faraday cage", B = "going into
acrylic"). When the signal "blips", the scripts save a camera burst around it.
The other scripts plot, review, turn into video, and compress the resulting
session folders.

Split out of [burtonlab](https://github.com/ben-e-christensen/burtonlab)
(`flies/` folder) with its git history kept.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install numpy matplotlib opencv-python pyserial pillow pandas
```

You also need:

- **PySpin** (Spinnaker SDK) for the FLIR camera. Install the wheel that
  matches your Python version from Teledyne FLIR's site. This is why the old
  `.venv312` existed.
- **ffmpeg / ffprobe** for the Brio capture, the video scripts and the
  compressor. Several scripts hard-code an `FFMPEG = r'...'` path in their
  CONFIG block, so check it.
- Set the instrument serial ports in **`coms.py`**.

Most scripts have a `# ============ CONFIG ============` block at the top, and
many default to paths like `E:\Ben Christensen\FLIES\...`. Edit it before you
run a script, or pass a session folder on the command line.

## A session folder

The acquisition scripts write one folder per run:

```
session_<YYYY-MM-DD_HH-MM-SS>/
    meta.txt                 key<TAB>value settings (cap_F, fps, ports, wall_clock_start ...)
    electrometer.csv         time,charge,trigger  |  time,voltage_V,trigger (voltage mode)
    events.csv               one row per blip / manual save (event,time,lamp,...)
    lamps.csv                lamp state over time (newer sessions only)
    flir/e001_0000.png ...   FLIR burst frames for event 1 (16-bit PNG)
    flir_frames.csv          time,event,filename
    brio1/e001_0000.jpg ...  Brio burst frames (+ brio1_frames.csv)
    *_cont/p003_000000.*     continuous frames during lamp phase 3 (some modes)
    analysis/                written by the analysis scripts
```

All timestamps are seconds since the Start button, on one shared clock, so the
CSVs line up with each other.

---

## Top-level files

### Shared config

| File | What it does |
|---|---|
| `coms.py` | The one place for serial ports. `port('keithley')` / `port('arduino')` returns the right `COMx` on Windows or `/dev/tty*` on Linux. Every acquisition script imports it. |

### Acquisition (Tkinter GUIs: live plot, lamp buttons, Start/Stop)

| File | What it does |
|---|---|
| `main.py` | Charge-mode Keithley + **one FLIR**, with automated lamps. Start runs the cycle: lamp A on → wait for a charge blip → lamp stays on 30 s → dark → lamp B → … . The FLIR keeps a RAM ring buffer and writes it to disk only when a blip fires, so each burst includes the lead-in. Pressing a manual lamp button turns the automation off. (The docstring still says "UNTESTED", but the commit history says it has been tested.) |
| `main_untested.py` | Same automation with **two FLIRs**. Only the camera opposite the lit lamp caches. The blip rule is a swing in either direction, and each burst includes 15 s of post-roll after the blip. |
| `main_test.py` | Dual-FLIR version from `main_untested.py`, plus **continuous 20 fps saving** from camera A while one lamp (`CONT_LAMP`) is on. Frames go to `flir_a_cont/p###_######.png` through a bounded writer pool. When the disk can't keep up, frames are dropped and counted so RAM doesn't fill. |
| `main_voltage.py` | The newest and largest version. The Keithley runs in **voltage mode** across a 1 nF cap (1 mV = 1 pC), with one FLIR (PySpin) and two Brio 101s (read through ffmpeg for real MJPG). Each lamp has its own settings for which cameras cache and which save continuously. It adds an "Auto trigger" toggle and a "Save burst" button (or the M key). Writes to `E:\Ben Christensen\FLIES\session_*`. |

### Session analysis and plotting

| File | What it does |
|---|---|
| `plot_main.py` | Plots charge for a whole session (`electrometer.csv`). Events show as dashed lines and FLIR burst spans as cyan bands. Saves `session_charge.png`. Works with charge-mode and voltage-mode data (Q = C·V, with C from `meta.txt`). |
| `plot_blips.py` | One plot per event, covering ±30 s around the blip. A second x-axis shows FLIR frame numbers and shades the burst span. |
| `session_overview.py` | Whole-session overview plus one plot per **episode** (events close together are merged). Colors come from the lamp state logged in `events.csv` / `lamps.csv`. Writes `analysis/overview/`. Pass a session folder, or the FLIES folder to process every session. |
| `session_overview_multiple.py` | One long charge plot across several sessions or folders, ordered by `wall_clock_start`. Sessions are either laid end to end or placed at their real clock times. Writes PNG, SVG (for Inkscape) and a sessions CSV. |
| `logger.py` | Despite the name, this is the interactive **blip viewer**. It shows the session trace, a zoom on the current event, and the Brio frame nearest the cursor. Events come from `events.csv`, your own `EXTRA_TIMES` and optional offline detection. Keyboard controls: n/p for events, arrow keys to step frames, space to play, s to take a snapshot, e to export. `--export` writes summary PNGs without opening the window. |

### Video making

| File | What it does |
|---|---|
| `flir_video.py` | Makes one MP4 per event from the FLIR burst frames (`e001_*.png` …), with the 16-bit frames scaled to 8-bit. |
| `video_and_charge.py` | Event video. Every camera that recorded the event is on top (FLIR, brio1, brio2), and the charge trace draws itself in underneath. The FLIR sets the clock. The other cameras show their latest frame at or before that time, labeled with the offset. Also writes a `_sync.csv` that maps each video frame to its source files. |
| `video_cont.py` | Same as `video_and_charge.py`, for sessions where the Brios saved **continuous** frames (`brio*_cont/`) instead of bursts. |
| `triple.py` | An older 3-panel version for one FLIR + one Brio: FLIR burst, Brio burst and charge trace side by side, with a sync CSV. |
| `trying.py` | Side-by-side clip of Brio continuous frames and a charge trace that draws itself in, for frame ranges you list in `CLIPS` (e.g. `p002_001550` → `p002_001560`). Supports slow motion. |

### Frame quality check

| File | What it does |
|---|---|
| `fly_finder.py` | Scans every continuous FLIR frame (`flir_a_cont/p###_######.png`) and compares it with a reference frame using mean diff, fraction of changed pixels, sharpness and brightness. It flags robust outliers. The scan can resume where it stopped. Writes `scores.csv`, `flagged.csv`, a plot and copies of flagged frames to `frame_check/`. |
| `flag.py` | Summarizes the results from `fly_finder.py`: counts by reason, by session and lamp phase, runs of consecutive flagged frames, the most extreme frames, and missing files. It can copy the worst N frames per metric so you can look through them. |
| `prune.py` | One-off: removes rows at or before a given reference frame from `scores.csv`, plus rows whose files no longer exist. Keeps a `.bak` copy. The paths are hard-coded. |

### Data transfer

| File | What it does |
|---|---|
| `compress.py` | Copies a FLIES folder for transfer, turning each image sequence into one video (H.264, or FFV1 for exact 16-bit) with a per-frame manifest CSV. Other files are copied as-is, and the source folder is never modified. Frame counts are checked with ffprobe, and the script can resume. `--restore` / `--extract` decode videos back into the original frame files. |
| `compress_v2.py` | `compress.py` plus lock files so several runs can go at once, and `--skip-cont`, `--only-cont` and `--reverse` for ordering. Each run writes its own log. |

---

## `other_modes/` — earlier acquisition variants

These put the repo root on `sys.path` so they can import `coms`. Output goes to
`other_modes/Kiethley_data/`.

| File | What it does |
|---|---|
| `main.py` | Minimal GUI: Keithley in **charge** mode, live scrolling plot, CSV log. No cameras, no relay. |
| `main_ubuntu.py` | The same minimal GUI in **current** mode (nA), with serial auto-detection for Linux `/dev/ttyUSB*` / `ttyACM*`. |
| `main_test.py` | Charge mode + **two Brio 101s** captured through ffmpeg (real MJPG). JPEGs are written as received, with no re-encode. |
| `main_current.py` | **Current** mode + two Brios (ffmpeg) + FLIR (PySpin). Charge is recovered afterwards by integrating the current. |
| `main_voltage.py` | **Voltage** mode across an RC network (R = 5.15 MΩ, C = 93.4 nF), with charge computed live as Q = C·V. Two Brios through OpenCV + FLIR. |
| `Kiethley_data/electrometer_2026-09-01_17-08-18.csv` | A small sample charge log (`time,charge` in coulombs). |

## `test_scripts/` — hardware tests and early prototypes

| File | What it does |
|---|---|
| `cam_test.py` | Opens one webcam with OpenCV/DSHOW and shows it live. q quits. |
| `cam_test_v2.py` | Records two Brios with ffmpeg and a live preview. SPACE starts or stops recording. `list` shows the dshow device names. |
| `pair_test.py` | Measures the real fps of two cameras by counting only frames whose content changed, because `grab()` on DSHOW returns True even when no new frame arrived. |
| `electro.py` | Headless Keithley charge logging for `DURATION_S` seconds, then a plot. No GUI. |
| `integrate.py` | Integrates a current-mode `electrometer.csv` (trapezoid rule) into charge. Writes `charge_integrated.csv` and a two-panel plot. |
| `integrated.py` | Early GUI: Keithley charge + two OpenCV webcams. |
| `old_main.py` | Earlier current-mode Keithley + two Brios (OpenCV) + FLIR logger, with charge integrated live. |
| `main_arduino.py` | `old_main.py` plus Arduino relay control of the lamps. `main.py` at the top level was built from this. |
| `main_no_cam.py` | Keithley charge mode + Arduino relay buttons, no cameras. |
| `plot_main.py` | Plots a single `electrometer.csv` (charge in pC). By default it uses the newest one. |
| `plot_blocks.py` | Splits a charge log at `SPLIT_TIMES` into labeled blocks (Idle / Fan On / Beads In …). Each block gets its own plot, an optional fit, and optional 10 s zoom windows. |
| `spike_review.py` | Finds charge spikes (|dq/dt| outliers) in a time window, copies the camera frames around each one and makes a movie of the block. |
| `make_clip.py` | Makes an MP4 of cam0/cam1 frames between `CLIP_START` and `CLIP_END`, per camera and side by side. |
| `identity_test.png` | Test image. |
