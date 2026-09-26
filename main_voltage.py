"""UNTESTED: voltage-mode electrometer + FLIR Grasshopper3 + two Logitech
Brio 101s, automated lights.

Changes from main_flir_dual_cont20_test.py:

  1. Keithley in VOLTAGE mode, medium rate (NPLC 1), reading the voltage on
     the board's 1 nF capacitor.  Q = C * V, so with CAP_F = 1 nF,
     1 mV = 1 pC.  Raw volts are logged; convert to charge later.

  2. Cameras: one FLIR Grasshopper3 (PySpin) + two Logitech Brio 101s.
     Each Brio is read through an ffmpeg subprocess (dshow, MJPEG), because
     OpenCV's DirectShow backend can't force MJPG on the Brio 101 and it
     falls back to ~5 fps YUY2.

  3. What each lamp does.  LAMP_CMDS[0] is the lamp that comes on first.

        Set by FLIR_CACHE_LAMPS / BRIO_CACHE_LAMPS / BRIO_CONT_LAMPS:
        either lamp      FLIR + both Brios keep RAM ring buffers.  A blip,
                         or the Save-burst button, dumps PRE_S before +
                         POST_ROLL_S after from every camera.
                         (Continuous Brio saving is off: BRIO_CONT_LAMPS
                         is empty.  Put a lamp in it to bring it back.)
        both off         Nothing is cached or saved.

     Caching and continuous saving follow the RELAY state, so they behave
     the same under automation and under the manual lamp buttons.

  4. The blip shape in voltage mode is unknown, so:
        - "Auto trigger" checkbox turns the detector on/off live
        - "Save burst" button (or the M key) dumps the ring buffers now,
          without touching the light cycle
        - thresholds are in mV (numerically equal to pC at 1 nF, so the
          old charge-mode numbers carry straight over)

Output:
    E:/Ben Christensen/FLIES/session_<stamp>/
        meta.txt
        electrometer.csv          time,voltage_V,trigger
        events.csv                event,time,source,lamp,rec_cams,frames
        flir/e001_0000.png ...    blip bursts from the FLIR  (16-bit PNG)
        flir_frames.csv           time,event,filename
        brio1/e001_0000.jpg ...   blip bursts from each Brio (lamp A)
        brio2/e001_0000.jpg ...
        brio1_frames.csv          time,event,filename
        brio2_frames.csv
        brio1_cont/p001_000000.jpg ...  continuous (lamp B), p = phase #
        brio2_cont/p001_000000.jpg ...  (same phase numbers on both)
        brio1_cont_frames.csv     time,phase,lamp,filename  (unsorted -
        brio2_cont_frames.csv                           writer threads)
"""

import math
import platform
import queue
import shutil
import subprocess
import threading
import time
import tkinter as tk
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path
from tkinter import ttk

import cv2
import numpy as np
import serial
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from coms import port

# ============ CONFIG ============
ROOT_FOLDER = Path(r'E:\Ben Christensen\FLIES')
ROOT_FOLDER.mkdir(parents=True, exist_ok=True)

IS_WINDOWS = platform.system() == 'Windows'

# --- electrometer (VOLTAGE mode) ---
DELAY_MS      = 5
NPLC          = 1             # medium rate (front panel MED = 1 PLC)
VOLT_RANGE    = 2             # V, fixed range.  None = auto-range, but
                              # auto-range steps can look like jumps.
CAP_F         = 1e-9          # board capacitor, for Q = C * V later
TO_MV         = 1e3           # V -> mV for the plot and the trigger
SERIAL_PORT   = port('keithley')
BAUDRATE      = 9600
PLOT_WINDOW_S = 10
ECHO_RAW      = False

# Sent at the start of each light phase (None = send nothing).
# In voltage mode zero-check does NOT discharge the external 1 nF cap, so the
# old charge-mode ZCHK reset no longer re-zeroes anything useful.  If you want
# the reading re-zeroed on the current value each phase, try (untested here):
#   RESET_CMD = b":CALC2:NULL:ACQ; :CALC2:NULL:STAT ON\n"
RESET_CMD = None

# --- Arduino relay controller ---
RELAY_ENABLED = True
RELAY_PORT    = port('arduino')
RELAY_BAUD    = 9600
LAMP_CMDS     = ('a', 'b')    # cycle order; the GUI picks which starts
LAMPS_OFF_CMD = 'o'
LAMP_NAMES    = {'a': 'gnd', 'b': 'acrylic'}
FIRST_LAMP    = 'a'           # default for the "Start with" buttons

# --- blip detector (units: mV; at 1 nF, 1 mV = 1 pC) ---
TRIGGER_ON       = True       # starting state of the "Auto trigger" box
RISE_MV          = 1.0        # swing inside the sliding window
TRIGGER_WINDOW_S = 5.0
JUMP_MV          = 1.0        # step between consecutive samples
COOLDOWN_S       = 10.0       # after ANY event (trigger or M press), the
                              # auto trigger ignores the signal this long

# --- light cycle ---
AUTOMATION_ON  = True         # Start also starts the lamp cycle
WARMUP_S       = 60           # let the Keithley settle before arming
DARK_S         = 20 * 60      # both lamps off between phases
LIGHT_LINGER_S = 30           # keep the lamp on this long after a blip
LIGHT_MAX_S    = None         # give up waiting after this many s
                              # (None = wait forever for a blip)

# --- what each lamp does (lamp letters from LAMP_CMDS) ---
FLIR_CACHE_LAMPS = ('a', 'b')     # FLIR ring buffer under these lamps
BRIO_CACHE_LAMPS = ('a', 'b')     # Brio ring buffers under these lamps
BRIO_CONT_LAMPS  = ()             # Brio continuous 20 fps under these
                                  # (empty = off; e.g. ('b',) to bring back)

# --- ring buffers (FLIR and Brios share the same window) ---
PRE_S         = 10.0          # seconds kept before a blip
POST_ROLL_S   = 10.0          # seconds kept after a blip, then dump
CACHE_SECONDS = PRE_S + POST_ROLL_S

# --- FLIR Grasshopper3 (PySpin) ---
FLIR_ENABLED  = True
FLIR_FPS      = 60
FLIR_ROI      = (1920, 1080)  # centered crop (w, h), or None for full sensor
FLIR_MONO16   = True          # False -> Mono8, halves RAM
FLIR_SAVE_FMT = '.png'
PNG_LEVEL     = 1             # PNG compression 0-9; 1 is fast

# --- Logitech Brio 101s (ffmpeg dshow subprocess, continuous save only) ---
# List device names:   ffmpeg -hide_banner -list_devices true -f dshow -i dummy
# List its modes:      ffmpeg -hide_banner -f dshow -list_options true -i video="Brio 101"
# Both Brios show up under the same name, so they're told apart by
# -video_device_number (0 = first one Windows lists, 1 = second).  If they
# come up swapped, swap the numbers.
FFMPEG = r'E:\Ben Christensen\ffmpeg-2026-09-21-git-a9cbcc2bbb-essentials_build\ffmpeg-2026-09-21-git-a9cbcc2bbb-essentials_build\bin\ffmpeg.exe'
BRIOS = [
    # (name,   dshow name,  device number)
    ('brio1', 'Brio 101', 0),
    ('brio2', 'Brio 101', 1),
]
BRIO_SIZE        = (1280, 720)    # must be an MJPEG mode from -list_options
BRIO_FPS         = 30             # request 30; the continuous save picks 20
BRIO_SAVE_FMT    = '.jpg'
JPEG_QUALITY     = 95

# --- continuous save: both Brios under BRIO_CONT_LAMPS ---
CONT_ENABLED   = True
CONT_FPS       = 20           # Brio runs 30 fps, so 20 fps is an AVERAGE
                              # (frames picked on a 50 ms clock).  15 or 30
                              # would give perfectly even spacing.
CONT_WRITERS   = 3
CONT_QUEUE_MAX = 1200         # frames buffered before dropping
CONT_FMT       = '.jpg'
CONT_SIZE      = None         # (w, h) to downscale to, or None

# --- preview ---
PREVIEW_W  = 720
PREVIEW_MS = 300
# ================================

CONT_PERIOD = 1.0 / CONT_FPS

_range = (b":SENS:VOLT:RANG:AUTO ON; " if VOLT_RANGE is None
          else f":SENS:VOLT:RANG {VOLT_RANGE}; ".encode())
SETUP_CMD = (b"*RST; :SYST:ZCH ON; :SENS:FUNC 'VOLT'; " + _range
             + f":SENS:VOLT:NPLC {NPLC}; ".encode()
             + b":FORM:ELEM READ; :SYST:ZCH OFF\n")

try:
    import PySpin
    HAS_PYSPIN = True
except ImportError:
    HAS_PYSPIN = False
    if FLIR_ENABLED:
        print('[FLIR] PySpin not installed - FLIR disabled')
        FLIR_ENABLED = False


def flir_ram_mb():
    w, h = FLIR_ROI if FLIR_ROI else (2448, 2048)
    return CACHE_SECONDS * FLIR_FPS * w * h * (2 if FLIR_MONO16 else 1) / 1e6


def brio_ram_mb():
    w, h = BRIO_SIZE
    return CACHE_SECONDS * BRIO_FPS * w * h * 3 / 1e6


def img_params(ext):
    if ext == '.png':
        return [cv2.IMWRITE_PNG_COMPRESSION, PNG_LEVEL]
    if ext in ('.jpg', '.jpeg'):
        return [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    return []


# ------------------------------------------------------------- blip detector

class SignalTrigger:
    """Fires once when EITHER condition is met (units: mV):

      1. signal swings by >= thresh inside a sliding window
      2. signal jumps by >= jump between two consecutive samples

    update() returns True for exactly one sample per armed cycle.
    `enabled` can be flipped live from the GUI.
    """

    def __init__(self, window_s=TRIGGER_WINDOW_S, thresh=RISE_MV,
                 jump=JUMP_MV):
        self.window_s = window_s
        self.thresh = thresh
        self.jump = jump
        self.enabled = TRIGGER_ON
        self.hold_until = float('-inf')   # cooldown: ignore until this t
        self._buf = deque()
        self._prev = None
        self._armed = False
        self.fired_at = None

    def arm(self):
        self._buf.clear()
        self._prev = None
        self._armed = True
        self.fired_at = None

    def disarm(self):
        self._buf.clear()
        self._prev = None
        self._armed = False

    def reset(self):
        self._buf.clear()
        self._prev = None

    def update(self, t, x):
        if not self.enabled:
            # stay clear so re-enabling starts from a fresh window
            if self._buf:
                self._buf.clear()
            self._prev = None
            return False
        if not self._armed or math.isnan(x):
            return False
        if t < self.hold_until:
            # cooling down: start a fresh window once it's over, so the
            # event we just had can't count toward the next one
            self._buf.clear()
            self._prev = None
            return False

        if self._prev is not None and abs(x - self._prev) >= self.jump:
            self.fired_at = t
            self._armed = False
            self._prev = x
            return True
        self._prev = x

        self._buf.append((t, x))
        while self._buf and t - self._buf[0][0] > self.window_s:
            self._buf.popleft()
        if t - self._buf[0][0] < self.window_s * 0.9:
            return False

        vals = [v for _, v in self._buf]
        if x - min(vals) >= self.thresh or max(vals) - x >= self.thresh:
            self.fired_at = t
            self._armed = False
            return True
        return False


# ----------------------------------------------------------- electrometer

class AcquisitionThread(threading.Thread):
    """Owns the serial port.  Pushes (t, V, trigger) samples to a queue."""

    def __init__(self, port, baudrate, delay_ms, out_queue, filepath, t0,
                 trigger):
        super().__init__(daemon=True)
        self.port = port
        self.baudrate = baudrate
        self.delay_s = delay_ms / 1000
        self.out_queue = out_queue
        self.filepath = filepath
        self.t0 = t0
        self.trigger = trigger
        self.cmd_queue = queue.Queue()
        self._stop_event = threading.Event()
        self.error = None

    def stop(self):
        self._stop_event.set()

    def run(self):
        try:
            with serial.Serial(self.port, self.baudrate, timeout=5) as em, \
                 open(self.filepath, 'w') as f:

                f.write('time,voltage_V,trigger\n')
                f.flush()
                em.write(SETUP_CMD)

                while not self._stop_event.is_set():
                    t = time.perf_counter() - self.t0
                    em.write(b'READ?\r')
                    raw = em.readline()
                    if ECHO_RAW:
                        print(repr(raw))
                    try:
                        val = float(raw)
                        if abs(val) > 1e30:     # overflow (9.9e37)
                            val = np.nan
                    except ValueError:
                        val = np.nan

                    trig = self.trigger.update(t, val * TO_MV)

                    f.write(f'{t},{val},{int(trig)}\n')
                    f.flush()
                    self.out_queue.put((t, val, trig))

                    try:
                        while True:
                            cmd = self.cmd_queue.get_nowait()
                            em.write(cmd)
                            self.trigger.reset()
                            print(f'[keithley] sent {cmd}')
                    except queue.Empty:
                        pass

                    time.sleep(self.delay_s)

        except Exception:
            self.error = traceback.format_exc()
            print(self.error)

        self.out_queue.put(None)


# ----------------------------------------------------------- cameras

class CameraBase(threading.Thread):
    """Ring buffer / burst / continuous-save logic shared by both cameras.

    Subclasses grab frames and call self._on_frame(raw, preview).

    - set_caching(True)  starts the ring buffer (cleared fresh each time it
                         goes off -> on).  Trimmed by TIME to CACHE_SECONDS,
                         so it holds the same span whatever the real fps.
    - burst(n)           schedules a dump POST_ROLL_S from now.  Caching is
                         left on afterwards; the relay state decides that.
    - cont_set(True)     saves one frame every CONT_PERIOD to cont_queue,
                         never blocking (drops and counts if full).
    """

    save_ext = '.png'

    def __init__(self, name, nominal_fps):
        super().__init__(daemon=True)
        self.cam_name = name
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest_frame = None

        self.cache = deque(maxlen=int(CACHE_SECONDS * nominal_fps * 1.5) + 10)
        self._cache_lock = threading.Lock()
        self.caching = threading.Event()

        self._dump_lock = threading.Lock()
        self.save_queue = None
        self.t0 = 0.0
        self._dump_at = None
        self._dump_event = 0
        self.saved = 0
        self.received = 0
        self.resolution = None
        self.fmt_str = ''
        self.fps = 0.0
        self._fps_t = time.perf_counter()
        self._fps_n = 0
        self.error = None

        self.cont_queue = None
        self.cont_active = threading.Event()
        self._cont_phase = 0
        self._cont_lamp = '-'
        self._cont_k = 0
        self._cont_next = 0.0
        self.cont_queued = 0
        self.cont_dropped = 0

    # ---------- control (called from the GUI thread) ----------

    def stop(self):
        self._stop_event.set()

    def usable(self):
        return self.is_alive() and self.error is None

    def arm(self, save_queue, t0):
        self.caching.clear()
        self.save_queue = save_queue
        self.t0 = t0
        self.saved = 0
        with self._dump_lock:
            self._dump_at = None
        with self._cache_lock:
            self.cache.clear()

    def cont_arm(self, cont_queue):
        self.cont_active.clear()
        self.cont_queue = cont_queue
        self.cont_queued = 0
        self.cont_dropped = 0

    def set_caching(self, on):
        if on:
            if not self.caching.is_set():
                if self._dump_at is None:       # don't wipe a pending dump
                    with self._cache_lock:
                        self.cache.clear()
                self.caching.set()
        else:
            self.caching.clear()

    def cont_set(self, on, phase=0, lamp='-'):
        if on:
            self._cont_phase = phase
            self._cont_lamp = lamp
            self._cont_k = 0
            self._cont_next = 0.0
            self.cont_active.set()
        else:
            self.cont_active.clear()

    def cache_len(self):
        with self._cache_lock:
            return len(self.cache)

    def burst(self, event_n):
        """Schedule a dump.  Returns False if one is already pending."""
        with self._dump_lock:
            if self._dump_at is not None:
                print(f'[{self.cam_name}] dump for event {self._dump_event} '
                      f'still pending - event {event_n} rides along with it')
                return False
            self._dump_event = event_n
            self._dump_at = time.perf_counter() + POST_ROLL_S
            return True

    def flush_pending(self):
        """Dump now if a burst is scheduled.  Safe from any thread."""
        with self._dump_lock:
            if self._dump_at is None:
                return
            self._dump_at = None
            ev = self._dump_event
        with self._cache_lock:
            frames = list(self.cache)
            self.cache.clear()
        if self.save_queue is None:
            return
        for k, (t, raw) in enumerate(frames):
            self.save_queue.put((t, raw, ev, k))
        self.saved += len(frames)
        span = frames[-1][0] - frames[0][0] if len(frames) > 1 else 0
        print(f'[{self.cam_name}] event {ev}: {len(frames)} frames '
              f'({span:.1f} s) queued')

    # ---------- display ----------

    def latest_frame_rgb(self):
        with self._lock:
            f = self._latest_frame
        if f is None:
            return None
        code = cv2.COLOR_GRAY2RGB if f.ndim == 2 else cv2.COLOR_BGR2RGB
        return cv2.cvtColor(f, code)

    def status_line(self):
        if self.error:
            return 'ERROR - see console'
        if self.received == 0:
            return 'waiting for stream...'
        s = f'{self.fps:.1f} fps'
        if self.resolution:
            s += f'  {self.resolution[0]}x{self.resolution[1]} {self.fmt_str}'
        if self.caching.is_set():
            with self._cache_lock:
                n = len(self.cache)
                span = (self.cache[-1][0] - self.cache[0][0]) if n > 1 else 0
            s += f'  cache {n} ({span:.0f} s)'
        else:
            s += '  cache off'
        if self._dump_at is not None:
            s += '  DUMP PENDING'
        if self.saved:
            s += f'  {self.saved} saved'
        if self.cont_queue is not None:
            state = 'ON' if self.cont_active.is_set() else 'off'
            s += (f'\ncont {state}: {self.cont_queued} queued, '
                  f'{self.cont_dropped} dropped, '
                  f'backlog {self.cont_queue.qsize()}')
        return s

    # ---------- per-frame (camera thread) ----------

    def _on_frame(self, raw, preview):
        now = time.perf_counter()
        t = now - self.t0
        self.received += 1
        with self._lock:
            self._latest_frame = preview

        self._fps_n += 1
        if now - self._fps_t >= 1.0:
            self.fps = self._fps_n / (now - self._fps_t)
            self._fps_n = 0
            self._fps_t = now

        if self.caching.is_set():
            with self._cache_lock:
                self.cache.append((t, raw))
                while self.cache and t - self.cache[0][0] > CACHE_SECONDS:
                    self.cache.popleft()

        if self.cont_active.is_set() and self.cont_queue is not None \
                and now >= self._cont_next:
            try:
                self.cont_queue.put_nowait(
                    (t, raw, self._cont_phase, self._cont_k,
                     self._cont_lamp))
                self.cont_queued += 1
            except queue.Full:
                self.cont_dropped += 1
            self._cont_k += 1            # advances on a drop, so gaps show
            nxt = self._cont_next + CONT_PERIOD
            self._cont_next = nxt if nxt > now else now + CONT_PERIOD

        if self._dump_at is not None and now >= self._dump_at:
            self.flush_pending()


class SpinnakerCamera(CameraBase):
    """FLIR Grasshopper3 via PySpin."""

    save_ext = FLIR_SAVE_FMT

    def __init__(self, name, pyspin_cam):
        super().__init__(name, FLIR_FPS)
        self._pyspin_cam = pyspin_cam
        self.pixel_fmt = None

    def _configure(self, nodemap):
        node_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        node_mode.SetIntValue(node_mode.GetEntryByName('Continuous').GetValue())

        if FLIR_ROI:
            self._set_roi(nodemap, *FLIR_ROI)

        # Frame rate.  Newer cameras use AcquisitionFrameRateEnable; the
        # Grasshopper3 uses AcquisitionFrameRateAuto=Off plus
        # AcquisitionFrameRateEnabled (with a 'd').  Try both.
        try:
            node_auto = PySpin.CEnumerationPtr(
                nodemap.GetNode('AcquisitionFrameRateAuto'))
            if PySpin.IsAvailable(node_auto) and PySpin.IsWritable(node_auto):
                node_auto.SetIntValue(
                    node_auto.GetEntryByName('Off').GetValue())
        except PySpin.SpinnakerException:
            pass
        for en_name in ('AcquisitionFrameRateEnable',
                        'AcquisitionFrameRateEnabled'):
            try:
                node_fr_en = PySpin.CBooleanPtr(nodemap.GetNode(en_name))
                if PySpin.IsAvailable(node_fr_en) \
                        and PySpin.IsWritable(node_fr_en):
                    node_fr_en.SetValue(True)
            except PySpin.SpinnakerException:
                pass
        try:
            node_fr = PySpin.CFloatPtr(nodemap.GetNode('AcquisitionFrameRate'))
            if PySpin.IsAvailable(node_fr) and PySpin.IsWritable(node_fr):
                node_fr.SetValue(min(FLIR_FPS, node_fr.GetMax()))
                print(f'[{self.cam_name}] frame rate set to '
                      f'{node_fr.GetValue():.1f} fps')
            else:
                print(f'[{self.cam_name}] AcquisitionFrameRate not writable '
                      f'- camera will free-run')
        except PySpin.SpinnakerException as e:
            print(f'[{self.cam_name}] could not set frame rate: {e}')

        self.pixel_fmt = 'Mono8'
        try:
            node_pf = PySpin.CEnumerationPtr(nodemap.GetNode('PixelFormat'))
            entry = node_pf.GetEntryByName('Mono16') if FLIR_MONO16 else None
            if entry is not None and PySpin.IsAvailable(entry) \
                    and PySpin.IsReadable(entry):
                node_pf.SetIntValue(entry.GetValue())
                self.pixel_fmt = 'Mono16'
            else:
                node_pf.SetIntValue(node_pf.GetEntryByName('Mono8').GetValue())
        except PySpin.SpinnakerException:
            pass
        self.fmt_str = self.pixel_fmt

    def _set_roi(self, nodemap, want_w, want_h):
        """Centered crop.  Offsets go to zero first so the resize always fits."""
        try:
            for name in ('OffsetX', 'OffsetY'):
                node = PySpin.CIntegerPtr(nodemap.GetNode(name))
                if PySpin.IsAvailable(node) and PySpin.IsWritable(node):
                    node.SetValue(0)

            for name, want in (('Width', want_w), ('Height', want_h)):
                node = PySpin.CIntegerPtr(nodemap.GetNode(name))
                if not (PySpin.IsAvailable(node) and PySpin.IsWritable(node)):
                    continue
                inc = node.GetInc() or 1
                val = min(node.GetMax(), max(node.GetMin(), want))
                node.SetValue(val - (val % inc))

            for off, dim in (('OffsetX', 'Width'), ('OffsetY', 'Height')):
                node = PySpin.CIntegerPtr(nodemap.GetNode(off))
                dnode = PySpin.CIntegerPtr(nodemap.GetNode(dim))
                if not (PySpin.IsAvailable(node) and PySpin.IsWritable(node)):
                    continue
                inc = node.GetInc() or 1
                val = max(0, (dnode.GetMax() - dnode.GetValue()) // 2)
                node.SetValue(min(node.GetMax(), val - (val % inc)))
        except PySpin.SpinnakerException as e:
            print(f'[{self.cam_name}] could not set ROI {want_w}x{want_h}: {e}')

    def run(self):
        cam = self._pyspin_cam
        if cam is None:
            self.error = 'no camera object provided'
            print(f'[{self.cam_name}] {self.error}')
            return
        try:
            self._configure(cam.GetNodeMap())
            cam.BeginAcquisition()

            while not self._stop_event.is_set():
                try:
                    image = cam.GetNextImage(1000)
                except PySpin.SpinnakerException:
                    continue
                if image.IsIncomplete():
                    image.Release()
                    continue

                w, h = image.GetWidth(), image.GetHeight()
                if self.resolution is None:
                    self.resolution = (w, h)
                    print(f'[{self.cam_name}] {w}x{h} {self.pixel_fmt}, '
                          f'ring buffer ~{flir_ram_mb():.0f} MB')

                if self.pixel_fmt == 'Mono16':
                    raw = np.frombuffer(image.GetData(), dtype=np.uint16
                                        ).reshape(h, w).copy()
                    preview = (raw >> 8).astype(np.uint8)
                else:
                    raw = np.frombuffer(image.GetData(), dtype=np.uint8
                                        ).reshape(h, w).copy()
                    preview = raw

                image.Release()
                self._on_frame(raw, preview)

            try:
                cam.EndAcquisition()
            except PySpin.SpinnakerException:
                pass

        except Exception:
            self.error = traceback.format_exc()
            print(f'[{self.cam_name}] {self.error}')


class FFmpegCamera(CameraBase):
    """Logitech Brio 101 through an ffmpeg dshow subprocess.

    ffmpeg asks the camera for MJPEG (the only way the Brio 101 does 30 fps
    over USB 2.0), decodes it, and pipes raw BGR frames to us.  Every frame
    is a fixed w*h*3 bytes, so reading is just read(n) in a loop.
    """

    save_ext = BRIO_SAVE_FMT

    def __init__(self, name, size, fps, dshow_name, device_num):
        super().__init__(name, fps)
        self.dshow_name = dshow_name
        self.device_num = device_num
        self.size = size
        self.want_fps = fps
        self.proc = None
        self._stderr_tail = deque(maxlen=20)

    def _cmd(self, exe):
        w, h = self.size
        return [exe, '-hide_banner', '-loglevel', 'warning',
                '-f', 'dshow',
                '-rtbufsize', '256M',
                '-vcodec', 'mjpeg',
                '-video_size', f'{w}x{h}',
                '-framerate', str(self.want_fps),
                '-video_device_number', str(self.device_num),
                '-i', f'video={self.dshow_name}',
                '-an',
                '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-']

    def _pump_stderr(self):
        """Echo ffmpeg warnings, and keep the pipe from filling up."""
        try:
            for line in iter(self.proc.stderr.readline, b''):
                txt = line.decode(errors='replace').rstrip()
                if txt:
                    self._stderr_tail.append(txt)
                    print(f'[{self.cam_name}/ffmpeg] {txt}')
        except Exception:
            pass

    def stop(self):
        super().stop()
        p = self.proc
        if p is not None and p.poll() is None:
            try:
                p.terminate()
            except Exception:
                pass

    def run(self):
        exe = shutil.which(FFMPEG) or (FFMPEG if Path(FFMPEG).exists()
                                       else None)
        if exe is None:
            self.error = (f'ffmpeg not found ({FFMPEG!r}) - install it or '
                          f'set FFMPEG to the full path of ffmpeg.exe')
            print(f'[{self.cam_name}] {self.error}')
            return

        w, h = self.size
        nbytes = w * h * 3
        cmd = self._cmd(exe)
        print(f'[{self.cam_name}] starting: {" ".join(cmd)}')
        try:
            flags = subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL, bufsize=nbytes * 4,
                creationflags=flags)
        except Exception:
            self.error = traceback.format_exc()
            print(f'[{self.cam_name}] {self.error}')
            return

        threading.Thread(target=self._pump_stderr, daemon=True).start()
        self.fmt_str = 'BGR (ffmpeg MJPEG)'

        try:
            while not self._stop_event.is_set():
                buf = self.proc.stdout.read(nbytes)
                if len(buf) < nbytes:
                    if not self._stop_event.is_set():
                        tail = ' | '.join(self._stderr_tail) or 'no output'
                        self.error = f'ffmpeg stream ended: {tail}'
                        print(f'[{self.cam_name}] {self.error}')
                    break
                frame = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
                if self.resolution is None:
                    self.resolution = (w, h)
                    print(f'[{self.cam_name}] {w}x{h} via ffmpeg MJPEG, '
                          f'first frame received')
                self._on_frame(frame, frame)
        except Exception:
            if not self._stop_event.is_set():
                self.error = traceback.format_exc()
                print(f'[{self.cam_name}] {self.error}')
        finally:
            if self.proc is not None and self.proc.poll() is None:
                try:
                    self.proc.terminate()
                    self.proc.wait(timeout=3)
                except Exception:
                    pass


# ----------------------------------------------------------- frame writers

class FrameWriter(threading.Thread):
    """Writes blip bursts to disk.  One file per frame, named by event."""

    def __init__(self, in_queue, out_dir, index_path, ext):
        super().__init__(daemon=True)
        self.in_queue = in_queue
        self.out_dir = out_dir
        self.index_path = index_path
        self.ext = ext
        self.params = img_params(ext)
        self.written = 0
        self.error = None

    def run(self):
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            with open(self.index_path, 'w') as idx:
                idx.write('time,event,filename\n')
                while True:
                    item = self.in_queue.get()
                    if item is None:
                        break
                    t, data, event, k = item
                    fname = f'e{event:03d}_{k:04d}{self.ext}'
                    cv2.imwrite(str(self.out_dir / fname), data, self.params)
                    idx.write(f'{t:.6f},{event},{fname}\n')
                    self.written += 1
                    if self.written % 20 == 0:
                        idx.flush()
                idx.flush()
        except Exception:
            self.error = traceback.format_exc()
            print(self.error)


class ContinuousWriterPool:
    """N threads draining one bounded queue.  Shared index file under a lock.
    Sort by filename, not by index row, to get frames in order."""

    def __init__(self, in_queue, out_dir, index_path, n_threads=CONT_WRITERS,
                 ext=CONT_FMT):
        self.in_queue = in_queue
        self.out_dir = out_dir
        self.index_path = index_path
        self.n_threads = max(1, n_threads)
        self.ext = ext
        self.params = img_params(ext)
        self._idx = None
        self._idx_lock = threading.Lock()
        self._threads = []
        self.written = 0
        self.error = None

    def start(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._idx = open(self.index_path, 'w')
        self._idx.write('time,phase,lamp,filename\n')
        for i in range(self.n_threads):
            th = threading.Thread(target=self._run, daemon=True,
                                  name=f'cont_writer_{i}')
            th.start()
            self._threads.append(th)

    def _run(self):
        try:
            while True:
                item = self.in_queue.get()
                if item is None:
                    break
                t, data, phase, k, lamp = item
                if CONT_SIZE is not None:
                    data = cv2.resize(data, CONT_SIZE,
                                      interpolation=cv2.INTER_AREA)
                fname = f'p{phase:03d}_{k:06d}{self.ext}'
                path = str(self.out_dir / fname)
                if self.ext == '.npy':
                    np.save(path, data)
                else:
                    cv2.imwrite(path, data, self.params)
                with self._idx_lock:
                    if self._idx is not None:
                        self._idx.write(
                            f'{t:.6f},{phase},{lamp},{fname}\n')
                        self.written += 1
                        if self.written % 50 == 0:
                            self._idx.flush()
        except Exception:
            self.error = traceback.format_exc()
            print(f'[cont] writer error:\n{self.error}')

    def finish(self, timeout=300):
        backlog = self.in_queue.qsize()
        if backlog:
            print(f'[cont] draining {backlog} queued frames...')
        for _ in self._threads:
            self.in_queue.put(None)
        for th in self._threads:
            th.join(timeout=timeout)
        with self._idx_lock:
            if self._idx is not None:
                self._idx.flush()
                self._idx.close()
                self._idx = None
        print(f'[cont] {self.written} continuous frames written')


# ------------------------------------------------------------ light cycle

class LightSequencer:
    """lamp 0 -> blip -> linger -> dark -> lamp 1 -> blip -> ...

    Only switches lamps and arms the trigger.  What the cameras do follows
    from the relay state (ElectrometerApp._apply_lamp).
    """

    def __init__(self, app):
        self.app = app
        self.running = False
        self.state = 'idle'
        self.lamp_i = 0
        self._job = None
        self.phase_start = None      # monotonic time the current state began
        self.deadline = None         # monotonic time it ends (None = open)

    def _set_state(self, text, duration_s=None):
        self.state = text
        self.phase_start = time.monotonic()
        self.deadline = (self.phase_start + duration_s
                         if duration_s else None)

    def clock_text(self):
        """'19:42 left' for timed states, '3:12 elapsed' for open ones."""
        if self.phase_start is None:
            return ''
        now = time.monotonic()
        if self.deadline is not None:
            s = max(0, int(self.deadline - now + 0.999))
            return f'{s // 60:d}:{s % 60:02d} left'
        s = int(now - self.phase_start)
        return f'{s // 60:d}:{s % 60:02d} elapsed'

    def start(self):
        self.running = True
        first = self.app.first_lamp_var.get()
        self.lamp_i = LAMP_CMDS.index(first) if first in LAMP_CMDS else 0
        print(f'[auto] cycle starts with lamp {first.upper()} '
              f'({LAMP_NAMES.get(first, "")})')
        self._begin_light()

    def stop(self):
        self.running = False
        self.state = 'idle'
        self.phase_start = None
        self.deadline = None
        self._cancel()
        self.app.trigger.disarm()

    def _cancel(self):
        if self._job is not None:
            self.app.root.after_cancel(self._job)
            self._job = None

    def lamp(self):
        return LAMP_CMDS[self.lamp_i]

    def _begin_light(self):
        self._cancel()
        if not self.running:
            return
        if RESET_CMD and self.app.acq_thread is not None:
            self.app.acq_thread.cmd_queue.put(RESET_CMD)
        lamp = LAMP_CMDS[self.lamp_i]
        jobs = []
        if lamp in FLIR_CACHE_LAMPS:
            jobs.append('FLIR cache')
        if lamp in BRIO_CACHE_LAMPS:
            jobs.append('Brio cache')
        if lamp in BRIO_CONT_LAMPS:
            jobs.append(f'Brio {CONT_FPS} fps')
        role = ' + '.join(jobs) or 'nothing saved'
        wait = ('waiting for blip' if self.app.trigger.enabled
                else 'auto trigger OFF')
        self._set_state(f'lamp {lamp.upper()} ({role}), {wait}', LIGHT_MAX_S)
        self.app.send_relay(lamp, from_auto=True)
        self.app.trigger.arm()
        if LIGHT_MAX_S:
            self._job = self.app.root.after(int(LIGHT_MAX_S * 1000),
                                            self._light_timeout)

    def _light_timeout(self):
        print(f'[auto] no blip within {LIGHT_MAX_S} s - moving on')
        self._begin_dark()

    def advance(self):
        """A blip fired.  Linger with the lamp on, then go dark."""
        if not self.running:
            return
        self._cancel()
        self.app.trigger.disarm()
        if LIGHT_LINGER_S > 0:
            self._set_state(f'lamp {self.lamp().upper()} lingering',
                            LIGHT_LINGER_S)
            self._job = self.app.root.after(int(LIGHT_LINGER_S * 1000),
                                            self._begin_dark)
        else:
            self._begin_dark()

    def _begin_dark(self):
        self._cancel()
        self.app.trigger.disarm()
        self.app.send_relay(LAMPS_OFF_CMD, from_auto=True)
        self.lamp_i = (self.lamp_i + 1) % len(LAMP_CMDS)
        self._set_state(f'DARK - next: lamp {self.lamp().upper()}', DARK_S)
        self._job = self.app.root.after(int(DARK_S * 1000), self._begin_light)


# ------------------------------------------------------------------- GUI

class ElectrometerApp:

    def __init__(self, root):
        self.root = root
        self.root.title(f'Electrometer (voltage) + FLIR + Brio  |  '
                        f'cache {PRE_S:g}+{POST_ROLL_S:g} s: FLIR on '
                        f'{"/".join(FLIR_CACHE_LAMPS).upper()}, Brios on '
                        f'{"/".join(BRIO_CACHE_LAMPS).upper()}  |  Brios '
                        f'{CONT_FPS} fps on '
                        f'{"/".join(BRIO_CONT_LAMPS).upper()}')

        self.acq_thread = None
        self.data_queue = None
        self.recording = False
        self.t0 = 0.0
        self.t_vec = []
        self.v_vec = []
        self.session_dir = None
        self.events_path = None
        self.blip_n = 0
        self.lamp_state = LAMPS_OFF_CMD

        self.trigger = SignalTrigger()
        self.sequencer = LightSequencer(self)

        self.burst_writers = {}        # cam_name -> (queue, FrameWriter)
        self.cont_pools = {}           # brio name -> ContinuousWriterPool
        self.cont_phase = 0
        self.cont_lamp = None          # lamp the current phase is under

        # --- Arduino relay ---
        self.relay_ser = None
        if RELAY_ENABLED:
            try:
                self.relay_ser = serial.Serial(RELAY_PORT, RELAY_BAUD,
                                               timeout=1)
                time.sleep(2)       # Arduino resets on serial open
                print(f'[relay] connected on {RELAY_PORT}')
            except Exception as e:
                print(f'[relay] could not open {RELAY_PORT}: {e}')

        # --- controls ---
        controls = ttk.Frame(root, padding=8)
        controls.pack(side=tk.TOP, fill=tk.X)

        self.start_btn = ttk.Button(controls, text='Start', command=self.start)
        self.start_btn.pack(side=tk.LEFT, padx=4)

        self.stop_btn = ttk.Button(controls, text='Stop', command=self.stop,
                                   state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=4)

        self.auto_var = tk.BooleanVar(value=AUTOMATION_ON)
        ttk.Checkbutton(controls, text='Auto lights', variable=self.auto_var,
                        command=self._on_auto_toggle).pack(side=tk.LEFT, padx=8)

        # which lamp the auto cycle starts with (read each time it starts)
        self.first_lamp_var = tk.StringVar(value=FIRST_LAMP)
        start_frame = ttk.LabelFrame(controls, text='Auto lights start with',
                                     padding=(6, 0))
        start_frame.pack(side=tk.LEFT, padx=4)
        for cmd in LAMP_CMDS:
            ttk.Radiobutton(
                start_frame,
                text=f'lamp {cmd.upper()} ({LAMP_NAMES.get(cmd, "")})',
                variable=self.first_lamp_var, value=cmd,
            ).pack(side=tk.LEFT, padx=2)

        self.trig_var = tk.BooleanVar(value=TRIGGER_ON)
        ttk.Checkbutton(controls, text='Auto trigger', variable=self.trig_var,
                        command=self._on_trig_toggle).pack(side=tk.LEFT, padx=4)

        ttk.Button(controls, text='Save burst (M)',
                   command=self.manual_burst).pack(side=tk.LEFT, padx=8)
        self.root.bind('<m>', lambda e: self.manual_burst())
        self.root.bind('<M>', lambda e: self.manual_burst())

        self.status_var = tk.StringVar(value='Idle')
        ttk.Label(controls, textvariable=self.status_var).pack(side=tk.LEFT,
                                                               padx=12)

        relay_frame = ttk.LabelFrame(controls, text='Relays (manual)',
                                     padding=4)
        relay_frame.pack(side=tk.RIGHT, padx=8)
        for label, cmd in ((f'Lamp A ({LAMP_NAMES.get("a", "")})', 'a'),
                           (f'Lamp B ({LAMP_NAMES.get("b", "")})', 'b'),
                           ('Both Off', 'o')):
            ttk.Button(relay_frame, text=label,
                       command=lambda c=cmd: self.send_relay(c)
                       ).pack(side=tk.LEFT, padx=2)
        self.relay_var = tk.StringVar(
            value='connected' if self.relay_ser else 'not connected')
        ttk.Label(relay_frame, textvariable=self.relay_var).pack(side=tk.LEFT,
                                                                 padx=6)

        # --- body: plot left, previews right ---
        body = ttk.Frame(root)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(7, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('time [s]')
        self.ax.set_ylabel(f'voltage [mV]   '
                           f'(1 mV = {CAP_F * 1e9:g} pC at '
                           f'{CAP_F * 1e9:g} nF)')
        self.line, = self.ax.plot([], [], 'ro-', markersize=3)
        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=body)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH,
                                         expand=True)

        cam_panel = ttk.Frame(body, padding=4)
        cam_panel.pack(side=tk.RIGHT, fill=tk.Y)

        # cam_name -> (label, status_var, preview width)
        self.previews = {}

        def add_preview(parent, name, title, width, side=tk.TOP):
            fr = ttk.LabelFrame(parent, text=title, padding=4)
            fr.pack(side=side, fill=tk.X, pady=4, padx=2)
            lbl = tk.Label(fr, background='#222')
            lbl.pack()
            sv = tk.StringVar(value=f'{name} disabled')
            ttk.Label(fr, textvariable=sv, wraplength=width).pack(anchor='w')
            self.previews[name] = (lbl, sv, width)

        add_preview(cam_panel, 'flir',
                    f'FLIR (cache {PRE_S:g} s + {POST_ROLL_S:g} s on lamp '
                    f'{"/".join(FLIR_CACHE_LAMPS).upper()})', PREVIEW_W)
        brio_row = ttk.Frame(cam_panel)
        brio_row.pack(side=tk.TOP, fill=tk.X)
        for name, _, dev in BRIOS:
            add_preview(brio_row, name,
                        f'{name} (dev {dev}: cache on '
                        f'{"/".join(BRIO_CACHE_LAMPS).upper()}, '
                        f'{CONT_FPS} fps on '
                        f'{"/".join(BRIO_CONT_LAMPS).upper()})',
                        PREVIEW_W // 2 - 8, side=tk.LEFT)

        self.auto_status_var = tk.StringVar(value='automation idle')
        ttk.Label(cam_panel, textvariable=self.auto_status_var).pack(
            anchor='w', pady=(4, 0))
        self.clock_var = tk.StringVar(value='')
        tk.Label(cam_panel, textvariable=self.clock_var,
                 font=('Segoe UI', 20, 'bold')).pack(anchor='w', pady=(0, 4))
        ttk.Label(cam_panel,
                  text=f'ring buffers ~{flir_ram_mb():.0f} MB FLIR + '
                       f'~{brio_ram_mb():.0f} MB per Brio'
                  ).pack(anchor='w')

        # --- cameras ---
        self.flir = None
        self.brios = []
        self._spin_system = None
        self._spin_cam_list = None
        self._spin_cams = []

        if FLIR_ENABLED and HAS_PYSPIN:
            try:
                self._spin_system = PySpin.System.GetInstance()
                self._spin_cam_list = self._spin_system.GetCameras()
                n = self._spin_cam_list.GetSize()
                print(f'[spinnaker] {n} camera(s) found')
                for i in range(n):
                    c = self._spin_cam_list[i]
                    c.Init()
                    nm = c.GetTLDeviceNodeMap()
                    sn = PySpin.CStringPtr(
                        nm.GetNode('DeviceSerialNumber')).GetValue()
                    model = PySpin.CStringPtr(
                        nm.GetNode('DeviceModelName')).GetValue()
                    print(f'  [{i}] {model}  S/N {sn}')
                    self._spin_cams.append(c)
                if self._spin_cams:
                    self.flir = SpinnakerCamera('flir', self._spin_cams[0])
                    self.flir.start()
            except Exception as e:
                print(f'[spinnaker] init error: {e}')

        for name, dshow_name, dev in BRIOS:
            cam = FFmpegCamera(name, BRIO_SIZE, BRIO_FPS, dshow_name, dev)
            cam.start()
            self.brios.append(cam)
            time.sleep(0.5)      # stagger the dshow opens

        self.root.protocol('WM_DELETE_WINDOW', self._on_close)
        self.root.after(PREVIEW_MS, self._update_previews)

    # ---------- camera helpers ----------

    def _cams(self):
        return ([self.flir] if self.flir is not None else []) + self.brios

    def _apply_lamp(self, cmd):
        """Make the cameras match the lamp.  Called on every relay write,
        auto or manual."""
        self.lamp_state = cmd
        if self.flir is not None:
            self.flir.set_caching(self.recording
                                  and cmd in FLIR_CACHE_LAMPS
                                  and self.flir.save_queue is not None)
        for b in self.brios:
            b.set_caching(self.recording and cmd in BRIO_CACHE_LAMPS
                          and b.save_queue is not None)
        cont_on = self.recording and CONT_ENABLED and cmd in BRIO_CONT_LAMPS
        self._set_cont(cmd if cont_on else None)

    def _set_cont(self, lamp):
        """lamp = 'a'/'b' to save continuously under that lamp, None = off.
        A new phase number starts every time the lit lamp changes, and both
        Brios share it, so p003_... in brio1 and brio2 are the same stretch.
        """
        cams = [c for c in self.brios if c.cont_queue is not None]
        if not cams:
            self.cont_lamp = None
            return
        if lamp is not None:
            if lamp != self.cont_lamp:
                self.cont_phase += 1
                self.cont_lamp = lamp
                for c in cams:
                    c.cont_set(True, self.cont_phase, lamp)
                print(f'[cont] phase {self.cont_phase} (lamp {lamp.upper()}):'
                      f' {", ".join(c.cam_name for c in cams)} saving at '
                      f'~{CONT_FPS} fps')
        elif self.cont_lamp is not None:
            self.cont_lamp = None
            for c in cams:
                c.cont_set(False)
            print(f'[cont] phase {self.cont_phase} stopped  ' + '  '.join(
                f'{c.cam_name}: {c.cont_queued} queued/{c.cont_dropped} '
                f'dropped' for c in cams))

    # ---------- relay ----------

    def send_relay(self, cmd, from_auto=False):
        """Send one character to the relay Arduino.  A manual press switches
        the automation off so the two cannot fight."""
        if not from_auto and self.sequencer.running:
            self.auto_var.set(False)
            self._on_auto_toggle()

        if self.relay_ser is None or not self.relay_ser.is_open:
            # no relay: still move the cameras, so bench tests work
            self.relay_var.set(f'not connected (sim {cmd})')
            self._apply_lamp(cmd)
            return
        try:
            self.relay_ser.write(cmd.encode())
            labels = {'a': 'Lamp A ON', 'b': 'Lamp B ON', 'o': 'Both OFF'}
            self.relay_var.set(labels.get(cmd, cmd))
        except Exception as e:
            self.relay_var.set(f'error: {e}')
            print(f'[relay] write error: {e}')
            return
        self._apply_lamp(cmd)

    # ---------- blips ----------

    def on_blip(self, t, source='trigger'):
        """Dump whichever ring buffers are running.  Only a real trigger
        advances the light cycle; a manual burst just saves."""
        self.blip_n += 1
        fired, counts = [], []
        for c in self._cams():
            if c.caching.is_set():
                n = c.cache_len()
                if c.burst(self.blip_n):
                    fired.append(c.cam_name)
                    counts.append(f'{c.cam_name}:{n}')
        self.trigger.hold_until = t + COOLDOWN_S
        rec = ';'.join(fired) or '-'
        frames = ';'.join(counts) or '0'
        self._log_event(self.blip_n, t, source, self.lamp_state, rec, frames)
        print(f'[{source}] event {self.blip_n} at t={t:.2f} s, '
              f'lamp {self.lamp_state}, rec {rec}')

        if source == 'trigger':
            if self.sequencer.running:
                self.sequencer.advance()
            else:
                self.trigger.arm()

    def manual_burst(self):
        if not self.recording:
            return
        self.on_blip(time.perf_counter() - self.t0, source='manual')

    def _log_event(self, n, t, source, lamp, rec, frames):
        if self.events_path is None:
            return
        with open(self.events_path, 'a') as f:
            f.write(f'{n},{t:.3f},{source},{lamp},{rec},{frames}\n')

    def _on_auto_toggle(self):
        if self.auto_var.get():
            if self.recording:
                self.sequencer.start()
        else:
            self.sequencer.stop()
            if self.recording:
                self.trigger.arm()
            self.auto_status_var.set('automation off (manual)')

    def _on_trig_toggle(self):
        self.trigger.enabled = self.trig_var.get()
        print(f'[trigger] auto trigger '
              f'{"ON" if self.trigger.enabled else "OFF"}')

    # ---------- start / stop ----------

    def start(self):
        self.t_vec = []
        self.v_vec = []
        self.line.set_data([], [])
        self.canvas.draw_idle()

        stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.session_dir = ROOT_FOLDER / f'session_{stamp}'
        self.session_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        self.t0 = t0

        with open(self.session_dir / 'meta.txt', 'w') as f:
            f.write(f'wall_clock_start\t{datetime.now().isoformat()}\n')
            f.write('script\tvoltage mode, FLIR + Brio\n')
            f.write(f'serial_port\t{SERIAL_PORT}\n')
            f.write(f'relay_port\t{RELAY_PORT}\n')
            f.write(f'delay_ms\t{DELAY_MS}\n')
            f.write('mode\tvoltage\n')
            f.write(f'nplc\t{NPLC}\n')
            f.write(f'volt_range\t{VOLT_RANGE}\n')
            f.write(f'cap_F\t{CAP_F}\n')
            f.write(f'reset_cmd\t{RESET_CMD}\n')
            f.write(f'trigger_on\t{self.trig_var.get()}\n')
            f.write(f'rise_mV\t{RISE_MV}\n')
            f.write(f'jump_mV\t{JUMP_MV}\n')
            f.write(f'cooldown_s\t{COOLDOWN_S}\n')
            f.write(f'trigger_window_s\t{TRIGGER_WINDOW_S}\n')
            f.write(f'lamp_cmds\t{LAMP_CMDS}\n')
            f.write(f'flir_cache_lamps\t{FLIR_CACHE_LAMPS}\n')
            f.write(f'brio_cache_lamps\t{BRIO_CACHE_LAMPS}\n')
            f.write(f'brio_cont_lamps\t{BRIO_CONT_LAMPS}\n')
            f.write(f'pre_s\t{PRE_S}\n')
            f.write(f'dark_s\t{DARK_S}\n')
            f.write(f'light_linger_s\t{LIGHT_LINGER_S}\n')
            f.write(f'light_max_s\t{LIGHT_MAX_S}\n')
            f.write(f'automation\t{self.auto_var.get()}\n')
            f.write(f'first_lamp\t{self.first_lamp_var.get()}\n')
            f.write(f'lamp_names\t{LAMP_NAMES}\n')
            f.write(f'cache_seconds\t{CACHE_SECONDS}\n')
            f.write(f'post_roll_s\t{POST_ROLL_S}\n')
            f.write(f'flir_enabled\t{self.flir is not None}\n')
            f.write(f'flir_fps\t{FLIR_FPS}\n')
            f.write(f'flir_roi\t{FLIR_ROI}\n')
            f.write(f'flir_mono16\t{FLIR_MONO16}\n')
            f.write(f'png_level\t{PNG_LEVEL}\n')
            for name, dshow_name, dev in BRIOS:
                f.write(f'{name}\t{dshow_name} (device {dev})\n')
            f.write(f'brio_size\t{BRIO_SIZE}\n')
            f.write(f'brio_fps\t{BRIO_FPS}\n')
            f.write(f'jpeg_quality\t{JPEG_QUALITY}\n')
            f.write(f'cont_enabled\t{CONT_ENABLED}\n')
            f.write(f'cont_fps\t{CONT_FPS}\n')
            f.write(f'cont_writers\t{CONT_WRITERS}\n')
            f.write(f'cont_queue_max\t{CONT_QUEUE_MAX}\n')
            f.write(f'cont_fmt\t{CONT_FMT}\n')
            f.write(f'cont_size\t{CONT_SIZE}\n')

        self.events_path = self.session_dir / 'events.csv'
        with open(self.events_path, 'w') as f:
            f.write('event,time,source,lamp,rec_cams,frames\n')

        # --- burst writers: one per camera (FLIR and both Brios) ---
        self.burst_writers = {}
        for b in self.brios:
            b.arm(None, t0)          # t0 even if it turns out to be dead
        for c in self._cams():
            if not c.usable():
                print(f'[{c.cam_name}] not recording (dead or errored)')
                continue
            q = queue.Queue()
            w = FrameWriter(q, self.session_dir / c.cam_name,
                            self.session_dir / f'{c.cam_name}_frames.csv',
                            ext=c.save_ext)
            w.start()
            c.arm(q, t0)
            self.burst_writers[c.cam_name] = (q, w)

        # --- one continuous writer pool per Brio ---
        self.cont_pools = {}
        self.cont_phase = 0
        self.cont_lamp = None
        for b in (self.brios if CONT_ENABLED and BRIO_CONT_LAMPS else []):
            if not b.usable():
                print(f'[{b.cam_name}] no continuous save (dead or errored)')
                continue
            q = queue.Queue(maxsize=CONT_QUEUE_MAX)
            pool = ContinuousWriterPool(
                q, self.session_dir / f'{b.cam_name}_cont',
                self.session_dir / f'{b.cam_name}_cont_frames.csv')
            pool.start()
            b.cont_arm(q)
            self.cont_pools[b.cam_name] = pool

        self.trigger.disarm()
        self.trigger.hold_until = float('-inf')
        self.trigger.enabled = self.trig_var.get()
        self.blip_n = 0

        self.data_queue = queue.Queue()
        self.acq_thread = AcquisitionThread(
            SERIAL_PORT, BAUDRATE, DELAY_MS, self.data_queue,
            self.session_dir / 'electrometer.csv', t0, self.trigger,
        )
        self.acq_thread.start()

        self.recording = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set(f'Recording -> {self.session_dir.name}')

        if WARMUP_S > 0:
            self.status_var.set(
                f'Warmup {WARMUP_S}s - Keithley settling...')
            print(f'[warmup] {WARMUP_S}s before arming')
            self.root.after(int(WARMUP_S * 1000), self._after_warmup)
        else:
            self._after_warmup()

        self.root.after(50, self._poll_queue)

    def _after_warmup(self):
        if not self.recording:
            return
        print('[warmup] done - arming')
        if self.auto_var.get():
            self.sequencer.start()
        else:
            self.trigger.arm()
            self._apply_lamp(self.lamp_state)

    def stop(self):
        self.recording = False
        self.sequencer.stop()
        self.send_relay(LAMPS_OFF_CMD, from_auto=True)

        if self.acq_thread is not None:
            self.acq_thread.stop()
            self.acq_thread.join(timeout=6)

        self._finish_writers()

        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

        good = int(np.count_nonzero(~np.isnan(self.v_vec))) \
            if self.v_vec else 0
        saved = ', '.join(f'{c.cam_name} {c.saved}' for c in self._cams())
        cont = ''.join(f', {b.cam_name} {b.cont_queued} saved/'
                       f'{b.cont_dropped} dropped' for b in self.brios)
        self.status_var.set(
            f'Stopped. {good}/{len(self.v_vec)} valid readings, '
            f'{self.blip_n} events, burst frames: {saved}{cont} '
            f'-> {self.session_dir.name}')

    def _finish_writers(self):
        """Dump any pending bursts, then close every writer."""
        for c in self._cams():
            c.set_caching(False)
            c.flush_pending()           # don't lose a post-roll in progress
        self._set_cont(None)
        time.sleep(0.2)

        for name, (q, w) in self.burst_writers.items():
            q.put(None)
            w.join(timeout=120)
        for c in self._cams():
            c.save_queue = None

        for pool in self.cont_pools.values():
            pool.finish()
        for b in self.brios:
            b.cont_queue = None

        self.burst_writers = {}
        self.cont_pools = {}

    # ---------- queue draining / live plot ----------

    def _poll_queue(self):
        if self.acq_thread is None:
            return

        updated = False
        ended = False
        try:
            while True:
                item = self.data_queue.get_nowait()
                if item is None:
                    ended = True
                    break
                t, val, trig = item
                self.t_vec.append(t)
                self.v_vec.append(val)
                updated = True
                if trig:
                    self.on_blip(t, source='trigger')
        except queue.Empty:
            pass

        if updated:
            t_arr = np.array(self.t_vec)
            v_arr = np.array(self.v_vec) * TO_MV

            i0 = np.searchsorted(t_arr, t_arr[-1] - PLOT_WINDOW_S)
            self.line.set_data(t_arr[i0:], v_arr[i0:])

            right = max(t_arr[-1], PLOT_WINDOW_S)
            self.ax.set_xlim(right - PLOT_WINDOW_S, right)
            self.ax.relim()
            self.ax.autoscale_view(scalex=False)
            self.canvas.draw_idle()

            saved = sum(c.saved for c in self._cams())
            cont = sum(b.cont_queued for b in self.brios)
            self.status_var.set(
                f'Recording -> {self.session_dir.name}  '
                f'({len(self.t_vec)} samples, {self.blip_n} events, '
                f'{saved} burst, {cont} cont)')

        if ended:
            self.recording = False
            self.sequencer.stop()
            self._set_cont(None)
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            if self.acq_thread.error:
                self.status_var.set('THREAD DIED - see console for traceback')
            return

        if self.recording:
            self.root.after(50, self._poll_queue)

    # ---------- previews ----------

    def _update_previews(self):
        for c in self._cams():
            lbl, sv, pw = self.previews[c.cam_name]
            rgb = c.latest_frame_rgb()
            if rgb is not None:
                h, w = rgb.shape[:2]
                new_h = max(1, int(h * pw / w))
                small = cv2.resize(rgb, (pw, new_h),
                                   interpolation=cv2.INTER_AREA)
                photo = ImageTk.PhotoImage(Image.fromarray(small))
                lbl.configure(image=photo)
                lbl.image = photo
            sv.set(c.status_line())

        if self.sequencer.running:
            self.auto_status_var.set(
                f'automation: {self.sequencer.state}  '
                f'({self.blip_n} events)')
            self.clock_var.set(self.sequencer.clock_text())
        elif self.recording:
            self.auto_status_var.set(
                f'automation off (manual), lamp {self.lamp_state}')
            self.clock_var.set('')
        else:
            self.clock_var.set('')

        self.root.after(PREVIEW_MS, self._update_previews)

    def _on_close(self):
        self.recording = False
        self.sequencer.stop()
        if self.acq_thread is not None and self.acq_thread.is_alive():
            self.acq_thread.stop()
            self.acq_thread.join(timeout=6)
        self._finish_writers()
        for c in self._cams():
            c.stop()
            c.join(timeout=3)
        for c in reversed(self._spin_cams):
            try:
                c.DeInit()
            except Exception:
                pass
        self._spin_cams.clear()
        if self._spin_cam_list is not None:
            self._spin_cam_list.Clear()
        if self._spin_system is not None:
            self._spin_system.ReleaseInstance()
        if self.relay_ser is not None:
            try:
                self.relay_ser.write(LAMPS_OFF_CMD.encode())
                self.relay_ser.close()
            except Exception:
                pass
        self.root.destroy()


if __name__ == '__main__':
    print(f'[keithley] voltage mode, NPLC {NPLC}, range '
          f'{VOLT_RANGE if VOLT_RANGE is not None else "auto"} V, '
          f'C = {CAP_F * 1e9:g} nF')
    print(f'[cache] {PRE_S:g} s before + {POST_ROLL_S:g} s after a blip: '
          f'FLIR (~{flir_ram_mb():.0f} MB) on lamp '
          f'{"/".join(FLIR_CACHE_LAMPS).upper()}, {len(BRIOS)} Brios '
          f'(~{brio_ram_mb():.0f} MB each) on lamp '
          f'{"/".join(BRIO_CACHE_LAMPS).upper()}')
    if CONT_ENABLED and BRIO_CONT_LAMPS:
        print(f'[cont] {len(BRIOS)} Brios ~{CONT_FPS} fps on lamp '
              f'{"/".join(BRIO_CONT_LAMPS).upper()}, {CONT_WRITERS} writers '
              f'each, {CONT_FMT}')
    root = tk.Tk()
    app = ElectrometerApp(root)
    root.mainloop()