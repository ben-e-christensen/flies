"""TEST VARIANT (UNTESTED): dual FLIR burst capture + continuous 20 fps save
of camera A while lamp CONT_LAMP (currently B) is on.

Same as main_flir_dual.py (both cameras cache every light phase and dump on a
blip), plus:

  * While lamp CONT_LAMP is on (automation OR manual press), camera A also
    saves every CONT_EVERY-th frame (60 fps / 3 = 20 fps) straight to disk,
    blip or no blip, downscaled to CONT_SIZE.  Stops the moment that lamp
    turns off or the other lamp turns on.
  * Continuous frames go through a BOUNDED queue into a pool of
    CONT_WRITERS threads.  If the disk/encoder falls behind, frames are
    dropped and counted instead of eating RAM forever.
  * PNGs are written at compression level PNG_LEVEL (1 = fast).  Set
    CONT_FMT = '.npy' for zero encode cost if PNG can't keep up.

Output:
    E:/Ben Christensen/FLIES/session_<stamp>/
        meta.txt
        electrometer.csv          time,charge,trigger
        events.csv                event,time,lamp,rec_cam,frames
        flir_a/e001_0000.png ...  blip bursts from camera A
        flir_b/e001_0000.png ...  blip bursts from camera B
        flir_a_frames.csv         time,event,filename
        flir_b_frames.csv         time,event,filename
        flir_a_cont/p001_000000.png ...   continuous, p = lamp phase #
        flir_a_cont_frames.csv    time,phase,filename  (unsorted - multiple
                                                        writer threads)
"""

import math
import platform
import queue
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

IS_LINUX = platform.system() == 'Linux'

# --- electrometer ---
DELAY_MS      = 5
PREFACTOR     = 1e12          # C -> pC
SERIAL_PORT   = port('keithley')
BAUDRATE      = 9600
PLOT_WINDOW_S = 10
ECHO_RAW      = False

# --- Arduino relay controller ---
RELAY_ENABLED = True
RELAY_PORT    = port('arduino')
RELAY_BAUD    = 9600
LAMP_CMDS     = ('a', 'b')    # single chars the sketch understands
LAMPS_OFF_CMD = 'o'

# --- blip detector ---
CHARGE_RISE_PC   = 10.0       # pC of swing that counts as a blip (window)
TRIGGER_WINDOW_S = 5.0        # ...within this sliding window
JUMP_PC          = 1.0        # pC step between consecutive samples that
                              # counts as a blip regardless of the window

# --- light cycle ---
AUTOMATION_ON  = True         # Start also starts the lamp cycle
DARK_S         = 20 * 60      # both lamps off between blips
LIGHT_LINGER_S = 30           # keep the lamp on this long after a blip
LIGHT_MAX_S    = None         # give up waiting after this many s (None = forever)

# --- FLIR Grasshopper3 (PySpin) ---
FLIR_ENABLED    = True        # set False if Spinnaker not installed
FLIR2_ENABLED   = True        # second camera
CACHE_BOTH_CAMS = True        # True: both cams cache every light phase
                              # False: only the cam opposite the lit lamp
FLIR_FPS      = 60            # acquisition rate; also the ring-buffer rate
CACHE_SECONDS = 20.0          # 5 s lead-in + 15 s post-roll
POST_ROLL_S   = 15.0          # seconds to keep caching after a blip fires
FLIR_SAVE_FMT = '.png'        # .png for 16-bit mono, .jpg for 8-bit
PNG_LEVEL     = 1             # PNG compression 0-9; 1 is ~2-3x faster than 3

# --- continuous save (TEST): cam A while lamp A is on ---
CONT_ENABLED   = True
CONT_LAMP      = 'b'          # lamp that turns continuous saving on
CONT_FPS       = 20           # target save rate (should divide FLIR_FPS)
CONT_WRITERS   = 3            # writer threads; cv2.imwrite releases the GIL
CONT_QUEUE_MAX = 1200         # frames buffered before dropping
                              # (1200 x ~4.1 MB = ~5 GB, ~60 s of backlog)
CONT_FMT       = '.png'       # '.png' or '.npy' (raw, no encode cost)
CONT_SIZE      = (1280, 720)  # (w, h) to downscale continuous frames to,
                              # or None for the full ROI
CONT_8BIT      = False        # True -> also drop continuous frames to 8-bit

# --- FLIR resolution / RAM tradeoff -------------------------------------
# Ring buffer size = CACHE_SECONDS * FLIR_FPS * width * height * bytes/px
FLIR_ROI    = (1920, 1080)    # centered crop (w, h), or None for full sensor
FLIR_MONO16 = True            # False -> Mono8, which halves the RAM

# --- preview ---
PREVIEW_W  = 720
PREVIEW_MS = 300
# ================================

CACHE_FRAMES = max(1, int(round(CACHE_SECONDS * FLIR_FPS)))
CONT_EVERY   = max(1, int(round(FLIR_FPS / CONT_FPS)))
if FLIR_FPS % CONT_FPS:
    print(f'[cont] warning: {FLIR_FPS} fps / {CONT_FPS} is not an integer, '
          f'saving every {CONT_EVERY} frames '
          f'(~{FLIR_FPS / CONT_EVERY:.1f} fps)')

SETUP_CMD = (b"*RST; :SYST:ZCH ON; :SENS:FUNC 'CHAR'; CHAR:RANG 20e-9; "
             b":SENS:CHAR:NPLC 1; :FORM:ELEM READ; :SYST:ZCH OFF; "
             b":CALC2:NULL:STAT ON\n")

ZCHK_CMD = b":SYST:ZCH ON; :CALC2:NULL:STAT ON; :SYST:ZCH OFF\n"

try:
    import PySpin
    HAS_PYSPIN = True
except ImportError:
    HAS_PYSPIN = False
    if FLIR_ENABLED or FLIR2_ENABLED:
        print('[FLIR] PySpin not installed - FLIR cameras disabled')
        FLIR_ENABLED = False
        FLIR2_ENABLED = False


def cache_ram_mb():
    """Rough RAM ONE ring buffer will occupy, in MB."""
    w, h = FLIR_ROI if FLIR_ROI else (1920, 1200)
    return CACHE_FRAMES * w * h * (2 if FLIR_MONO16 else 1) / 1e6


def png_params(ext):
    return [cv2.IMWRITE_PNG_COMPRESSION, PNG_LEVEL] if ext == '.png' else []


# ------------------------------------------------------------- blip detector

class ChargeTrigger:
    """Fires once when EITHER condition is met:

      1. Charge swings by >= thresh_pc inside a sliding window
         (slow accumulation — e.g. 10 pC over 5 s).
      2. Charge jumps by >= jump_pc between two consecutive samples
         (sharp step — e.g. 1 pC in one reading).

    update() returns True for exactly one sample per armed cycle.
    """

    def __init__(self, window_s=TRIGGER_WINDOW_S, thresh_pc=CHARGE_RISE_PC,
                 jump_pc=JUMP_PC):
        self.window_s = window_s
        self.thresh_pc = thresh_pc
        self.jump_pc = jump_pc
        self._buf = deque()          # (t, q_pC) inside the window
        self._prev = None            # previous q_pC for jump detection
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
        """Clear detection state without disarming.  Call after ZCHK."""
        self._buf.clear()
        self._prev = None

    def update(self, t, q_pc):
        if not self._armed or math.isnan(q_pc):
            return False

        # --- condition 2: single-sample jump ---
        if self._prev is not None and abs(q_pc - self._prev) >= self.jump_pc:
            self.fired_at = t
            self._armed = False
            self._prev = q_pc
            return True
        self._prev = q_pc

        # --- condition 1: swing over the sliding window ---
        self._buf.append((t, q_pc))
        while self._buf and t - self._buf[0][0] > self.window_s:
            self._buf.popleft()

        # Need a full window before a swing across it means anything.
        if t - self._buf[0][0] < self.window_s * 0.9:
            return False

        vals = [q for _, q in self._buf]
        if q_pc - min(vals) >= self.thresh_pc \
                or max(vals) - q_pc >= self.thresh_pc:
            self.fired_at = t
            self._armed = False
            return True
        return False


# ----------------------------------------------------------- electrometer

class AcquisitionThread(threading.Thread):
    """Owns the serial port.  Pushes (t, Q, trigger) samples to a queue."""

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

                f.write('time,charge,trigger\n')
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
                    except ValueError:
                        val = np.nan

                    trig = self.trigger.update(t, val * PREFACTOR)

                    f.write(f'{t},{val},{int(trig)}\n')
                    f.flush()
                    self.out_queue.put((t, val, trig))

                    # drain any injected commands (e.g. ZCHK)
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


# ----------------------------------------------------------- FLIR / PySpin

class SpinnakerCamera(threading.Thread):
    """Grasshopper3 capture into a RAM ring buffer, dumped on a blip.

    While `caching` is set, every frame goes into a deque of CACHE_FRAMES.
    burst() schedules a dump: once POST_ROLL_S has elapsed the whole buffer
    is handed to the writer queue and caching stops until the next light
    phase turns it back on.

    Independently, while `cont_active` is set, every CONT_EVERY-th frame is
    pushed (non-blocking) onto `cont_queue` for continuous saving.
    """

    def __init__(self, name='flir', pyspin_cam=None):
        super().__init__(daemon=True)
        self.cam_name = name
        self._pyspin_cam = pyspin_cam   # passed in already Init'd

        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest_frame = None       # mono8, for the preview

        self.cache = deque(maxlen=CACHE_FRAMES)
        self._cache_lock = threading.Lock()
        self.caching = threading.Event()

        self.save_queue = None
        self.t0 = 0.0
        self._dump_at = None            # perf_counter deadline
        self._dump_event = 0
        self.last_burst = 0             # frames written by the last dump
        self.saved = 0
        self.received = 0
        self.resolution = None
        self.pixel_fmt = None
        self.fps = 0.0
        self._fps_t = time.perf_counter()
        self._fps_n = 0
        self.error = None

        # --- continuous save ---
        self.cont_queue = None
        self.cont_active = threading.Event()
        self._cont_phase = 0
        self._cont_n = 0                # frames seen this phase
        self._cont_k = 0                # frames saved this phase
        self.cont_queued = 0            # total pushed this session
        self.cont_dropped = 0           # total dropped (queue full)

    def stop(self):
        self._stop_event.set()

    def latest_frame_rgb(self):
        with self._lock:
            if self._latest_frame is None:
                return None
            return cv2.cvtColor(self._latest_frame, cv2.COLOR_GRAY2RGB)

    def arm(self, save_queue, t0):
        """Attach a writer queue for this session."""
        self.save_queue = save_queue
        self.t0 = t0
        self.saved = 0
        self.last_burst = 0
        self._dump_at = None
        with self._cache_lock:
            self.cache.clear()

    def cont_arm(self, cont_queue):
        """Attach the continuous-save queue for this session."""
        self.cont_active.clear()
        self.cont_queue = cont_queue
        self.cont_queued = 0
        self.cont_dropped = 0

    def cont_set(self, on, phase=0):
        if on:
            self._cont_phase = phase
            self._cont_n = 0
            self._cont_k = 0
            self.cont_active.set()
        else:
            self.cont_active.clear()

    def cache_len(self):
        with self._cache_lock:
            return len(self.cache)

    def burst(self, event_n):
        """Schedule a dump of the ring buffer, tagged as event `event_n`."""
        self._dump_event = event_n
        self._dump_at = time.perf_counter() + POST_ROLL_S

    def status_line(self):
        if self.error:
            return 'ERROR - see console'
        if self.received == 0:
            return 'waiting for stream...'
        s = f'{self.fps:.1f} fps'
        if self.resolution:
            s += f'  {self.resolution[0]}x{self.resolution[1]} {self.pixel_fmt}'
        if self.caching.is_set():
            with self._cache_lock:
                n = len(self.cache)
            s += f'  cache {n}/{CACHE_FRAMES}'
        else:
            s += '  cache off'
        if self.saved:
            s += f'  {self.saved} saved'
        if self.cont_queue is not None:
            state = 'ON' if self.cont_active.is_set() else 'off'
            s += (f'\ncont {state}: {self.cont_queued} queued, '
                  f'{self.cont_dropped} dropped, '
                  f'backlog {self.cont_queue.qsize()}')
        return s

    # ---------- camera setup ----------

    def _configure(self, nodemap):
        node_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        node_mode.SetIntValue(node_mode.GetEntryByName('Continuous').GetValue())

        if FLIR_ROI:
            self._set_roi(nodemap, *FLIR_ROI)

        try:
            node_fr_en = PySpin.CBooleanPtr(
                nodemap.GetNode('AcquisitionFrameRateEnable'))
            if PySpin.IsAvailable(node_fr_en) and PySpin.IsWritable(node_fr_en):
                node_fr_en.SetValue(True)
            node_fr = PySpin.CFloatPtr(nodemap.GetNode('AcquisitionFrameRate'))
            if PySpin.IsAvailable(node_fr) and PySpin.IsWritable(node_fr):
                node_fr.SetValue(min(FLIR_FPS, node_fr.GetMax()))
        except PySpin.SpinnakerException:
            pass

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
                          f'ring buffer ~{cache_ram_mb():.0f} MB')

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

    def _on_frame(self, raw, preview):
        now = time.perf_counter()
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
                self.cache.append((now - self.t0, raw))

        # --- continuous save: every CONT_EVERY-th frame, never blocks ---
        if self.cont_active.is_set() and self.cont_queue is not None:
            if self._cont_n % CONT_EVERY == 0:
                try:
                    self.cont_queue.put_nowait(
                        (now - self.t0, raw, self._cont_phase, self._cont_k))
                    self.cont_queued += 1
                except queue.Full:
                    self.cont_dropped += 1
                self._cont_k += 1       # index advances even on a drop,
                                        # so gaps show up in the filenames
            self._cont_n += 1

        if self._dump_at is not None and now >= self._dump_at:
            self._dump_at = None
            self._flush_cache()

    def _flush_cache(self):
        """Hand the whole ring buffer to the writer, then stop caching."""
        with self._cache_lock:
            frames = list(self.cache)
            self.cache.clear()
        self.caching.clear()

        if self.save_queue is None:
            return
        for k, (t, raw) in enumerate(frames):
            self.save_queue.put((t, raw, self._dump_event, k))
        self.last_burst = len(frames)
        self.saved += len(frames)
        print(f'[{self.cam_name}] event {self._dump_event}: '
              f'{len(frames)} frames queued')


# ----------------------------------------------------------- frame writers

class FrameWriter(threading.Thread):
    """Writes blip bursts to disk.  One file per frame, named by event."""

    def __init__(self, in_queue, out_dir, index_path, ext=FLIR_SAVE_FMT):
        super().__init__(daemon=True)
        self.in_queue = in_queue
        self.out_dir = out_dir
        self.index_path = index_path
        self.ext = ext
        self.params = png_params(ext)
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

    Filenames are p<phase>_<k>, so sort by filename (not by index row) to get
    frames in order.
    """

    def __init__(self, in_queue, out_dir, index_path, n_threads=CONT_WRITERS,
                 ext=CONT_FMT):
        self.in_queue = in_queue
        self.out_dir = out_dir
        self.index_path = index_path
        self.n_threads = max(1, n_threads)
        self.ext = ext
        self.params = png_params(ext)
        self._idx = None
        self._idx_lock = threading.Lock()
        self._threads = []
        self.written = 0
        self.error = None

    def start(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._idx = open(self.index_path, 'w')
        self._idx.write('time,phase,filename\n')
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
                t, data, phase, k = item
                if CONT_SIZE is not None:
                    data = cv2.resize(data, CONT_SIZE,
                                      interpolation=cv2.INTER_AREA)
                if CONT_8BIT and data.dtype == np.uint16:
                    data = (data >> 8).astype(np.uint8)
                fname = f'p{phase:03d}_{k:06d}{self.ext}'
                path = str(self.out_dir / fname)
                if self.ext == '.npy':
                    np.save(path, data)
                else:
                    cv2.imwrite(path, data, self.params)
                with self._idx_lock:
                    if self._idx is not None:
                        self._idx.write(f'{t:.6f},{phase},{fname}\n')
                        self.written += 1
                        if self.written % 50 == 0:
                            self._idx.flush()
        except Exception:
            self.error = traceback.format_exc()
            print(f'[cont] writer error:\n{self.error}')

    def finish(self, timeout=300):
        """Drain the backlog, stop the threads, close the index."""
        backlog = self.in_queue.qsize()
        if backlog:
            print(f'[cont] draining {backlog} queued frames...')
        for _ in self._threads:
            self.in_queue.put(None)     # blocks if full; writers are draining
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
    """lamp A -> blip -> linger -> dark (ZCHK) -> lamp B -> blip -> ...

    Driven entirely from the Tk event loop, so there is no extra thread and
    the relay serial port stays owned by the main thread.
    """

    def __init__(self, app):
        self.app = app
        self.running = False
        self.state = 'idle'
        self.lamp_i = 0
        self._job = None

    def start(self):
        self.running = True
        self._begin_light()

    def stop(self):
        self.running = False
        self.state = 'idle'
        self._cancel()
        self.app.trigger.disarm()
        self.app.flir_caching_off_all()

    def _cancel(self):
        if self._job is not None:
            self.app.root.after_cancel(self._job)
            self._job = None

    def _begin_light(self):
        """Turn on the next lamp, arm the detector, start caching."""
        self._cancel()
        if not self.running:
            return
        # reset charge baseline at the end of the dark period
        if self.app.acq_thread is not None:
            self.app.acq_thread.cmd_queue.put(ZCHK_CMD)
        lamp = LAMP_CMDS[self.lamp_i]
        self.state = f'lamp {lamp.upper()}'
        self.app.send_relay(lamp, from_auto=True)   # also toggles cont save
        self.app.trigger.arm()
        self.app.flir_caching_for_lamp(lamp)
        if LIGHT_MAX_S:
            self._job = self.app.root.after(int(LIGHT_MAX_S * 1000),
                                            self._light_timeout)

    def _light_timeout(self):
        print(f'[auto] no blip within {LIGHT_MAX_S} s - moving on')
        self._begin_dark()

    def lamp(self):
        return LAMP_CMDS[self.lamp_i]

    def advance(self):
        """A blip fired.  Linger with the lamp on, then go dark."""
        if not self.running:
            return
        self._cancel()
        self.app.trigger.disarm()
        if LIGHT_LINGER_S > 0:
            self.state = f'lingering ({LIGHT_LINGER_S}s)'
            self._job = self.app.root.after(int(LIGHT_LINGER_S * 1000),
                                            self._begin_dark)
        else:
            self._begin_dark()

    def _begin_dark(self):
        self._cancel()
        self.app.trigger.disarm()
        self.app.send_relay(LAMPS_OFF_CMD, from_auto=True)  # stops cont save

        # caching is cleared by the camera itself once it has flushed
        self.lamp_i = (self.lamp_i + 1) % len(LAMP_CMDS)
        self.state = f'dark ({DARK_S / 60:.0f} min)'
        self._job = self.app.root.after(int(DARK_S * 1000), self._begin_light)


# ------------------------------------------------------------------- GUI

class ElectrometerApp:

    # Camera A = Spinnaker index 0, camera B = index 1.

    def __init__(self, root):
        self.root = root
        self.root.title(f'Electrometer + dual FLIR - TEST: cam A '
                        f'{CONT_FPS} fps while lamp {CONT_LAMP.upper()} on')

        self.acq_thread = None
        self.data_queue = None
        self.recording = False
        self.t_vec = []
        self.q_vec = []
        self.session_dir = None
        self.events_path = None
        self.blip_n = 0
        self.last_lamp = '-'
        self.lamp_state = 'o'           # what we last told the relay

        self.trigger = ChargeTrigger()
        self.sequencer = LightSequencer(self)

        # --- per-camera save infrastructure ---
        self.save_queue_a = None
        self.save_queue_b = None
        self.writer_a = None
        self.writer_b = None

        # --- continuous save (cam A) ---
        self.cont_queue = None
        self.cont_pool = None
        self.cont_phase = 0

        self._active_cams = []

        # --- Arduino relay connection ---
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

        self.status_var = tk.StringVar(value='Idle')
        ttk.Label(controls, textvariable=self.status_var).pack(side=tk.LEFT,
                                                               padx=12)

        # --- relay buttons ---
        relay_frame = ttk.LabelFrame(controls, text='Relays (manual)',
                                     padding=4)
        relay_frame.pack(side=tk.RIGHT, padx=8)

        ttk.Button(relay_frame, text='Lamp A',
                   command=lambda: self.send_relay('a')).pack(side=tk.LEFT,
                                                              padx=2)
        ttk.Button(relay_frame, text='Lamp B',
                   command=lambda: self.send_relay('b')).pack(side=tk.LEFT,
                                                              padx=2)
        ttk.Button(relay_frame, text='Both Off',
                   command=lambda: self.send_relay('o')).pack(side=tk.LEFT,
                                                              padx=2)

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
        self.ax.set_ylabel('charge [pC]')
        self.line, = self.ax.plot([], [], 'ro-', markersize=3)
        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=body)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH,
                                         expand=True)

        cam_panel = ttk.Frame(body, padding=4)
        cam_panel.pack(side=tk.RIGHT, fill=tk.Y)

        # --- camera A preview ---
        frame_a = ttk.LabelFrame(cam_panel,
                                 text=f'Camera A ({CONT_FPS} fps save while '
                                      f'lamp {CONT_LAMP.upper()} on)',
                                 padding=4)
        frame_a.pack(side=tk.TOP, fill=tk.X, pady=4)
        self.cam_label = tk.Label(frame_a, background='#222')
        self.cam_label.pack()
        self.cam_status_var = tk.StringVar(value='cam A disabled')
        ttk.Label(frame_a, textvariable=self.cam_status_var).pack(anchor='w')

        # --- camera B preview ---
        frame_b = ttk.LabelFrame(cam_panel, text='Camera B', padding=4)
        frame_b.pack(side=tk.TOP, fill=tk.X, pady=4)
        self.cam2_label = tk.Label(frame_b, background='#222')
        self.cam2_label.pack()
        self.cam2_status_var = tk.StringVar(value='cam B disabled')
        ttk.Label(frame_b, textvariable=self.cam2_status_var).pack(anchor='w')

        self.auto_status_var = tk.StringVar(value='automation idle')
        ttk.Label(cam_panel, textvariable=self.auto_status_var).pack(
            anchor='w', pady=4)
        ttk.Label(cam_panel,
                  text=f'ring buffer {CACHE_FRAMES} frames '
                       f'(~{cache_ram_mb():.0f} MB each, '
                       f'~{cache_ram_mb() * 2:.0f} MB total)').pack(anchor='w')

        self.cam = None       # camera A
        self.cam2 = None      # camera B
        self._spin_system = None
        self._spin_cam_list = None
        self._spin_cams = []          # keep refs alive

        if (FLIR_ENABLED or FLIR2_ENABLED) and HAS_PYSPIN:
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

                if FLIR_ENABLED and len(self._spin_cams) >= 1:
                    self.cam = SpinnakerCamera('cam_a',
                                              pyspin_cam=self._spin_cams[0])
                    self.cam.start()

                if FLIR2_ENABLED and len(self._spin_cams) >= 2:
                    self.cam2 = SpinnakerCamera('cam_b',
                                               pyspin_cam=self._spin_cams[1])
                    self.cam2.start()
                elif FLIR2_ENABLED:
                    print('[spinnaker] only 1 camera found, cam_b disabled')
            except Exception as e:
                print(f'[spinnaker] init error: {e}')

        self.root.protocol('WM_DELETE_WINDOW', self._on_close)
        self.root.after(PREVIEW_MS, self._update_previews)

    # ---------- dual-camera helpers ----------

    def _opposite_cam(self, lamp_cmd):
        """Return the camera on the opposite side of the given lamp."""
        if lamp_cmd == 'a':
            return self.cam2
        elif lamp_cmd == 'b':
            return self.cam
        return None

    def flir_caching_for_lamp(self, lamp_cmd):
        """Start caching on the camera(s) for this light phase."""
        if CACHE_BOTH_CAMS:
            self.flir_caching_both()
            return
        self.flir_caching_off_all()
        opp = self._opposite_cam(lamp_cmd)
        if opp is not None and opp.is_alive() and opp.error is None:
            opp.caching.set()
            self._active_cams = [opp]
        else:
            self._active_cams = []

    def flir_caching_both(self):
        """Cache on all available cameras."""
        self._active_cams = []
        for c in (self.cam, self.cam2):
            if c is not None and c.is_alive() and c.error is None:
                c.caching.set()
                self._active_cams.append(c)

    def flir_caching_off_all(self):
        """Stop caching on all cameras."""
        for c in (self.cam, self.cam2):
            if c is not None:
                c.caching.clear()
        self._active_cams = []

    def flir_burst_active(self, event_n):
        """Burst-dump whichever camera(s) are actively caching."""
        for c in self._active_cams:
            c.burst(event_n)

    # ---------- continuous save control ----------

    def _set_cont(self, on):
        """Turn cam A's continuous 20 fps save on/off.  Called on every
        relay write, so it tracks the lamp exactly - auto or manual."""
        cam = self.cam
        if not CONT_ENABLED or cam is None or cam.cont_queue is None:
            return
        if on and self.recording:
            if not cam.cont_active.is_set():
                self.cont_phase += 1
                cam.cont_set(True, self.cont_phase)
                print(f'[cont] phase {self.cont_phase}: cam A saving at '
                      f'{FLIR_FPS / CONT_EVERY:.0f} fps')
        elif cam.cont_active.is_set():
            cam.cont_set(False)
            print(f'[cont] phase {self.cont_phase} stopped '
                  f'({cam.cont_queued} queued, {cam.cont_dropped} dropped '
                  f'so far)')

    # ---------- things the sequencer calls ----------

    def send_relay(self, cmd, from_auto=False):
        """Send one character to the relay Arduino.

        A manual press switches the automation off so the two cannot fight.
        """
        if not from_auto:
            self.last_lamp = cmd if cmd in LAMP_CMDS else '-'
            if self.sequencer.running:
                self.auto_var.set(False)
                self._on_auto_toggle()

        if self.relay_ser is None or not self.relay_ser.is_open:
            self.relay_var.set('not connected')
            return
        try:
            self.relay_ser.write(cmd.encode())
            self.lamp_state = cmd
            labels = {'a': 'Lamp A ON', 'b': 'Lamp B ON', 'o': 'Both OFF'}
            self.relay_var.set(labels.get(cmd, cmd))
        except Exception as e:
            self.relay_var.set(f'error: {e}')
            print(f'[relay] write error: {e}')
            return

        self._set_cont(cmd == CONT_LAMP)

    def on_blip(self, t):
        """A blip fired.  Dump the frame buffer(s); only the automation
        reacts by changing the lights."""
        self.blip_n += 1
        lamp = self.sequencer.lamp() if self.sequencer.running else self.last_lamp

        n_frames = sum(c.cache_len() for c in self._active_cams)
        rec_names = ','.join(c.cam_name for c in self._active_cams) or '-'

        self.flir_burst_active(self.blip_n)
        self._log_event(self.blip_n, t, lamp, rec_names, n_frames)
        print(f'[blip] {self.blip_n} at t={t:.2f} s, lamp {lamp}, '
              f'rec {rec_names}, {n_frames} frames')

        if self.sequencer.running:
            self.sequencer.advance()        # lights off, dark, then next lamp
        else:
            self._begin_manual()            # keep watching, leave lamps alone

    def _begin_manual(self):
        """Arm the detector and both ring buffers without touching the lamps."""
        self.trigger.arm()
        self.flir_caching_both()

    def _log_event(self, n, t, lamp, rec_cam, n_frames):
        if self.events_path is None:
            return
        with open(self.events_path, 'a') as f:
            f.write(f'{n},{t:.3f},{lamp},{rec_cam},{n_frames}\n')

    def _on_auto_toggle(self):
        if self.auto_var.get():
            if self.recording:
                self.sequencer.start()
        else:
            self.sequencer.stop()
            if self.recording:
                self._begin_manual()
            self.auto_status_var.set('automation off (manual)')

    # ---------- button handlers ----------

    def start(self):
        self.t_vec = []
        self.q_vec = []
        self.line.set_data([], [])
        self.canvas.draw_idle()

        stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.session_dir = ROOT_FOLDER / f'session_{stamp}'
        self.session_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()

        with open(self.session_dir / 'meta.txt', 'w') as f:
            f.write(f'wall_clock_start\t{datetime.now().isoformat()}\n')
            f.write('script\tTEST cont20 variant\n')
            f.write(f'serial_port\t{SERIAL_PORT}\n')
            f.write(f'relay_port\t{RELAY_PORT}\n')
            f.write(f'delay_ms\t{DELAY_MS}\n')
            f.write('mode\tcharge\n')
            f.write(f'charge_rise_pC\t{CHARGE_RISE_PC}\n')
            f.write(f'jump_pC\t{JUMP_PC}\n')
            f.write(f'trigger_window_s\t{TRIGGER_WINDOW_S}\n')
            f.write(f'dark_s\t{DARK_S}\n')
            f.write(f'light_linger_s\t{LIGHT_LINGER_S}\n')
            f.write(f'light_max_s\t{LIGHT_MAX_S}\n')
            f.write(f'automation\t{self.auto_var.get()}\n')
            f.write(f'flir_enabled\t{FLIR_ENABLED}\n')
            f.write(f'flir2_enabled\t{FLIR2_ENABLED}\n')
            f.write(f'cache_both_cams\t{CACHE_BOTH_CAMS}\n')
            if FLIR_ENABLED or FLIR2_ENABLED:
                f.write(f'flir_fps\t{FLIR_FPS}\n')
                f.write(f'flir_roi\t{FLIR_ROI}\n')
                f.write(f'flir_mono16\t{FLIR_MONO16}\n')
                f.write(f'cache_seconds\t{CACHE_SECONDS}\n')
                f.write(f'cache_frames\t{CACHE_FRAMES}\n')
                f.write(f'post_roll_s\t{POST_ROLL_S}\n')
                f.write(f'png_level\t{PNG_LEVEL}\n')
            f.write(f'cont_enabled\t{CONT_ENABLED}\n')
            if CONT_ENABLED:
                f.write(f'cont_lamp\t{CONT_LAMP}\n')
                f.write(f'cont_fps\t{FLIR_FPS / CONT_EVERY}\n')
                f.write(f'cont_every\t{CONT_EVERY}\n')
                f.write(f'cont_writers\t{CONT_WRITERS}\n')
                f.write(f'cont_queue_max\t{CONT_QUEUE_MAX}\n')
                f.write(f'cont_fmt\t{CONT_FMT}\n')
                f.write(f'cont_size\t{CONT_SIZE}\n')
                f.write(f'cont_8bit\t{CONT_8BIT}\n')

        self.events_path = self.session_dir / 'events.csv'
        with open(self.events_path, 'w') as f:
            f.write('event,time,lamp,rec_cam,frames\n')

        # --- per-camera burst writers ---
        self.save_queue_a = None
        self.save_queue_b = None
        self.writer_a = None
        self.writer_b = None

        def _arm_cam(cam, label, subdir, index_csv):
            if cam is None or cam.error is not None or not cam.is_alive():
                if cam is not None:
                    print(f'[{label}] not recording (dead or errored)')
                return None, None
            sq = queue.Queue()
            w = FrameWriter(sq,
                            self.session_dir / subdir,
                            self.session_dir / index_csv)
            w.start()
            cam.arm(sq, t0)
            return sq, w

        self.save_queue_a, self.writer_a = _arm_cam(
            self.cam, 'cam_a', 'flir_a', 'flir_a_frames.csv')
        self.save_queue_b, self.writer_b = _arm_cam(
            self.cam2, 'cam_b', 'flir_b', 'flir_b_frames.csv')

        # --- continuous writer pool for cam A ---
        self.cont_queue = None
        self.cont_pool = None
        self.cont_phase = 0
        if CONT_ENABLED and self.save_queue_a is not None:
            self.cont_queue = queue.Queue(maxsize=CONT_QUEUE_MAX)
            self.cont_pool = ContinuousWriterPool(
                self.cont_queue,
                self.session_dir / 'flir_a_cont',
                self.session_dir / 'flir_a_cont_frames.csv',
            )
            self.cont_pool.start()
            self.cam.cont_arm(self.cont_queue)

        self.trigger.disarm()
        self.blip_n = 0
        self.sequencer.lamp_i = 0

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

        if self.auto_var.get():
            self.sequencer.start()
        else:
            self._begin_manual()
            # lamp A may already be on from a manual press before Start
            self._set_cont(self.lamp_state == CONT_LAMP)

        self.root.after(50, self._poll_queue)

    def stop(self):
        self.recording = False
        self.sequencer.stop()
        self.send_relay(LAMPS_OFF_CMD, from_auto=True)
        self._set_cont(False)           # in case the relay write failed

        if self.acq_thread is not None:
            self.acq_thread.stop()
            self.acq_thread.join(timeout=6)

        self._finish_writers()

        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

        good = int(np.count_nonzero(~np.isnan(self.q_vec))) \
            if self.q_vec else 0
        frames_a = self.cam.saved if self.cam else 0
        frames_b = self.cam2.saved if self.cam2 else 0
        cont = (f', cont {self.cam.cont_queued} saved/'
                f'{self.cam.cont_dropped} dropped') if self.cam else ''
        self.status_var.set(
            f'Stopped. {good}/{len(self.q_vec)} valid readings, '
            f'{self.blip_n} blips, {frames_a}+{frames_b} burst frames{cont} '
            f'-> {self.session_dir.name}'
        )

    def _finish_writers(self):
        """Let the cameras hand over anything queued, then close writers."""
        self.flir_caching_off_all()
        if self.cam is not None:
            self.cam.cont_set(False)
        time.sleep(0.2)

        for sq, w in ((self.save_queue_a, self.writer_a),
                      (self.save_queue_b, self.writer_b)):
            if sq is not None:
                sq.put(None)
            if w is not None:
                w.join(timeout=120)

        if self.cont_pool is not None:
            self.cont_pool.finish()
        if self.cam is not None:
            self.cam.cont_queue = None

        self.save_queue_a = self.save_queue_b = None
        self.writer_a = self.writer_b = None
        self.cont_queue = None
        self.cont_pool = None

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
                self.q_vec.append(val)
                updated = True
                if trig:
                    self.on_blip(t)
        except queue.Empty:
            pass

        if updated:
            t_arr = np.array(self.t_vec)
            q_arr = np.array(self.q_vec) * PREFACTOR

            i0 = np.searchsorted(t_arr, t_arr[-1] - PLOT_WINDOW_S)
            self.line.set_data(t_arr[i0:], q_arr[i0:])

            right = max(t_arr[-1], PLOT_WINDOW_S)
            self.ax.set_xlim(right - PLOT_WINDOW_S, right)
            self.ax.relim()
            self.ax.autoscale_view(scalex=False)
            self.canvas.draw_idle()

            frames_a = self.cam.saved if self.cam else 0
            frames_b = self.cam2.saved if self.cam2 else 0
            cont = self.cam.cont_queued if self.cam else 0
            self.status_var.set(
                f'Recording -> {self.session_dir.name}  '
                f'({len(self.t_vec)} samples, '
                f'{self.blip_n} blips, {frames_a}+{frames_b} burst, '
                f'{cont} cont)'
            )

        if ended:
            self.recording = False
            self.sequencer.stop()
            self._set_cont(False)
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            if self.acq_thread.error:
                self.status_var.set('THREAD DIED - see console for traceback')
            return

        if self.recording:
            self.root.after(50, self._poll_queue)

    # ---------- previews ----------

    def _update_previews(self):
        if self.cam is not None:
            rgb = self.cam.latest_frame_rgb()
            if rgb is not None:
                h, w = rgb.shape[:2]
                new_h = max(1, int(h * PREVIEW_W / w))
                small = cv2.resize(rgb, (PREVIEW_W, new_h),
                                   interpolation=cv2.INTER_AREA)
                photo = ImageTk.PhotoImage(Image.fromarray(small))
                self.cam_label.configure(image=photo)
                self.cam_label.image = photo
            self.cam_status_var.set(self.cam.status_line())

        if self.cam2 is not None:
            rgb2 = self.cam2.latest_frame_rgb()
            if rgb2 is not None:
                h, w = rgb2.shape[:2]
                new_h = max(1, int(h * PREVIEW_W / w))
                small = cv2.resize(rgb2, (PREVIEW_W, new_h),
                                   interpolation=cv2.INTER_AREA)
                photo2 = ImageTk.PhotoImage(Image.fromarray(small))
                self.cam2_label.configure(image=photo2)
                self.cam2_label.image = photo2
            self.cam2_status_var.set(self.cam2.status_line())

        if self.sequencer.running:
            self.auto_status_var.set(
                f'automation: {self.sequencer.state}  '
                f'({self.blip_n} blips)')
        elif self.recording:
            self.auto_status_var.set('automation off (manual)')

        self.root.after(PREVIEW_MS, self._update_previews)

    def _on_close(self):
        self.recording = False
        self.sequencer.stop()
        self._set_cont(False)
        if self.acq_thread is not None and self.acq_thread.is_alive():
            self.acq_thread.stop()
            self.acq_thread.join(timeout=6)
        self._finish_writers()
        if self.cam is not None:
            self.cam.stop()
            self.cam.join(timeout=3)
        if self.cam2 is not None:
            self.cam2.stop()
            self.cam2.join(timeout=3)
        # Spinnaker teardown: DeInit in reverse, then clear, then release
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
    per_cam = cache_ram_mb()
    print(f'[cache] {CACHE_FRAMES} frames @ {FLIR_FPS} fps = {CACHE_SECONDS} s'
          f', ~{per_cam:.0f} MB/cam, ~{per_cam * 2:.0f} MB total')
    if CONT_ENABLED:
        print(f'[cont] cam A every {CONT_EVERY} frames '
              f'(~{FLIR_FPS / CONT_EVERY:.0f} fps) while lamp '
              f'{CONT_LAMP.upper()} is on, {CONT_WRITERS} writers, '
              f'queue {CONT_QUEUE_MAX}, {CONT_FMT}')
    root = tk.Tk()
    app = ElectrometerApp(root)
    root.mainloop()