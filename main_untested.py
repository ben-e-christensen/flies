"""UNTESTED: automated light-cycling electrometer + FLIR burst capture.

Built from main_arduino.py, with three changes:

  1. No Brio webcams.  The FLIR Grasshopper3 is the only camera.
  2. The lamps run themselves.  Start begins the cycle:

         lamp A on  ->  wait for a charge blip  ->  both lamps off for
         DARK_S  ->  lamp B on  ->  wait for a blip  ->  dark  ->  ...

     A "blip" is a rise of >= CHARGE_RISE_PC picocoulombs inside a
     TRIGGER_WINDOW_S sliding window.
  3. The FLIR does not save continuously.  It keeps the last CACHE_SECONDS
     of frames in a RAM ring buffer and dumps them to disk only when a blip
     fires.  Because charge is cumulative, the blip is detected at its END,
     so a 10 s buffer holds the 5 s blip plus the 5 s leading up to it.
     Nothing is cached at all while the lamps are off.

The manual Lamp A / Lamp B / Both Off buttons still work; pressing one
switches the automation off so it cannot fight you.

Output:
    Kiethley_data/session_<stamp>/
        meta.txt
        electrometer.csv          time,charge,trigger
        events.csv                event,time,lamp,frames
        flir/e001_0000.png ...    one burst per event
        flir_frames.csv           time,event,filename
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
ROOT_FOLDER = Path(__file__).resolve().parent / 'Kiethley_data'
ROOT_FOLDER.mkdir(exist_ok=True)

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
CHARGE_RISE_PC   = 10.0       # pC of rise that counts as a blip
TRIGGER_WINDOW_S = 5.0        # ...within this sliding window

# --- light cycle ---
AUTOMATION_ON = True          # Start also starts the lamp cycle
DARK_S        = 20 * 60       # both lamps off between blips
LIGHT_MAX_S   = None          # give up waiting after this many s (None = forever)

# --- FLIR Grasshopper3 (PySpin) ---
FLIR_ENABLED  = True          # set False if Spinnaker not installed
FLIR_FPS      = 60            # acquisition rate; also the ring-buffer rate
CACHE_SECONDS = 10.0          # 5 s blip + 5 s of lead-in
POST_ROLL_S   = 0.0           # extra seconds to keep caching after a blip.
                              # Leave at 0: the lamps go off at the trigger,
                              # so post-roll frames would just be dark.
FLIR_SAVE_FMT = '.png'        # .png for 16-bit mono, .jpg for 8-bit

# --- FLIR resolution / RAM tradeoff -------------------------------------
# The ring buffer holds RAW frames in memory, so its size is
#     CACHE_SECONDS * FLIR_FPS * width * height * bytes_per_pixel
# Full sensor at Mono16 and 60 fps is a few GB.  Shrink it here if that is
# too much; the startup banner and the GUI both print the actual number.
# These are read when the camera thread starts, so change them and restart.
FLIR_ROI    = (1280, 960)     # centered crop (w, h), or None for full sensor
FLIR_MONO16 = True            # False -> Mono8, which halves the RAM

# --- preview ---
PREVIEW_W  = 360
PREVIEW_MS = 150
# ================================

CACHE_FRAMES = max(1, int(round(CACHE_SECONDS * FLIR_FPS)))

SETUP_CMD = (b"*RST; :SYST:ZCH ON; :SENS:FUNC 'CHAR'; CHAR:RANG 20e-9; "
             b":SENS:CHAR:NPLC 1; :FORM:ELEM READ; :SYST:ZCH OFF; "
             b":CALC2:NULL:STAT ON\n")

try:
    import PySpin
    HAS_PYSPIN = True
except ImportError:
    HAS_PYSPIN = False
    if FLIR_ENABLED:
        print('[FLIR] PySpin not installed - FLIR camera disabled')
        FLIR_ENABLED = False


def cache_ram_mb():
    """Rough RAM the ring buffer will occupy, in MB."""
    w, h = FLIR_ROI if FLIR_ROI else (1920, 1200)
    return CACHE_FRAMES * w * h * (2 if FLIR_MONO16 else 1) / 1e6


# ------------------------------------------------------------- blip detector

class ChargeTrigger:
    """Fires once when charge rises by >= thresh_pc inside a sliding window.

    Charge mode is cumulative, so this is really a rate threshold.  update()
    returns True for exactly one sample per armed cycle - the sample at which
    the rise completed.  That single True is the `trigger` column in
    electrometer.csv; the blip it refers to is the window
    [t - TRIGGER_WINDOW_S, t].
    """

    def __init__(self, window_s=TRIGGER_WINDOW_S, thresh_pc=CHARGE_RISE_PC):
        self.window_s = window_s
        self.thresh_pc = thresh_pc
        self._buf = deque()          # (t, q_pC) inside the window
        self._armed = False
        self.fired_at = None

    def arm(self):
        self._buf.clear()
        self._armed = True
        self.fired_at = None

    def disarm(self):
        self._buf.clear()
        self._armed = False

    def update(self, t, q_pc):
        if not self._armed or math.isnan(q_pc):
            return False

        self._buf.append((t, q_pc))
        while self._buf and t - self._buf[0][0] > self.window_s:
            self._buf.popleft()

        # Need a full window before a rise across it means anything.
        if t - self._buf[0][0] < self.window_s * 0.9:
            return False

        # Compare against the window minimum, not just its oldest sample, so
        # a dip partway through cannot mask the rise.
        if q_pc - min(q for _, q in self._buf) >= self.thresh_pc:
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
    phase turns it back on.  While `caching` is clear the preview still
    updates, but no frame can ever reach the disk.
    """

    def __init__(self, name='flir'):
        super().__init__(daemon=True)
        self.cam_name = name

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
        system = None
        cam = None
        cam_list = None
        try:
            system = PySpin.System.GetInstance()
            cam_list = system.GetCameras()
            if cam_list.GetSize() == 0:
                self.error = 'no Spinnaker cameras found'
                print(f'[{self.cam_name}] {self.error}')
                return

            cam = cam_list[0]
            cam.Init()
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
        finally:
            if cam is not None:
                try:
                    cam.DeInit()
                except Exception:
                    pass
                del cam
            if cam_list is not None:
                cam_list.Clear()
            if system is not None:
                system.ReleaseInstance()

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


# ----------------------------------------------------------- frame writer

class FrameWriter(threading.Thread):
    """Writes bursts of frames to disk.  One file per frame, named by event."""

    def __init__(self, in_queue, out_dir, index_path, ext=FLIR_SAVE_FMT):
        super().__init__(daemon=True)
        self.in_queue = in_queue
        self.out_dir = out_dir
        self.index_path = index_path
        self.ext = ext
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
                    cv2.imwrite(str(self.out_dir / fname), data)
                    idx.write(f'{t:.6f},{event},{fname}\n')
                    self.written += 1
                    if self.written % 20 == 0:
                        idx.flush()
                idx.flush()
        except Exception:
            self.error = traceback.format_exc()
            print(self.error)


# ------------------------------------------------------------ light cycle

class LightSequencer:
    """lamp A -> blip -> dark -> lamp B -> blip -> dark -> ...

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
        self.app.flir_caching(False)

    def _cancel(self):
        if self._job is not None:
            self.app.root.after_cancel(self._job)
            self._job = None

    def _begin_light(self):
        """Turn on the next lamp, arm the detector, start caching frames."""
        self._cancel()
        if not self.running:
            return
        self.state = f'lamp {LAMP_CMDS[self.lamp_i].upper()}'
        self.app.send_relay(LAMP_CMDS[self.lamp_i], from_auto=True)
        self.app.trigger.arm()
        self.app.flir_caching(True)
        if LIGHT_MAX_S:
            self._job = self.app.root.after(int(LIGHT_MAX_S * 1000),
                                            self._light_timeout)

    def _light_timeout(self):
        print(f'[auto] no blip within {LIGHT_MAX_S} s - moving on')
        self._begin_dark()

    def lamp(self):
        return LAMP_CMDS[self.lamp_i]

    def advance(self):
        """A blip fired: kill the lights and start the dark interval."""
        if self.running:
            self._begin_dark()

    def _begin_dark(self):
        self._cancel()
        self.app.trigger.disarm()
        self.app.send_relay(LAMPS_OFF_CMD, from_auto=True)
        # caching is cleared by the camera itself once it has flushed
        self.lamp_i = (self.lamp_i + 1) % len(LAMP_CMDS)
        self.state = f'dark ({DARK_S / 60:.0f} min)'
        self._job = self.app.root.after(int(DARK_S * 1000), self._begin_light)


# ------------------------------------------------------------------- GUI

class ElectrometerApp:

    def __init__(self, root):
        self.root = root
        self.root.title('Electrometer (Charge) + FLIR - automated lights')

        self.acq_thread = None
        self.data_queue = None
        self.writer = None
        self.save_queue = None
        self.recording = False
        self.t_vec = []
        self.q_vec = []
        self.session_dir = None
        self.events_path = None
        self.blip_n = 0
        self.last_lamp = '-'

        self.trigger = ChargeTrigger()
        self.sequencer = LightSequencer(self)

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

        # --- body: plot left, preview right ---
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

        frame = ttk.LabelFrame(cam_panel, text='flir', padding=4)
        frame.pack(side=tk.TOP, fill=tk.X, pady=4)
        self.cam_label = tk.Label(frame, background='#222')
        self.cam_label.pack()
        self.cam_status_var = tk.StringVar(value='FLIR disabled')
        ttk.Label(frame, textvariable=self.cam_status_var).pack(anchor='w')

        self.auto_status_var = tk.StringVar(value='automation idle')
        ttk.Label(cam_panel, textvariable=self.auto_status_var).pack(
            anchor='w', pady=4)
        ttk.Label(cam_panel,
                  text=f'ring buffer {CACHE_FRAMES} frames '
                       f'(~{cache_ram_mb():.0f} MB)').pack(anchor='w')

        self.cam = None
        if FLIR_ENABLED and HAS_PYSPIN:
            self.cam = SpinnakerCamera('flir')
            self.cam.start()

        self.root.protocol('WM_DELETE_WINDOW', self._on_close)
        self.root.after(PREVIEW_MS, self._update_previews)

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
            labels = {'a': 'Lamp A ON', 'b': 'Lamp B ON', 'o': 'Both OFF'}
            self.relay_var.set(labels.get(cmd, cmd))
        except Exception as e:
            self.relay_var.set(f'error: {e}')
            print(f'[relay] write error: {e}')

    def flir_caching(self, on):
        if self.cam is None:
            return
        if on:
            self.cam.caching.set()
        else:
            self.cam.caching.clear()

    def flir_burst(self, event_n):
        if self.cam is not None:
            self.cam.burst(event_n)

    def on_blip(self, t):
        """A blip fired.  Dump the frame buffer either way; only the
        automation reacts by changing the lights."""
        self.blip_n += 1
        lamp = self.sequencer.lamp() if self.sequencer.running else self.last_lamp
        n_frames = self.cam.cache_len() if self.cam else 0
        self.flir_burst(self.blip_n)
        self._log_event(self.blip_n, t, lamp, n_frames)
        print(f'[blip] {self.blip_n} at t={t:.2f} s, lamp {lamp}, '
              f'{n_frames} frames')

        if self.sequencer.running:
            self.sequencer.advance()        # lights off, dark, then next lamp
        else:
            self._begin_manual()            # keep watching, leave lamps alone

    def _begin_manual(self):
        """Arm the detector and the ring buffer without touching the lamps."""
        self.trigger.arm()
        self.flir_caching(True)

    def _log_event(self, n, t, lamp, n_frames):
        if self.events_path is None:
            return
        with open(self.events_path, 'a') as f:
            f.write(f'{n},{t:.3f},{lamp},{n_frames}\n')

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
            f.write(f'serial_port\t{SERIAL_PORT}\n')
            f.write(f'relay_port\t{RELAY_PORT}\n')
            f.write(f'delay_ms\t{DELAY_MS}\n')
            f.write('mode\tcharge\n')
            f.write(f'charge_rise_pC\t{CHARGE_RISE_PC}\n')
            f.write(f'trigger_window_s\t{TRIGGER_WINDOW_S}\n')
            f.write(f'dark_s\t{DARK_S}\n')
            f.write(f'light_max_s\t{LIGHT_MAX_S}\n')
            f.write(f'automation\t{self.auto_var.get()}\n')
            f.write(f'flir_enabled\t{FLIR_ENABLED}\n')
            if FLIR_ENABLED:
                f.write(f'flir_fps\t{FLIR_FPS}\n')
                f.write(f'flir_roi\t{FLIR_ROI}\n')
                f.write(f'flir_mono16\t{FLIR_MONO16}\n')
                f.write(f'cache_seconds\t{CACHE_SECONDS}\n')
                f.write(f'cache_frames\t{CACHE_FRAMES}\n')
                f.write(f'post_roll_s\t{POST_ROLL_S}\n')

        self.events_path = self.session_dir / 'events.csv'
        with open(self.events_path, 'w') as f:
            f.write('event,time,lamp,frames\n')

        self.writer = None
        self.save_queue = None
        if self.cam is not None and self.cam.error is None \
                and self.cam.is_alive():
            # Unbounded: a burst is handed over in one go and must not drop.
            self.save_queue = queue.Queue()
            self.writer = FrameWriter(
                self.save_queue,
                self.session_dir / 'flir',
                self.session_dir / 'flir_frames.csv',
            )
            self.writer.start()
            self.cam.arm(self.save_queue, t0)
        elif self.cam is not None:
            print('[flir] not recording (dead or errored)')

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

        self.root.after(50, self._poll_queue)

    def stop(self):
        self.recording = False
        self.sequencer.stop()
        self.send_relay(LAMPS_OFF_CMD, from_auto=True)

        if self.acq_thread is not None:
            self.acq_thread.stop()
            self.acq_thread.join(timeout=6)

        self._finish_writer()

        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

        good = int(np.count_nonzero(~np.isnan(self.q_vec))) \
            if self.q_vec else 0
        frames = self.cam.saved if self.cam else 0
        self.status_var.set(
            f'Stopped. {good}/{len(self.q_vec)} valid readings, '
            f'{self.blip_n} blips, {frames} frames '
            f'-> {self.session_dir.name}'
        )

    def _finish_writer(self):
        """Let the camera hand over anything queued, then close the writer."""
        if self.cam is not None:
            self.cam.caching.clear()
        time.sleep(0.2)
        if self.save_queue is not None:
            self.save_queue.put(None)
        if self.writer is not None:
            self.writer.join(timeout=120)
        self.save_queue = None
        self.writer = None

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

            frames = self.cam.saved if self.cam else 0
            self.status_var.set(
                f'Recording -> {self.session_dir.name}  '
                f'({len(self.t_vec)} samples, '
                f'{self.blip_n} blips, {frames} frames)'
            )

        if ended:
            self.recording = False
            self.sequencer.stop()
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
        if self.acq_thread is not None and self.acq_thread.is_alive():
            self.acq_thread.stop()
            self.acq_thread.join(timeout=6)
        self._finish_writer()
        if self.cam is not None:
            self.cam.stop()
            self.cam.join(timeout=3)
        if self.relay_ser is not None:
            try:
                self.relay_ser.write(LAMPS_OFF_CMD.encode())
                self.relay_ser.close()
            except Exception:
                pass
        self.root.destroy()


if __name__ == '__main__':
    print(f'[cache] {CACHE_FRAMES} frames @ {FLIR_FPS} fps = {CACHE_SECONDS} s'
          f', ~{cache_ram_mb():.0f} MB RAM')
    root = tk.Tk()
    app = ElectrometerApp(root)
    root.mainloop()
