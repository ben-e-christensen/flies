"""Electrometer (voltage mode) + dual Brio 101 + FLIR Grasshopper3 logger.

Keithley 6514 measures voltage across a known parallel RC network.
Charge is derived in real time as Q = C × V.

    R = 5.15 MΩ,  C = 93.4 nF  →  τ ≈ 0.48 s

Brio webcams are captured via OpenCV (cross-platform).
FLIR Grasshopper3 is captured via PySpin / Spinnaker SDK.

Output:
    Kiethley_data/session_<stamp>/
        meta.txt
        electrometer.csv          time,voltage,charge
        cam0/000000.jpg ...       (Brio 0)
        cam0_frames.csv
        cam1/... cam1_frames.csv  (Brio 1)
        flir/000000.png ...       (Grasshopper3, 16-bit mono)
        flir_frames.csv
"""

import os
import platform
import queue
import subprocess
import threading
import time
import tkinter as tk
import traceback
from datetime import datetime
from pathlib import Path
from tkinter import ttk

import cv2
import numpy as np
import serial
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ============ CONFIG ============
ROOT_FOLDER = Path(__file__).resolve().parent / 'Kiethley_data'
ROOT_FOLDER.mkdir(exist_ok=True)

IS_LINUX = platform.system() == 'Linux'

# --- electrometer ---
DELAY_MS      = 5
SERIAL_PORT   = '/dev/ttyUSB0' if IS_LINUX else 'COM4'
BAUDRATE      = 9600
PLOT_WINDOW_S = 10
ECHO_RAW      = False

# --- RC circuit ---
CAP_F         = 93.4e-9      # 93.4 nF
RES_OHM       = 5.15e6       # 5.15 MΩ
TAU_S         = RES_OHM * CAP_F   # ≈ 0.481 s

# --- Brio webcams (OpenCV) ---
BRIO_INDICES   = [0,1]          # v4l2 /dev/video indices or DShow indices
BRIO_WIDTH     = 1920
BRIO_HEIGHT    = 1080
BRIO_FPS       = 15
BRIO_SAVE_FPS  = 5
BRIO_QUEUE_MAX = 200

# --- FLIR Grasshopper3 (PySpin) ---
FLIR_ENABLED   = True            # set False if Spinnaker not installed
FLIR_FPS       = 10              # acquisition frame rate
FLIR_SAVE_FPS  = 5
FLIR_QUEUE_MAX = 100
FLIR_SAVE_FMT  = '.png'         # .png for 16-bit mono, .jpg for 8-bit

# --- preview ---
PREVIEW_W  = 360
PREVIEW_MS = 150
# ================================

BRIO_SAVE_EVERY = max(1, round(BRIO_FPS / BRIO_SAVE_FPS))
FLIR_SAVE_EVERY = max(1, round(FLIR_FPS / FLIR_SAVE_FPS))

SETUP_CMD = (b"*RST; :SYST:ZCH ON; :SENS:FUNC 'VOLT'; :VOLT:RANG 2; "
             b":SENS:VOLT:NPLC 1; :FORM:ELEM READ; :SYST:ZCH OFF; "
             b":CALC2:NULL:STAT ON\n")

# Try importing PySpin
try:
    import PySpin
    HAS_PYSPIN = True
except ImportError:
    HAS_PYSPIN = False
    if FLIR_ENABLED:
        print('[FLIR] PySpin not installed — FLIR camera disabled')
        FLIR_ENABLED = False


# ----------------------------------------------------------- electrometer

class AcquisitionThread(threading.Thread):
    """Owns the serial port.  Pushes (t, V) samples to a queue for the GUI."""

    def __init__(self, port, baudrate, delay_ms, out_queue, filepath, t0):
        super().__init__(daemon=True)
        self.port = port
        self.baudrate = baudrate
        self.delay_s = delay_ms / 1000
        self.out_queue = out_queue
        self.filepath = filepath
        self.t0 = t0
        self._stop_event = threading.Event()
        self.error = None

    def stop(self):
        self._stop_event.set()

    def run(self):
        try:
            with serial.Serial(self.port, self.baudrate, timeout=5) as em, \
                 open(self.filepath, 'w') as f:

                f.write('time,voltage,charge\n')
                f.flush()
                em.write(SETUP_CMD)

                while not self._stop_event.is_set():
                    t = time.perf_counter() - self.t0
                    em.write(b'READ?\r')
                    raw = em.readline()
                    if ECHO_RAW:
                        print(repr(raw))
                    try:
                        v = float(raw)
                    except ValueError:
                        v = np.nan

                    q = v * CAP_F if not np.isnan(v) else np.nan
                    f.write(f'{t},{v},{q}\n')
                    f.flush()
                    self.out_queue.put((t, v))
                    time.sleep(self.delay_s)

        except Exception:
            self.error = traceback.format_exc()
            print(self.error)

        self.out_queue.put(None)


# ----------------------------------------------------------- OpenCV webcam

class OpenCVCamera(threading.Thread):
    """Cross-platform webcam capture via OpenCV."""

    def __init__(self, dev_index, name):
        super().__init__(daemon=True)
        self.dev_index = dev_index
        self.cam_name = name

        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest_frame = None   # numpy BGR

        self.save_queue = None
        self.recording = threading.Event()
        self._was_recording = False
        self._decimate = 0
        self.t0 = 0.0
        self.n = 0
        self.saved = 0
        self.dropped = 0
        self.received = 0
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
            return cv2.cvtColor(self._latest_frame, cv2.COLOR_BGR2RGB)

    def arm(self, save_queue, t0):
        self.n = 0
        self.saved = 0
        self.dropped = 0
        self._decimate = 0
        self.t0 = t0
        self.save_queue = save_queue

    def status_line(self):
        if self.error:
            return 'ERROR - see console'
        if self.received == 0:
            return 'waiting for stream...'
        s = f'{self.fps:.1f} fps in'
        if self.recording.is_set():
            s += f'  REC {self.saved} saved'
            if self.dropped:
                s += f'  {self.dropped} DROPPED'
        return s

    def run(self):
        backend = cv2.CAP_V4L2 if IS_LINUX else cv2.CAP_DSHOW
        cap = cv2.VideoCapture(self.dev_index, backend)
        if not cap.isOpened():
            self.error = f'cannot open camera index {self.dev_index}'
            print(f'[{self.cam_name}] {self.error}')
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, BRIO_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, BRIO_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, BRIO_FPS)
        if IS_LINUX:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    continue
                self._on_frame(frame)
        except Exception:
            self.error = traceback.format_exc()
            print(f'[{self.cam_name}] {self.error}')
        finally:
            cap.release()
            self._finish_recording()

    def _on_frame(self, frame):
        now = time.perf_counter()
        self.received += 1
        with self._lock:
            self._latest_frame = frame

        self._fps_n += 1
        if now - self._fps_t >= 1.0:
            self.fps = self._fps_n / (now - self._fps_t)
            self._fps_n = 0
            self._fps_t = now

        if self.recording.is_set():
            self._was_recording = True
            if self._decimate % BRIO_SAVE_EVERY == 0:
                # encode to JPEG for saving
                _, jpeg = cv2.imencode('.jpg', frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 95])
                try:
                    self.save_queue.put_nowait((now - self.t0,
                                                bytes(jpeg), self.n))
                    self.n += 1
                    self.saved += 1
                except queue.Full:
                    self.dropped += 1
            self._decimate += 1
        elif self._was_recording:
            self._finish_recording()

    def _finish_recording(self):
        if self._was_recording and self.save_queue is not None:
            self._was_recording = False
            try:
                self.save_queue.put_nowait(None)
            except queue.Full:
                pass


# ----------------------------------------------------------- FLIR / PySpin

class SpinnakerCamera(threading.Thread):
    """Grasshopper3 capture via PySpin (Spinnaker SDK)."""

    def __init__(self, name='flir'):
        super().__init__(daemon=True)
        self.cam_name = name

        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest_frame = None   # numpy mono8 for preview

        self.save_queue = None
        self.recording = threading.Event()
        self._was_recording = False
        self._decimate = 0
        self.t0 = 0.0
        self.n = 0
        self.saved = 0
        self.dropped = 0
        self.received = 0
        self.fps = 0.0
        self._fps_t = time.perf_counter()
        self._fps_n = 0
        self.error = None

    def stop(self):
        self._stop_event.set()

    def latest_frame_rgb(self):
        """Return a BGR→RGB preview frame (mono displayed as grey)."""
        with self._lock:
            if self._latest_frame is None:
                return None
            mono8 = self._latest_frame
            return cv2.cvtColor(mono8, cv2.COLOR_GRAY2RGB)

    def arm(self, save_queue, t0):
        self.n = 0
        self.saved = 0
        self.dropped = 0
        self._decimate = 0
        self.t0 = t0
        self.save_queue = save_queue

    def status_line(self):
        if self.error:
            return 'ERROR - see console'
        if self.received == 0:
            return 'waiting for stream...'
        s = f'{self.fps:.1f} fps in'
        if self.recording.is_set():
            s += f'  REC {self.saved} saved'
            if self.dropped:
                s += f'  {self.dropped} DROPPED'
        return s

    def run(self):
        system = None
        cam = None
        try:
            system = PySpin.System.GetInstance()
            cam_list = system.GetCameras()
            if cam_list.GetSize() == 0:
                self.error = 'no Spinnaker cameras found'
                print(f'[{self.cam_name}] {self.error}')
                cam_list.Clear()
                system.ReleaseInstance()
                return

            cam = cam_list[0]
            cam.Init()

            # --- configure ---
            nodemap = cam.GetNodeMap()

            # Continuous acquisition
            node_mode = PySpin.CEnumerationPtr(
                nodemap.GetNode('AcquisitionMode'))
            node_cont = node_mode.GetEntryByName('Continuous')
            node_mode.SetIntValue(node_cont.GetValue())

            # Frame rate
            try:
                node_fr_en = PySpin.CBooleanPtr(
                    nodemap.GetNode('AcquisitionFrameRateEnable'))
                if PySpin.IsAvailable(node_fr_en) and \
                   PySpin.IsWritable(node_fr_en):
                    node_fr_en.SetValue(True)

                node_fr = PySpin.CFloatPtr(
                    nodemap.GetNode('AcquisitionFrameRate'))
                if PySpin.IsAvailable(node_fr) and PySpin.IsWritable(node_fr):
                    node_fr.SetValue(min(FLIR_FPS, node_fr.GetMax()))
            except PySpin.SpinnakerException:
                pass  # some cams don't expose this node

            # Pixel format — try Mono16 first, fall back to Mono8
            self._pixel_fmt = 'Mono8'
            try:
                node_pf = PySpin.CEnumerationPtr(
                    nodemap.GetNode('PixelFormat'))
                entry_16 = node_pf.GetEntryByName('Mono16')
                if PySpin.IsAvailable(entry_16) and \
                   PySpin.IsReadable(entry_16):
                    node_pf.SetIntValue(entry_16.GetValue())
                    self._pixel_fmt = 'Mono16'
                else:
                    entry_8 = node_pf.GetEntryByName('Mono8')
                    node_pf.SetIntValue(entry_8.GetValue())
            except PySpin.SpinnakerException:
                pass

            print(f'[{self.cam_name}] pixel format: {self._pixel_fmt}')

            cam.BeginAcquisition()

            while not self._stop_event.is_set():
                try:
                    image = cam.GetNextImage(1000)  # 1 s timeout
                except PySpin.SpinnakerException:
                    continue

                if image.IsIncomplete():
                    image.Release()
                    continue

                w = image.GetWidth()
                h = image.GetHeight()

                if self._pixel_fmt == 'Mono16':
                    raw = np.frombuffer(image.GetData(), dtype=np.uint16
                                        ).reshape(h, w).copy()
                    preview = (raw >> 8).astype(np.uint8)
                else:
                    raw = np.frombuffer(image.GetData(), dtype=np.uint8
                                        ).reshape(h, w).copy()
                    preview = raw

                image.Release()
                self._on_frame(raw, preview)

            cam.EndAcquisition()

        except Exception:
            self.error = traceback.format_exc()
            print(f'[{self.cam_name}] {self.error}')
        finally:
            self._finish_recording()
            if cam is not None:
                try:
                    cam.DeInit()
                except Exception:
                    pass
                del cam
            if 'cam_list' in dir():
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

        if self.recording.is_set():
            self._was_recording = True
            if self._decimate % FLIR_SAVE_EVERY == 0:
                try:
                    self.save_queue.put_nowait((now - self.t0,
                                                raw, self.n))
                    self.n += 1
                    self.saved += 1
                except queue.Full:
                    self.dropped += 1
            self._decimate += 1
        elif self._was_recording:
            self._finish_recording()

    def _finish_recording(self):
        if self._was_recording and self.save_queue is not None:
            self._was_recording = False
            try:
                self.save_queue.put_nowait(None)
            except queue.Full:
                pass


# ----------------------------------------------------------- frame writer

class FrameWriter(threading.Thread):
    """Writes frames from a queue to disk.  Works for both JPEG and PNG."""

    def __init__(self, in_queue, out_dir, index_path, ext='.jpg'):
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
                idx.write('time,filename\n')
                while True:
                    item = self.in_queue.get()
                    if item is None:
                        break
                    t, data, n = item
                    fname = f'{n:06d}{self.ext}'
                    fpath = self.out_dir / fname

                    if self.ext == '.jpg' and isinstance(data, bytes):
                        # raw JPEG bytes from OpenCV imencode
                        with open(fpath, 'wb') as f:
                            f.write(data)
                    else:
                        # numpy array — use cv2.imwrite (handles 16-bit PNG)
                        cv2.imwrite(str(fpath), data)

                    idx.write(f'{t:.6f},{fname}\n')
                    self.written += 1
                    if self.written % 5 == 0:
                        idx.flush()
                idx.flush()
        except Exception:
            self.error = traceback.format_exc()
            print(self.error)


# ------------------------------------------------------------------- GUI

class ElectrometerApp:

    def __init__(self, root):
        self.root = root
        self.root.title(
            f'Electrometer (Voltage) + Cameras   '
            f'C={CAP_F*1e9:.1f} nF   R={RES_OHM/1e6:.2f} MΩ   '
            f'τ={TAU_S*1e3:.0f} ms'
        )

        self.acq_thread = None
        self.data_queue = None
        self.writers = []
        self.recording = False
        self.t_vec = []
        self.v_vec = []
        self.session_dir = None

        # --- controls ---
        controls = ttk.Frame(root, padding=8)
        controls.pack(side=tk.TOP, fill=tk.X)

        self.start_btn = ttk.Button(controls, text='Start',
                                    command=self.start)
        self.start_btn.pack(side=tk.LEFT, padx=4)

        self.stop_btn = ttk.Button(controls, text='Stop', command=self.stop,
                                   state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=4)

        self.status_var = tk.StringVar(value='Idle')
        ttk.Label(controls, textvariable=self.status_var).pack(side=tk.LEFT,
                                                               padx=12)

        # --- body: plots left, previews right ---
        body = ttk.Frame(root)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.ax_v = self.fig.add_subplot(211)
        self.ax_q = self.fig.add_subplot(212, sharex=self.ax_v)

        self.ax_v.set_ylabel('voltage [mV]')
        self.ax_v.tick_params(labelbottom=False)
        self.line_v, = self.ax_v.plot([], [], 'b.-', markersize=3,
                                      linewidth=0.8)

        self.ax_q.set_xlabel('time [s]')
        self.ax_q.set_ylabel('charge [pC]')
        self.line_q, = self.ax_q.plot([], [], 'r.-', markersize=3,
                                      linewidth=0.8)

        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=body)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH,
                                         expand=True)

        # --- camera previews (scrollable column) ---
        cam_panel = ttk.Frame(body, padding=4)
        cam_panel.pack(side=tk.RIGHT, fill=tk.Y)

        self.cams = []          # all camera threads (OpenCV + FLIR)
        self.cam_labels = []
        self.cam_status_vars = []
        self.cam_exts = []      # save extension per camera

        # Brio webcams
        for idx in BRIO_INDICES:
            name = f'cam{idx}'
            cam = OpenCVCamera(idx, name)
            self._add_cam_panel(cam_panel, cam, '.jpg')
            cam.start()
            time.sleep(0.3)

        # FLIR Grasshopper3
        if FLIR_ENABLED and HAS_PYSPIN:
            flir_cam = SpinnakerCamera('flir')
            self._add_cam_panel(cam_panel, flir_cam, FLIR_SAVE_FMT)
            flir_cam.start()

        self.root.protocol('WM_DELETE_WINDOW', self._on_close)
        self.root.after(PREVIEW_MS, self._update_previews)

    def _add_cam_panel(self, parent, cam, ext):
        frame = ttk.LabelFrame(parent, text=cam.cam_name, padding=4)
        frame.pack(side=tk.TOP, fill=tk.X, pady=4)

        label = tk.Label(frame, background='#222')
        label.pack()
        self.cam_labels.append(label)

        svar = tk.StringVar(value='starting...')
        ttk.Label(frame, textvariable=svar).pack(anchor='w')
        self.cam_status_vars.append(svar)

        self.cams.append(cam)
        self.cam_exts.append(ext)

    # ---------- button handlers ----------

    def start(self):
        self.t_vec = []
        self.v_vec = []
        self.line_v.set_data([], [])
        self.line_q.set_data([], [])
        self.canvas.draw_idle()

        stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.session_dir = ROOT_FOLDER / f'session_{stamp}'
        self.session_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()

        with open(self.session_dir / 'meta.txt', 'w') as f:
            f.write(f'wall_clock_start\t{datetime.now().isoformat()}\n')
            f.write(f'serial_port\t{SERIAL_PORT}\n')
            f.write(f'delay_ms\t{DELAY_MS}\n')
            f.write(f'mode\tvoltage\n')
            f.write(f'capacitance_F\t{CAP_F}\n')
            f.write(f'resistance_Ohm\t{RES_OHM}\n')
            f.write(f'tau_s\t{TAU_S}\n')
            f.write(f'brio_resolution\t{BRIO_WIDTH}x{BRIO_HEIGHT}\n')
            f.write(f'brio_capture_fps\t{BRIO_FPS}\n')
            f.write(f'brio_save_fps\t{BRIO_SAVE_FPS}\n')
            f.write(f'flir_enabled\t{FLIR_ENABLED}\n')
            f.write(f'flir_capture_fps\t{FLIR_FPS}\n')
            f.write(f'flir_save_fps\t{FLIR_SAVE_FPS}\n')
            for cam in self.cams:
                f.write(f'{cam.cam_name}\ttype={type(cam).__name__}\n')

        self.writers = []
        for cam, ext in zip(self.cams, self.cam_exts):
            if cam.error is not None or not cam.is_alive():
                print(f'[{cam.cam_name}] not recording (dead or errored)')
                continue

            q_max = FLIR_QUEUE_MAX if isinstance(cam, SpinnakerCamera) \
                else BRIO_QUEUE_MAX
            sq = queue.Queue(maxsize=q_max)
            writer = FrameWriter(
                sq,
                self.session_dir / cam.cam_name,
                self.session_dir / f'{cam.cam_name}_frames.csv',
                ext=ext,
            )
            writer.start()
            cam.arm(sq, t0)
            cam.recording.set()
            self.writers.append(writer)

        self.data_queue = queue.Queue()
        self.acq_thread = AcquisitionThread(
            SERIAL_PORT, BAUDRATE, DELAY_MS, self.data_queue,
            self.session_dir / 'electrometer.csv', t0
        )
        self.acq_thread.start()

        self.recording = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set(f'Recording -> {self.session_dir.name}')

        self.root.after(50, self._poll_queue)

    def stop(self):
        self.recording = False

        for cam in self.cams:
            cam.recording.clear()

        if self.acq_thread is not None:
            self.acq_thread.stop()
            self.acq_thread.join(timeout=6)

        time.sleep(0.2)
        for cam in self.cams:
            if cam.save_queue is not None:
                try:
                    cam.save_queue.put_nowait(None)
                except queue.Full:
                    pass
        for writer in self.writers:
            writer.join(timeout=20)

        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

        good = int(np.count_nonzero(~np.isnan(self.v_vec))) \
            if self.v_vec else 0
        cam_bits = ' | '.join(
            f'{c.cam_name}: {c.saved}'
            + (f' ({c.dropped} dropped)' if c.dropped else '')
            for c in self.cams
        )
        self.status_var.set(
            f'Stopped. {good}/{len(self.v_vec)} valid readings. '
            f'Frames {cam_bits}. -> {self.session_dir.name}'
        )

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
                t, v = item
                self.t_vec.append(t)
                self.v_vec.append(v)
                updated = True
        except queue.Empty:
            pass

        if updated:
            t_arr = np.array(self.t_vec)
            v_arr = np.array(self.v_vec)

            v_mv = v_arr * 1e3
            q_pc = v_arr * CAP_F * 1e12

            i0 = np.searchsorted(t_arr, t_arr[-1] - PLOT_WINDOW_S)
            self.line_v.set_data(t_arr[i0:], v_mv[i0:])
            self.line_q.set_data(t_arr[i0:], q_pc[i0:])

            right = max(t_arr[-1], PLOT_WINDOW_S)
            self.ax_v.set_xlim(right - PLOT_WINDOW_S, right)
            self.ax_v.relim()
            self.ax_v.autoscale_view(scalex=False)
            self.ax_q.relim()
            self.ax_q.autoscale_view(scalex=False)

            self.canvas.draw_idle()

            frames = sum(c.saved for c in self.cams)
            self.status_var.set(
                f'Recording -> {self.session_dir.name}  '
                f'({len(self.t_vec)} samples, {frames} frames)'
            )

        if ended:
            self.recording = False
            for cam in self.cams:
                cam.recording.clear()
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            if self.acq_thread.error:
                self.status_var.set('THREAD DIED - see console for traceback')
            return

        if self.recording:
            self.root.after(50, self._poll_queue)

    # ---------- previews ----------

    def _update_previews(self):
        for cam, label, svar in zip(self.cams, self.cam_labels,
                                    self.cam_status_vars):
            rgb = cam.latest_frame_rgb()
            if rgb is not None:
                h, w = rgb.shape[:2]
                new_h = max(1, int(h * PREVIEW_W / w))
                small = cv2.resize(rgb, (PREVIEW_W, new_h),
                                   interpolation=cv2.INTER_AREA)
                photo = ImageTk.PhotoImage(Image.fromarray(small))
                label.configure(image=photo)
                label.image = photo

            svar.set(cam.status_line())

        self.root.after(PREVIEW_MS, self._update_previews)

    def _on_close(self):
        self.recording = False
        for cam in self.cams:
            cam.recording.clear()
        if self.acq_thread is not None and self.acq_thread.is_alive():
            self.acq_thread.stop()
            self.acq_thread.join(timeout=6)
        time.sleep(0.2)
        for cam in self.cams:
            if cam.save_queue is not None:
                try:
                    cam.save_queue.put_nowait(None)
                except queue.Full:
                    pass
        for writer in self.writers:
            writer.join(timeout=20)
        for cam in self.cams:
            cam.stop()
        for cam in self.cams:
            cam.join(timeout=3)
        self.root.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    app = ElectrometerApp(root)
    root.mainloop()