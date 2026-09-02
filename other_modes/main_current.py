"""Electrometer (current mode) + dual Brio 101 + FLIR Grasshopper3 logger.

Keithley 6514 in current mode; post-hoc integration recovers charge.
Brio webcams captured via ffmpeg subprocesses (-vcodec mjpeg).
FLIR Grasshopper3 captured via PySpin / Spinnaker SDK.

All streams share one t0 (time.perf_counter at Start), so electrometer.csv
and all *_frames.csv timestamps are directly comparable.

Output:
    Kiethley_data/session_<stamp>/
        meta.txt
        electrometer.csv          time,current
        cam0/000000.jpg ...       (Brio 0)
        cam0_frames.csv
        cam1/... cam1_frames.csv  (Brio 1)
        flir/000000.png ...       (Grasshopper3, 16-bit mono)
        flir_frames.csv
"""

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

# --- electrometer ---
DELAY_MS      = 5
PREFACTOR     = 1e9           # A -> nA
SERIAL_PORT   = 'COM22'
BAUDRATE      = 9600
PLOT_WINDOW_S = 10
ECHO_RAW      = False

# --- Brio webcams (ffmpeg / DirectShow) ---
FFMPEG = r'C:\Users\Ben\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.1-full_build\bin\ffmpeg.exe'
DEVICE_NAME    = 'Brio 101'
DEVICE_NUMBERS = [0]
WIDTH, HEIGHT  = 1920, 1080
CAPTURE_FPS    = 15
SAVE_FPS       = 5
QUEUE_MAX      = 200

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

SAVE_EVERY      = max(1, round(CAPTURE_FPS / SAVE_FPS))
FLIR_SAVE_EVERY = max(1, round(FLIR_FPS / FLIR_SAVE_FPS))

SETUP_CMD = (b"*RST; :SYST:ZCH ON; :SENS:FUNC 'CURR'; :CURR:RANG 20e-9; "
             b":SENS:CURR:NPLC 1; :FORM:ELEM READ; :SYST:ZCH OFF; "
             b":CALC2:NULL:STAT ON\n")

SOI = b'\xff\xd8'
EOI = b'\xff\xd9'

# Try importing PySpin
try:
    import PySpin
    HAS_PYSPIN = True
except ImportError:
    HAS_PYSPIN = False
    if FLIR_ENABLED:
        print('[FLIR] PySpin not installed — FLIR camera disabled')
        FLIR_ENABLED = False


# ---------------------------------------------------------------- electrometer

class AcquisitionThread(threading.Thread):
    """Owns the serial port. Pushes (t, I) samples to a queue for the GUI."""

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

                f.write('time,current\n')
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

                    f.write(f'{t},{val}\n')
                    f.flush()
                    self.out_queue.put((t, val))
                    time.sleep(self.delay_s)

        except Exception:
            self.error = traceback.format_exc()
            print(self.error)

        self.out_queue.put(None)


# --------------------------------------------------------- Brio (ffmpeg/dshow)

class FFmpegCamera(threading.Thread):
    """One ffmpeg subprocess piping camera-native MJPEG; splits into frames."""

    def __init__(self, device_number, name):
        super().__init__(daemon=True)
        self.device_number = device_number
        self.cam_name = name

        self.proc = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest_jpeg = None

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
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass

    def latest_frame_rgb(self):
        """Decode latest JPEG to RGB numpy array for preview."""
        with self._lock:
            jpeg = self._latest_jpeg
        if jpeg is None:
            return None
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

    def _cmd(self):
        return [
            FFMPEG, '-hide_banner', '-loglevel', 'error',
            '-f', 'dshow',
            '-video_device_number', str(self.device_number),
            '-vcodec', 'mjpeg',
            '-video_size', f'{WIDTH}x{HEIGHT}',
            '-framerate', str(CAPTURE_FPS),
            '-rtbufsize', '256M',
            '-i', f'video={DEVICE_NAME}',
            '-c:v', 'copy',
            '-f', 'mjpeg', 'pipe:1',
        ]

    def _drain_stderr(self):
        for line in self.proc.stderr:
            txt = line.decode('utf-8', 'replace').rstrip()
            if txt:
                print(f'[{self.cam_name}] {txt}')
                if self.error is None:
                    self.error = txt

    def run(self):
        try:
            self.proc = subprocess.Popen(
                self._cmd(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                bufsize=0
            )
        except FileNotFoundError:
            self.error = f'ffmpeg not found at {FFMPEG}'
            print(f'[{self.cam_name}] {self.error}')
            return

        threading.Thread(target=self._drain_stderr, daemon=True).start()

        buf = bytearray()
        try:
            while not self._stop_event.is_set():
                chunk = self.proc.stdout.read(65536)
                if not chunk:
                    break
                buf.extend(chunk)

                while True:
                    start = buf.find(SOI)
                    if start < 0:
                        buf.clear()
                        break
                    end = buf.find(EOI, start + 2)
                    if end < 0:
                        del buf[:start]
                        break
                    jpeg = bytes(buf[start:end + 2])
                    del buf[:end + 2]
                    self._on_frame(jpeg)
        except Exception:
            self.error = traceback.format_exc()
            print(f'[{self.cam_name}] {self.error}')
        finally:
            self._finish_recording()
            self.stop()

    def _on_frame(self, jpeg):
        now = time.perf_counter()
        self.received += 1
        with self._lock:
            self._latest_jpeg = jpeg

        self._fps_n += 1
        if now - self._fps_t >= 1.0:
            self.fps = self._fps_n / (now - self._fps_t)
            self._fps_n = 0
            self._fps_t = now

        if self.recording.is_set():
            self._was_recording = True
            if self._decimate % SAVE_EVERY == 0:
                try:
                    self.save_queue.put_nowait((now - self.t0, jpeg, self.n))
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


# --------------------------------------------------------- FLIR / PySpin

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
        """Return mono preview as RGB for the shared preview logic."""
        with self._lock:
            if self._latest_frame is None:
                return None
            return cv2.cvtColor(self._latest_frame, cv2.COLOR_GRAY2RGB)

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
                pass

            # Pixel format — try Mono16, fall back to Mono8
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
                    image = cam.GetNextImage(1000)
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
                    self.save_queue.put_nowait((now - self.t0, raw, self.n))
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


# ----------------------------------------------------------- frame writers

class JpegWriter(threading.Thread):
    """Writes raw JPEG bytes from ffmpeg straight to disk."""

    def __init__(self, in_queue, out_dir, index_path):
        super().__init__(daemon=True)
        self.in_queue = in_queue
        self.out_dir = out_dir
        self.index_path = index_path
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
                    t, jpeg, n = item
                    fname = f'{n:06d}.jpg'
                    with open(self.out_dir / fname, 'wb') as f:
                        f.write(jpeg)
                    idx.write(f'{t:.6f},{fname}\n')
                    self.written += 1
                    if self.written % SAVE_FPS == 0:
                        idx.flush()
                idx.flush()
        except Exception:
            self.error = traceback.format_exc()
            print(self.error)


class ArrayWriter(threading.Thread):
    """Writes numpy arrays to disk via cv2.imwrite (handles 16-bit PNG)."""

    def __init__(self, in_queue, out_dir, index_path, ext='.png'):
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
                    t, arr, n = item
                    fname = f'{n:06d}{self.ext}'
                    cv2.imwrite(str(self.out_dir / fname), arr)
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
        self.root.title('Electrometer (Current) + Brios + FLIR')

        self.acq_thread = None
        self.data_queue = None
        self.writers = []
        self.recording = False
        self.t_vec = []
        self.i_vec = []
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

        # --- body: plot left, previews right ---
        body = ttk.Frame(root)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.ax_i = self.fig.add_subplot(211)
        self.ax_q = self.fig.add_subplot(212, sharex=self.ax_i)

        self.ax_i.set_ylabel('current [nA]')
        self.ax_i.tick_params(labelbottom=False)
        self.line_i, = self.ax_i.plot([], [], 'ro-', markersize=3,
                                      linewidth=0.8)

        self.ax_q.set_xlabel('time [s]')
        self.ax_q.set_ylabel('charge [pC]')
        self.line_q, = self.ax_q.plot([], [], 'g.-', markersize=3,
                                      linewidth=0.8)

        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=body)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH,
                                         expand=True)

        cam_panel = ttk.Frame(body, padding=4)
        cam_panel.pack(side=tk.RIGHT, fill=tk.Y)

        # --- all camera objects share the same preview interface ---
        self.all_cams = []       # (cam_obj, writer_factory)
        self.cam_labels = []
        self.cam_status_vars = []

        # Brio webcams
        for num in DEVICE_NUMBERS:
            name = f'cam{num}'
            cam = FFmpegCamera(num, name)
            self._add_preview_panel(cam_panel, cam)
            self.all_cams.append((cam, 'jpeg'))
            cam.start()
            time.sleep(0.4)

        # FLIR Grasshopper3
        if FLIR_ENABLED and HAS_PYSPIN:
            flir = SpinnakerCamera('flir')
            self._add_preview_panel(cam_panel, flir)
            self.all_cams.append((flir, 'array'))
            flir.start()

        self.root.protocol('WM_DELETE_WINDOW', self._on_close)
        self.root.after(PREVIEW_MS, self._update_previews)

    def _add_preview_panel(self, parent, cam):
        frame = ttk.LabelFrame(parent, text=cam.cam_name, padding=4)
        frame.pack(side=tk.TOP, fill=tk.X, pady=4)

        label = tk.Label(frame, background='#222')
        label.pack()
        self.cam_labels.append(label)

        svar = tk.StringVar(value='starting...')
        ttk.Label(frame, textvariable=svar).pack(anchor='w')
        self.cam_status_vars.append(svar)

    # ---------- button handlers ----------

    def start(self):
        self.t_vec = []
        self.i_vec = []
        self.line_i.set_data([], [])
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
            f.write(f'mode\tcurrent\n')
            f.write(f'brio_resolution\t{WIDTH}x{HEIGHT}\n')
            f.write(f'brio_capture_fps\t{CAPTURE_FPS}\n')
            f.write(f'brio_save_fps\t{SAVE_FPS}\n')
            f.write(f'flir_enabled\t{FLIR_ENABLED}\n')
            if FLIR_ENABLED:
                f.write(f'flir_capture_fps\t{FLIR_FPS}\n')
                f.write(f'flir_save_fps\t{FLIR_SAVE_FPS}\n')
                f.write(f'flir_save_fmt\t{FLIR_SAVE_FMT}\n')
            for cam, _ in self.all_cams:
                f.write(f'{cam.cam_name}\ttype={type(cam).__name__}\n')

        # arm cameras and start writers
        self.writers = []
        for cam, wtype in self.all_cams:
            if cam.error is not None or not cam.is_alive():
                print(f'[{cam.cam_name}] not recording (dead or errored)')
                continue

            if wtype == 'jpeg':
                sq = queue.Queue(maxsize=QUEUE_MAX)
                writer = JpegWriter(
                    sq,
                    self.session_dir / cam.cam_name,
                    self.session_dir / f'{cam.cam_name}_frames.csv',
                )
            else:
                sq = queue.Queue(maxsize=FLIR_QUEUE_MAX)
                writer = ArrayWriter(
                    sq,
                    self.session_dir / cam.cam_name,
                    self.session_dir / f'{cam.cam_name}_frames.csv',
                    ext=FLIR_SAVE_FMT,
                )

            writer.start()
            cam.arm(sq, t0)
            cam.recording.set()
            self.writers.append(writer)

        # electrometer
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

        for cam, _ in self.all_cams:
            cam.recording.clear()

        if self.acq_thread is not None:
            self.acq_thread.stop()
            self.acq_thread.join(timeout=6)

        time.sleep(0.2)
        for cam, _ in self.all_cams:
            if cam.save_queue is not None:
                try:
                    cam.save_queue.put_nowait(None)
                except queue.Full:
                    pass
        for writer in self.writers:
            writer.join(timeout=20)

        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

        good = int(np.count_nonzero(~np.isnan(self.i_vec))) \
            if self.i_vec else 0
        cam_bits = ' | '.join(
            f'{c.cam_name}: {c.saved}'
            + (f' ({c.dropped} dropped)' if c.dropped else '')
            for c, _ in self.all_cams
        )
        self.status_var.set(
            f'Stopped. {good}/{len(self.i_vec)} valid readings. '
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
                t, val = item
                self.t_vec.append(t)
                self.i_vec.append(val)
                updated = True
        except queue.Empty:
            pass

        if updated:
            t_arr = np.array(self.t_vec)
            i_arr_raw = np.array(self.i_vec)
            i_arr_nA = i_arr_raw * PREFACTOR          # A -> nA

            # trapezoidal integration: current (A) -> charge (C) -> pC
            i_clean = np.where(np.isnan(i_arr_raw), 0.0, i_arr_raw)
            dt = np.diff(t_arr)
            avg_i = 0.5 * (i_clean[:-1] + i_clean[1:])
            q_arr_pC = np.concatenate(([0.0], np.cumsum(avg_i * dt))) * 1e12

            i0 = np.searchsorted(t_arr, t_arr[-1] - PLOT_WINDOW_S)
            self.line_i.set_data(t_arr[i0:], i_arr_nA[i0:])
            self.line_q.set_data(t_arr[i0:], q_arr_pC[i0:])

            right = max(t_arr[-1], PLOT_WINDOW_S)
            self.ax_i.set_xlim(right - PLOT_WINDOW_S, right)
            self.ax_i.relim()
            self.ax_i.autoscale_view(scalex=False)
            self.ax_q.relim()
            self.ax_q.autoscale_view(scalex=False)

            self.canvas.draw_idle()

            frames = sum(c.saved for c, _ in self.all_cams)
            self.status_var.set(
                f'Recording -> {self.session_dir.name}  '
                f'({len(self.t_vec)} samples, {frames} frames)'
            )

        if ended:
            self.recording = False
            for cam, _ in self.all_cams:
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
        for (cam, _), label, svar in zip(self.all_cams, self.cam_labels,
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
        for cam, _ in self.all_cams:
            cam.recording.clear()
        if self.acq_thread is not None and self.acq_thread.is_alive():
            self.acq_thread.stop()
            self.acq_thread.join(timeout=6)
        time.sleep(0.2)
        for cam, _ in self.all_cams:
            if cam.save_queue is not None:
                try:
                    cam.save_queue.put_nowait(None)
                except queue.Full:
                    pass
        for writer in self.writers:
            writer.join(timeout=20)
        for cam, _ in self.all_cams:
            cam.stop()
        for cam, _ in self.all_cams:
            cam.join(timeout=3)
        self.root.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    app = ElectrometerApp(root)
    root.mainloop()