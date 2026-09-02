"""Electrometer + dual Brio 101 logger.

Cameras are captured through ffmpeg subprocesses (-vcodec mjpeg), which is the
only path that gets real MJPG out of these cameras -- OpenCV's DSHOW backend
pins them to YUY2, capped at 5fps above 640x480. Frames arrive as camera-native
JPEGs and are written to disk verbatim: no decode, no re-encode.

All streams share one t0 (time.perf_counter at Start), so electrometer.csv and
camN_frames.csv timestamps are directly comparable.

Output:
    Kiethley_data/session_<stamp>/
        meta.txt
        electrometer.csv          time,charge
        cam0/000000.jpg ...
        cam0_frames.csv           time,filename
        cam1/... cam1_frames.csv
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
PREFACTOR     = 1e12          # C -> pC
SERIAL_PORT   = 'COM21'
BAUDRATE      = 9600
PLOT_WINDOW_S = 10
ECHO_RAW      = False

# --- cameras ---
FFMPEG = r'C:\Users\Ben\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.1-full_build\bin\ffmpeg.exe'
DEVICE_NAME    = 'Brio 101'
DEVICE_NUMBERS = [0, 1]       # -video_device_number for the two identical cams
WIDTH, HEIGHT  = 1920, 1080
CAPTURE_FPS    = 15           # what the camera streams (smooth-ish preview)
SAVE_FPS       = 5            # what hits the disk
QUEUE_MAX      = 200

# --- preview ---
PREVIEW_W  = 360
PREVIEW_MS = 150
# ================================

SAVE_EVERY = max(1, round(CAPTURE_FPS / SAVE_FPS))

SETUP_CMD = (b"*RST; :SYST:ZCH ON; :SENS:FUNC 'CHAR'; CHAR:RANG 20e-9; "
             b":SENS:CHAR:NPLC 1; :FORM:ELEM READ; :SYST:ZCH OFF; "
             b":CALC2:NULL:STAT ON\n")

SOI = b'\xff\xd8'
EOI = b'\xff\xd9'


# ---------------------------------------------------------------- electrometer

class AcquisitionThread(threading.Thread):
    """Owns the serial port. Pushes (t, q) samples to a queue for the GUI."""

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

                f.write('time,charge\n')
                f.flush()
                em.write(SETUP_CMD)

                while not self._stop_event.is_set():
                    t = time.perf_counter() - self.t0
                    em.write(b'READ?\r')
                    raw = em.readline()
                    if ECHO_RAW:
                        print(repr(raw))
                    try:
                        q = float(raw)
                    except ValueError:
                        q = np.nan

                    f.write(f'{t},{q}\n')
                    f.flush()
                    self.out_queue.put((t, q))
                    time.sleep(self.delay_s)

        except Exception:
            self.error = traceback.format_exc()
            print(self.error)

        self.out_queue.put(None)               # sentinel: acquisition ended


# --------------------------------------------------------------------- cameras

class FFmpegCamera(threading.Thread):
    """One ffmpeg subprocess piping camera-native MJPEG; splits into frames.

    Streams continuously from app launch. While `recording` is set, every
    SAVE_EVERY-th frame goes to the save queue with its shared-clock timestamp.
    """

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

    # -- GUI thread --

    def stop(self):
        self._stop_event.set()
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass

    def latest_jpeg(self):
        with self._lock:
            return self._latest_jpeg

    def arm(self, save_queue, t0):
        """Attach a writer queue and reset counters. Call before recording.set()."""
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

    # -- camera thread --

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
        # Sentinel sent from this thread so it cannot outrun a queued frame.
        if self._was_recording and self.save_queue is not None:
            self._was_recording = False
            try:
                self.save_queue.put_nowait(None)
            except queue.Full:
                pass


class FrameWriter(threading.Thread):
    """Writes camera-native JPEG bytes straight to disk plus an index CSV."""

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


# ------------------------------------------------------------------------- GUI

class ElectrometerApp:

    def __init__(self, root):
        self.root = root
        self.root.title('Electrometer + Cameras')

        self.acq_thread = None
        self.data_queue = None
        self.writers = []
        self.recording = False
        self.t_vec = []
        self.q_vec = []
        self.session_dir = None

        # --- controls ---
        controls = ttk.Frame(root, padding=8)
        controls.pack(side=tk.TOP, fill=tk.X)

        self.start_btn = ttk.Button(controls, text='Start', command=self.start)
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

        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('time [s]')
        self.ax.set_ylabel('charge [pC]')
        self.line, = self.ax.plot([], [], 'ro-', markersize=3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=body)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH,
                                         expand=True)

        cam_panel = ttk.Frame(body, padding=4)
        cam_panel.pack(side=tk.RIGHT, fill=tk.Y)

        # ffmpeg processes launch now and stream for the life of the app, so
        # Start is instant and all streams share the clock from sample one.
        self.cams = []
        self.cam_labels = []
        self.cam_status_vars = []

        for num in DEVICE_NUMBERS:
            name = f'cam{num}'
            frame = ttk.LabelFrame(cam_panel, text=name, padding=4)
            frame.pack(side=tk.TOP, fill=tk.X, pady=4)

            label = tk.Label(frame, background='#222')
            label.pack()
            self.cam_labels.append(label)

            svar = tk.StringVar(value='starting...')
            ttk.Label(frame, textvariable=svar).pack(anchor='w')
            self.cam_status_vars.append(svar)

            cam = FFmpegCamera(num, name)
            cam.start()
            self.cams.append(cam)
            time.sleep(0.4)        # stagger the two ffmpeg launches

        self.root.protocol('WM_DELETE_WINDOW', self._on_close)
        self.root.after(PREVIEW_MS, self._update_previews)

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
            f.write(f'delay_ms\t{DELAY_MS}\n')
            f.write(f'resolution\t{WIDTH}x{HEIGHT}\n')
            f.write(f'capture_fps\t{CAPTURE_FPS}\n')
            f.write(f'save_fps\t{SAVE_FPS}\n')
            for cam in self.cams:
                f.write(f'{cam.cam_name}\tdevice_number={cam.device_number}\n')

        # cameras first, so they are armed before the first charge sample
        self.writers = []
        for cam in self.cams:
            if cam.error is not None or not cam.is_alive():
                print(f'[{cam.cam_name}] not recording (dead or errored)')
                continue
            q = queue.Queue(maxsize=QUEUE_MAX)
            writer = FrameWriter(
                q,
                self.session_dir / cam.cam_name,
                self.session_dir / f'{cam.cam_name}_frames.csv',
            )
            writer.start()
            cam.arm(q, t0)
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
            self.acq_thread.join(timeout=6)    # readline can block up to 5 s

        # cameras push their own sentinel on the next frame; give them one
        # frame period, then push a fallback in case a stream has stalled
        time.sleep(1.5 / CAPTURE_FPS)
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

        good = int(np.count_nonzero(~np.isnan(self.q_vec))) if self.q_vec else 0
        cam_bits = ' | '.join(
            f'{c.cam_name}: {c.saved}'
            + (f' ({c.dropped} dropped)' if c.dropped else '')
            for c in self.cams
        )
        self.status_var.set(
            f'Stopped. {good}/{len(self.q_vec)} valid charge. '
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
                t, q = item
                self.t_vec.append(t)
                self.q_vec.append(q)
                updated = True
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
            jpeg = cam.latest_jpeg()
            if jpeg is not None:
                frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8),
                                     cv2.IMREAD_COLOR)
                if frame is not None:
                    h, w = frame.shape[:2]
                    new_h = max(1, int(h * PREVIEW_W / w))
                    small = cv2.resize(frame, (PREVIEW_W, new_h),
                                       interpolation=cv2.INTER_AREA)
                    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                    photo = ImageTk.PhotoImage(Image.fromarray(rgb))
                    label.configure(image=photo)
                    label.image = photo    # keep a ref or Tk drops it

            svar.set(cam.status_line())

        self.root.after(PREVIEW_MS, self._update_previews)

    def _on_close(self):
        self.recording = False
        for cam in self.cams:
            cam.recording.clear()
        if self.acq_thread is not None and self.acq_thread.is_alive():
            self.acq_thread.stop()
            self.acq_thread.join(timeout=6)
        time.sleep(1.5 / CAPTURE_FPS)
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