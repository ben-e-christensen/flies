import os

# Must be set before cv2 is imported. Kills the slow MSMF hardware-transform path.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import queue
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
SERIAL_PORT   = 'COM19'
BAUDRATE      = 9600
PLOT_WINDOW_S = 10            # width of the scrolling view
ECHO_RAW      = False         # printing every sample slows the whole GUI down

# --- cameras ---
CAMERAS = [
    {'name': 'cam0', 'index': 0},
    {'name': 'cam1', 'index': 1},
]
CAM_BACKEND   = cv2.CAP_DSHOW
CAM_WIDTH     = 1280
CAM_HEIGHT    = 720
FORCE_MJPG    = True          # required for 2x 720p on one USB controller
SAVE_FPS      = 15
JPEG_QUALITY  = 90
QUEUE_MAX     = 150           # ~10 s of slack if the disk stalls

# --- preview ---
PREVIEW_WIDTH = 360
PREVIEW_MS    = 66            # ~15 Hz redraw
# ================================

SETUP_CMD = (b"*RST; :SYST:ZCH ON; :SENS:FUNC 'CHAR'; CHAR:RANG 20e-9; "
             b":SENS:CHAR:NPLC 1; :FORM:ELEM READ; :SYST:ZCH OFF; "
             b":CALC2:NULL:STAT ON\n")


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

class FrameWriter(threading.Thread):
    """Pulls frames off a queue and writes JPEGs + an index CSV.

    Kept separate from capture so a slow disk never stalls the camera.
    """

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
            params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            with open(self.index_path, 'w') as idx:
                idx.write('time,filename\n')
                while True:
                    item = self.in_queue.get()
                    if item is None:
                        break
                    t, frame, n = item
                    fname = f'{n:06d}.jpg'
                    cv2.imwrite(str(self.out_dir / fname), frame, params)
                    idx.write(f'{t:.6f},{fname}\n')
                    self.written += 1
                    if self.written % SAVE_FPS == 0:
                        idx.flush()
                idx.flush()
        except Exception:
            self.error = traceback.format_exc()
            print(self.error)


class CameraThread(threading.Thread):
    """Owns one VideoCapture for the life of the app.

    Reads continuously (so the driver buffer never backs up and the preview
    stays live), but only forwards frames to the writer at SAVE_FPS while
    recording is set.
    """

    def __init__(self, name, index):
        super().__init__(daemon=True)
        self.cam_name = name
        self.index = index

        self.save_queue = None                 # set by the GUI before recording
        self.frame_no = 0
        self.t0 = 0.0

        self.recording = threading.Event()
        self.opened = threading.Event()
        self._stop_event = threading.Event()

        self._latest = None
        self._lock = threading.Lock()

        self.saved = 0
        self.dropped = 0
        self.status = 'opening...'
        self.error = None

    # -- called from the GUI thread --

    def stop(self):
        self._stop_event.set()

    def get_latest(self):
        with self._lock:
            return self._latest

    def arm(self, save_queue, t0):
        """Attach a writer queue and reset counters. Call before recording.set()."""
        self.frame_no = 0
        self.saved = 0
        self.dropped = 0
        self.t0 = t0
        self.save_queue = save_queue

    # -- camera thread --

    def _open(self):
        cap = cv2.VideoCapture(self.index, CAM_BACKEND)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f'could not open camera index {self.index}')

        # FOURCC before the frame size, or DSHOW may ignore it.
        if FORCE_MJPG:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_s = ''.join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))
        self.status = f'{w}x{h} {fourcc_s.strip()}'
        print(f'[{self.cam_name}] opened index {self.index}: {self.status}')
        return cap

    def run(self):
        cap = None
        was_recording = False
        next_save = 0.0
        interval = 1.0 / SAVE_FPS
        fail_streak = 0

        try:
            cap = self._open()
            self.opened.set()

            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    fail_streak += 1
                    if fail_streak > 100:
                        raise RuntimeError('camera stopped returning frames')
                    time.sleep(0.01)
                    continue
                fail_streak = 0
                now = time.perf_counter()

                with self._lock:
                    self._latest = frame

                if self.recording.is_set():
                    if not was_recording:
                        was_recording = True
                        next_save = now           # save immediately on arming
                    if now >= next_save:
                        # Don't let drift accumulate, but don't burst-catch-up
                        # either if we fell badly behind.
                        next_save += interval
                        if next_save < now:
                            next_save = now + interval
                        try:
                            self.save_queue.put_nowait(
                                (now - self.t0, frame, self.frame_no)
                            )
                            self.frame_no += 1
                            self.saved += 1
                        except queue.Full:
                            self.dropped += 1
                elif was_recording:
                    # Sentinel is pushed from this thread so it can never race
                    # ahead of a frame we already queued.
                    was_recording = False
                    if self.save_queue is not None:
                        try:
                            self.save_queue.put_nowait(None)
                        except queue.Full:
                            self.save_queue.put(None)

        except Exception:
            self.error = traceback.format_exc()
            self.status = 'ERROR - see console'
            print(f'[{self.cam_name}] {self.error}')
        finally:
            if was_recording and self.save_queue is not None:
                try:
                    self.save_queue.put_nowait(None)
                except queue.Full:
                    pass
            if cap is not None:
                cap.release()
            self.opened.set()      # unblock anything waiting, even on failure


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
        ttk.Label(controls, textvariable=self.status_var).pack(side=tk.LEFT, padx=12)

        # --- body: plot left, camera previews right ---
        body = ttk.Frame(root)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('time [s]')
        self.ax.set_ylabel('charge [pC]')
        self.line, = self.ax.plot([], [], 'ro-', markersize=3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=body)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        cam_panel = ttk.Frame(body, padding=4)
        cam_panel.pack(side=tk.RIGHT, fill=tk.Y)

        # Cameras open once at startup and stay open. Opening takes several
        # seconds on Windows; doing it here means Start is instant and the
        # electrometer and cameras share a timebase from the first sample.
        self.cams = []
        self.cam_labels = []
        self.cam_status_vars = []

        for cfg in CAMERAS:
            frame = ttk.LabelFrame(cam_panel, text=cfg['name'], padding=4)
            frame.pack(side=tk.TOP, fill=tk.X, pady=4)

            label = tk.Label(frame, width=PREVIEW_WIDTH,
                             height=int(PREVIEW_WIDTH * CAM_HEIGHT / CAM_WIDTH),
                             background='#222')
            label.pack()
            self.cam_labels.append(label)

            svar = tk.StringVar(value='opening...')
            ttk.Label(frame, textvariable=svar).pack(anchor='w')
            self.cam_status_vars.append(svar)

            cam = CameraThread(cfg['name'], cfg['index'])
            cam.start()
            self.cams.append(cam)

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
            f.write(f'save_fps\t{SAVE_FPS}\n')
            f.write(f'resolution\t{CAM_WIDTH}x{CAM_HEIGHT}\n')
            for cam in self.cams:
                f.write(f'{cam.cam_name}\tindex={cam.index}\t{cam.status}\n')

        # cameras first, so they are armed before the first charge sample
        self.writers = []
        for cam in self.cams:
            if cam.error is not None or not cam.is_alive():
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

        for writer in self.writers:
            writer.join(timeout=15)            # drains whatever is still queued

        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

        good = int(np.count_nonzero(~np.isnan(self.q_vec))) if self.q_vec else 0
        cam_bits = ' | '.join(
            f'{c.cam_name}: {c.saved} saved'
            + (f', {c.dropped} DROPPED' if c.dropped else '')
            for c in self.cams
        )
        self.status_var.set(
            f'Stopped. {good}/{len(self.q_vec)} valid charge. '
            f'{cam_bits}. -> {self.session_dir.name}'
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
            dropped = sum(c.dropped for c in self.cams)
            msg = (f'Recording -> {self.session_dir.name}  '
                   f'({len(self.t_vec)} samples, {frames} frames)')
            if dropped:
                msg += f'  [{dropped} frames dropped]'
            self.status_var.set(msg)

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
            frame = cam.get_latest()
            if frame is not None:
                h, w = frame.shape[:2]
                new_h = max(1, int(h * PREVIEW_WIDTH / w))
                small = cv2.resize(frame, (PREVIEW_WIDTH, new_h),
                                   interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                photo = ImageTk.PhotoImage(Image.fromarray(rgb))
                label.configure(image=photo, width=PREVIEW_WIDTH, height=new_h)
                label.image = photo        # keep a ref or Tk garbage-collects it

            state = cam.status
            if cam.recording.is_set():
                state += f'  REC {cam.saved}'
                if cam.dropped:
                    state += f'  drop {cam.dropped}'
            svar.set(state)

        self.root.after(PREVIEW_MS, self._update_previews)

    def _on_close(self):
        self.recording = False
        for cam in self.cams:
            cam.recording.clear()
            cam.stop()
        if self.acq_thread is not None and self.acq_thread.is_alive():
            self.acq_thread.stop()
            self.acq_thread.join(timeout=6)
        for writer in self.writers:
            writer.join(timeout=15)
        for cam in self.cams:
            cam.join(timeout=3)
        self.root.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    app = ElectrometerApp(root)
    root.mainloop()