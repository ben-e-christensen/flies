"""Electrometer (charge mode) + Arduino relay control.

No cameras — just the Keithley 6514 and relay buttons.

Output:
    Kiethley_data/session_<stamp>/
        electrometer.csv          time,charge
"""

import glob
import platform
import queue
import threading
import time
import tkinter as tk
import traceback
from datetime import datetime
from pathlib import Path
from tkinter import ttk

import numpy as np
import serial
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ============ CONFIG ============
ROOT_FOLDER = Path(__file__).resolve().parent / 'Kiethley_data'
ROOT_FOLDER.mkdir(exist_ok=True)

IS_LINUX = platform.system() == 'Linux'

# --- electrometer ---
DELAY_MS      = 5
PREFACTOR     = 1e12          # C -> pC
SERIAL_PORT   = '/dev/ttyUSB0' if IS_LINUX else 'COM25'
BAUDRATE      = 9600
PLOT_WINDOW_S = 10
ECHO_RAW      = False

# --- Arduino relay controller ---
RELAY_ENABLED = True
RELAY_PORT    = '/dev/ttyACM0' if IS_LINUX else 'COM24'
RELAY_BAUD    = 9600
# ================================

SETUP_CMD = (b"*RST; :SYST:ZCH ON; :SENS:FUNC 'CHAR'; CHAR:RANG 20e-9; "
             b":SENS:CHAR:NPLC 1; :FORM:ELEM READ; :SYST:ZCH OFF; "
             b":CALC2:NULL:STAT ON\n")


# ---------------------------------------------------------------- electrometer

class AcquisitionThread(threading.Thread):
    """Owns the serial port. Pushes (t, q) samples to a queue for the GUI."""

    def __init__(self, port, baudrate, delay_ms, out_queue, filepath):
        super().__init__(daemon=True)
        self.port = port
        self.baudrate = baudrate
        self.delay_s = delay_ms / 1000
        self.out_queue = out_queue
        self.filepath = filepath
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
                t0 = time.perf_counter()

                while not self._stop_event.is_set():
                    t = time.perf_counter() - t0
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

        self.out_queue.put(None)


# ------------------------------------------------------------------- GUI

class ElectrometerApp:

    def __init__(self, root):
        self.root = root
        self.root.title('Electrometer (Charge) + Relays')

        self.acq_thread = None
        self.data_queue = None
        self.recording = False
        self.t_vec = []
        self.q_vec = []
        self.filepath = None

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

        self.status_var = tk.StringVar(value='Idle')
        ttk.Label(controls, textvariable=self.status_var).pack(side=tk.LEFT,
                                                               padx=12)

        # --- relay buttons ---
        relay_frame = ttk.LabelFrame(controls, text='Relays', padding=4)
        relay_frame.pack(side=tk.RIGHT, padx=8)

        ttk.Button(relay_frame, text='Lamp A',
                   command=lambda: self._relay_send('a')).pack(side=tk.LEFT,
                                                               padx=2)
        ttk.Button(relay_frame, text='Lamp B',
                   command=lambda: self._relay_send('b')).pack(side=tk.LEFT,
                                                               padx=2)
        ttk.Button(relay_frame, text='Both Off',
                   command=lambda: self._relay_send('o')).pack(side=tk.LEFT,
                                                               padx=2)

        self.relay_var = tk.StringVar(
            value='connected' if self.relay_ser else 'not connected')
        ttk.Label(relay_frame, textvariable=self.relay_var).pack(side=tk.LEFT,
                                                                  padx=6)

        # --- plot ---
        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('time [s]')
        self.ax.set_ylabel('charge [pC]')
        self.line, = self.ax.plot([], [], 'ro-', markersize=3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.root.protocol('WM_DELETE_WINDOW', self._on_close)

    # ---------- relay ----------

    def _relay_send(self, cmd):
        """Send a single character to the Arduino relay controller."""
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

    # ---------- button handlers ----------

    def start(self):
        self.t_vec = []
        self.q_vec = []
        self.line.set_data([], [])
        self.canvas.draw_idle()

        stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        session_dir = ROOT_FOLDER / f'session_{stamp}'
        session_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = session_dir / 'electrometer.csv'

        self.data_queue = queue.Queue()
        self.acq_thread = AcquisitionThread(
            SERIAL_PORT, BAUDRATE, DELAY_MS, self.data_queue, self.filepath
        )
        self.acq_thread.start()

        self.recording = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set(f'Recording -> {session_dir.name}')

        self.root.after(50, self._poll_queue)

    def stop(self):
        self.recording = False
        if self.acq_thread is not None:
            self.acq_thread.stop()
            self.acq_thread.join(timeout=6)

        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        good = int(np.count_nonzero(~np.isnan(self.q_vec))) if self.q_vec else 0
        self.status_var.set(
            f'Stopped. {good}/{len(self.q_vec)} valid. '
            f'Saved to {self.filepath.name}'
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
            self.status_var.set(
                f'Recording ({len(self.t_vec)} samples)'
            )

        if ended:
            self.recording = False
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            if self.acq_thread.error:
                self.status_var.set('THREAD DIED — see console for traceback')
            return

        if self.recording:
            self.root.after(50, self._poll_queue)

    def _on_close(self):
        self.recording = False
        if self.acq_thread is not None and self.acq_thread.is_alive():
            self.acq_thread.stop()
            self.acq_thread.join(timeout=6)
        if self.relay_ser is not None:
            try:
                self.relay_ser.write(b'o')
                self.relay_ser.close()
            except Exception:
                pass
        self.root.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    app = ElectrometerApp(root)
    root.mainloop()