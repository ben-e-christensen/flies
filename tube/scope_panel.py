"""Live scope panel (Tk frame): one trace per ADC channel, fed by an AdcReader."""

import tkinter as tk

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from adc_serial import AdcReader
from channels import CHANNEL_NAMES, COLORS, PLOT_ORDER

# ============ CONFIG ============
WINDOW_S = 10.0          # seconds of signal shown
PLOT_FPS = 20
MAX_POINTS = 2000        # per trace after min/max decimation (~ screen width)
# ================================


def minmax_decimate(t, y, max_points=MAX_POINTS):
    """Shrink a trace for drawing without hiding short spikes: keep the min and
    max of each bucket (so 2 points per bucket)."""
    n = len(t)
    if n <= max_points:
        return t, y
    per = int(np.ceil(n / (max_points // 2)))
    m = n // per * per
    yb = y[:m].reshape(-1, per)
    tb = t[:m].reshape(-1, per)
    lo, hi = np.nanargmin(yb, axis=1), np.nanargmax(yb, axis=1)
    first = np.minimum(lo, hi)
    second = np.maximum(lo, hi)
    rows = np.arange(len(yb))
    tt = np.column_stack([tb[rows, first], tb[rows, second]]).ravel()
    yy = np.column_stack([yb[rows, first], yb[rows, second]]).ravel()
    return tt, yy


class ScopePanel(tk.Frame):
    """Stacked, live-updating traces (one per channel) from an AdcReader."""

    def __init__(self, parent, reader: AdcReader, **kw):
        super().__init__(parent, **kw)
        self.reader = reader
        self.paused = False
        self.window_s = tk.DoubleVar(value=WINDOW_S)
        self.autoscale = tk.BooleanVar(value=True)

        self.fig = Figure(figsize=(7, 6), dpi=100, layout="constrained")
        self.axes = self.fig.subplots(len(PLOT_ORDER), 1, sharex=True)
        self.lines = []
        for ax, i in zip(self.axes, PLOT_ORDER):
            (line,) = ax.plot([], [], color=COLORS[i], lw=1)
            ax.set_ylabel(f"{CHANNEL_NAMES[i]}\nV (electrometer)")
            ax.grid(True, alpha=0.3)
            self.lines.append(line)
        self.axes[-1].set_xlabel("time (s, 0 = now)")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        bar = tk.Frame(self)
        bar.pack(fill="x")
        self.pause_btn = tk.Button(bar, text="Pause", width=8, command=self.toggle_pause)
        self.pause_btn.pack(side="left")
        tk.Button(bar, text="Clear", command=reader.clear).pack(side="left")
        tk.Checkbutton(bar, text="Autoscale Y", variable=self.autoscale).pack(side="left")
        tk.Label(bar, text="  Window (s):").pack(side="left")
        tk.Spinbox(bar, from_=1, to=60, increment=1, width=4,
                   textvariable=self.window_s).pack(side="left")
        self.status = tk.Label(bar, anchor="e")
        self.status.pack(side="right", fill="x", expand=True)

        self._last_n = 0
        self._rate = 0.0
        self.after(200, self._update)
        self.after(1000, self._update_rate)

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_btn.config(text="Resume" if self.paused else "Pause")

    def _window(self):
        try:
            return max(0.1, float(self.window_s.get()))
        except (tk.TclError, ValueError):
            return WINDOW_S

    def _update(self):
        if not self.paused:
            self._redraw()
        self.after(int(1000 / PLOT_FPS), self._update)

    def _redraw(self):
        win = self._window()
        t, vs = self.reader.snapshot(last_s=win)
        if len(t) and len(vs) == len(CHANNEL_NAMES):
            tr = t - t[-1]
            for line, ax, v in zip(self.lines, self.axes, [vs[i] for i in PLOT_ORDER]):
                td, vk = minmax_decimate(tr, v)
                line.set_data(td, vk)
                if self.autoscale.get():
                    finite = vk[np.isfinite(vk)]
                    if finite.size:
                        lo, hi = finite.min(), finite.max()
                        pad = max((hi - lo) * 0.1, 1e-3)
                        ax.set_ylim(lo - pad, hi + pad)
        self.axes[0].set_xlim(-win, 0)
        self.canvas.draw_idle()

    def _update_rate(self):
        r = self.reader
        n = r.n_samples
        self._rate = n - self._last_n
        self._last_n = n
        if r.connected:
            txt = f"{r.port}  {self._rate:.0f} S/s per channel"
            if r.n_gaps:
                txt += f"  [!] {r.n_gaps} gaps ({r.n_missing} samples lost)"
            warn = [s for s in r.info if "WARNING" in s]
            if warn:
                txt += "  [!] " + warn[0].lstrip("# ")
        else:
            txt = f"[!] {r.err or 'connecting to ' + r.port + '…'}"
        self.status.config(text=txt)
        self.after(1000, self._update_rate)
