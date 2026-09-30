"""Live 3-channel scope panel (Tk frame) fed by an AdcReader."""

import tkinter as tk

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from adc_serial import AdcReader

# ============ CONFIG ============
WINDOW_S = 10.0          # seconds of signal shown
PLOT_FPS = 20
# Indexed by ADC number (v0/v1/v2 in the data). ADDR pin sets which is which.
CHANNEL_NAMES = ["Bottom ring (0x48)", "Middle ring (0x49)", "Top ring (0x4A)"]
COLORS = ["tab:blue", "tab:orange", "tab:green"]
PLOT_ORDER = [2, 1, 0]  # plot panels top-to-bottom like the tube: top ring first
# ================================


class ScopePanel(tk.Frame):
    """Three stacked, live-updating traces from an AdcReader."""

    def __init__(self, parent, reader: AdcReader, **kw):
        super().__init__(parent, **kw)
        self.reader = reader
        self.paused = False
        self.window_s = tk.DoubleVar(value=WINDOW_S)
        self.autoscale = tk.BooleanVar(value=True)

        self.fig = Figure(figsize=(7, 6), dpi=100, layout="constrained")
        self.axes = self.fig.subplots(3, 1, sharex=True)
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
        t, vs = self.reader.snapshot()
        if len(t):
            keep = t >= t[-1] - win
            tr = t[keep] - t[-1]
            for line, ax, v in zip(self.lines, self.axes, [vs[i] for i in PLOT_ORDER]):
                vk = v[keep]
                line.set_data(tr, vk)
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
            txt = f"{r.port}  {self._rate:.0f} S/s"
            if r.info:
                missing = [s for s in r.info if "NOT FOUND" in s]
                if missing:
                    txt += "  [!] " + "; ".join(missing)
        else:
            txt = f"[!] {r.err or 'connecting to ' + r.port + '…'}"
        self.status.config(text=txt)
        self.after(1000, self._update_rate)
