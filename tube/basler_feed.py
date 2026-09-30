#!/usr/bin/env python3
"""Basler live feed for Tkinter (no processing). See claude.md.

CameraPanel is a tk.Frame you can pack into any window. Run this file on its
own for a standalone camera window.
"""

import tkinter as tk
import threading, queue
from PIL import Image, ImageTk
import cv2
from pypylon import pylon

TARGET_UI_FPS = 30
DISPLAY_MAX = (800, 600)  # frames are shrunk to fit inside this (w, h)


class CameraGrabber(threading.Thread):
    def __init__(self, frame_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.q = frame_queue
        self.stop_event = stop_event
        self.cam = None
        self.err = None
        self.model = None

    def run(self):
        try:
            tl = pylon.TlFactory.GetInstance()
            if not tl.EnumerateDevices():
                self.err = "No Basler cameras found."
                return

            self.cam = pylon.InstantCamera(tl.CreateFirstDevice())
            self.cam.Open()
            self.model = self.cam.GetDeviceInfo().GetModelName()
            print("Opened:", self.model)

            self.cam.GainAuto.SetValue("Off")
            self.cam.Gain.SetValue(28.0)
            self.cam.PixelFormat.SetValue(self.cam.PixelFormat.GetSymbolics()[0])

            converter = pylon.ImageFormatConverter()
            if "Mono" in self.cam.PixelFormat.GetValue():
                converter.OutputPixelFormat = pylon.PixelType_Mono8
            else:
                converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            while not self.stop_event.is_set() and self.cam.IsGrabbing():
                grab = self.cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if grab.GrabSucceeded():
                    img = converter.Convert(grab).GetArray()
                    try:
                        self.q.put_nowait(img)
                    except queue.Full:
                        pass
                grab.Release()
        except Exception as e:
            self.err = f"{type(e).__name__}: {e}"
        finally:
            try:
                if self.cam:
                    if self.cam.IsGrabbing():
                        self.cam.StopGrabbing()
                    if self.cam.IsOpen():
                        self.cam.Close()
            except Exception:
                pass


class CameraPanel(tk.Frame):
    """Live Basler feed as an embeddable frame. Call stop() before closing."""

    def __init__(self, parent: tk.Misc, display_max=DISPLAY_MAX, **kw):
        super().__init__(parent, **kw)
        self.display_max = display_max

        self.label = tk.Label(self, bg="black")
        self.label.pack(fill="both", expand=True)
        self.info = tk.Label(self, text="Initializing camera…", anchor="w")
        self.info.pack(fill="x")

        self.frame_queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        self.grabber = CameraGrabber(self.frame_queue, self.stop_event)
        self.grabber.start()

        self.latest = None     # newest numpy frame, for anything that wants it later
        self._tk_image = None  # keep a reference or Tk garbage-collects the image
        self.after(100, self._update_ui)

    def _update_ui(self):
        if self.stop_event.is_set():
            return

        frame = None
        try:
            while True:  # drain to newest frame
                frame = self.frame_queue.get_nowait()
        except queue.Empty:
            pass

        if frame is not None:
            self.latest = frame
            if frame.ndim == 2:
                pil = Image.fromarray(frame)  # mono8
            else:
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pil.thumbnail(self.display_max)
            self._tk_image = ImageTk.PhotoImage(pil)
            self.label.configure(image=self._tk_image)

        if self.grabber.err:
            self.info.config(text=f"[!] {self.grabber.err}")
        elif self.grabber.model:
            self.info.config(text=f"Streaming: {self.grabber.model}")
        self.after(int(1000 / TARGET_UI_FPS), self._update_ui)

    def stop(self):
        # Wait for the grabber to release the camera. Exiting while it's still
        # inside pylon aborts the interpreter ("exception not rethrown").
        self.stop_event.set()
        self.grabber.join(timeout=6)  # RetrieveResult times out at 5 s


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Basler Camera Feed")
    panel = CameraPanel(root)
    panel.pack(fill="both", expand=True)

    def on_close():
        panel.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
