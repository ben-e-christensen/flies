#!/usr/bin/env python3
"""Tube rig: Basler camera + 3x ADS1115 (via ESP32-S3) in one window.

Runs three things at once:
    - AdcReader thread     reads t_us,raw0,raw1,raw2 lines from the ESP32, converts to volts
    - CameraGrabber thread pulls frames from the Basler camera
    - Tk main loop         shows the camera (left) and the 3 traces (right)

    python3 main.py                  # auto-detect the ESP32
    python3 main.py --port /dev/ttyACM1
    python3 main.py --no-camera      # scope only
    python3 main.py --flash          # upload esp32_ads firmware first
"""

import argparse
import signal
import subprocess
import sys
import tkinter as tk
from pathlib import Path

from adc_serial import AdcReader, find_port
from scope_panel import ScopePanel

# ============ CONFIG ============
FQBN = "esp32:esp32:adafruit_feather_esp32s3"
SKETCH = Path(__file__).resolve().parent / "esp32_ads"
# ================================


def flash_firmware(port):
    """Compile and upload the ESP32 sketch. Exits if arduino-cli fails."""
    print(f"Flashing {SKETCH.name} to {port} ...")
    cmd = ["arduino-cli", "compile", "--upload", "--fqbn", FQBN, "-p", port, str(SKETCH)]
    if subprocess.run(cmd).returncode != 0:
        sys.exit("Flash failed. If it can't connect: hold BOOT, tap RESET, "
                 "release BOOT, and try again.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="ESP32 serial port (default: auto-detect)")
    ap.add_argument("--no-camera", action="store_true", help="skip the Basler feed")
    ap.add_argument("--flash", action="store_true", help="upload the ESP32 firmware first")
    args = ap.parse_args()

    port = args.port or find_port()
    if args.flash:
        flash_firmware(port)  # before the reader opens the port

    reader = AdcReader(port)
    reader.start()

    root = tk.Tk()
    root.title("Tube rig")

    cam = None
    if not args.no_camera:
        from basler_feed import CameraPanel  # pypylon only needed when used
        cam = CameraPanel(root)
        cam.pack(side="left", fill="both", expand=True, padx=4, pady=4)

    ScopePanel(root, reader).pack(side="left", fill="both", expand=True, padx=4, pady=4)

    def shutdown():
        reader.stop()
        if cam:
            cam.stop()  # blocks until the camera is released
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", shutdown)
    # Ctrl-C in the terminal: close cleanly so the camera gets released
    signal.signal(signal.SIGINT, lambda *_: root.after(0, shutdown))
    root.mainloop()


if __name__ == "__main__":
    main()
