#!/usr/bin/env python3
"""Background reader for the esp32_ads firmware (3x ADS1115 over serial).

The ESP prints `t_us,raw0,raw1,raw2` lines (raw ADS1115 counts) at a fixed rate;
'#' lines are info. Counts are converted to electrometer volts here, see
convert_to_voltage() and tube/claude.md.
AdcReader keeps the last `maxlen` samples in ring buffers and reconnects on
its own if the board is unplugged or resets.

Run this file directly to print incoming samples (quick wiring check):
    python3 adc_serial.py            # stream converted samples
    python3 adc_serial.py --pins     # print A0-A3 pin voltages on every ADC
"""

import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import serial
from serial.tools import list_ports

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coms import port as coms_port

BAUD = 921600

# Must match the PGA setting in esp32_ads.ino (FSR_V there).
ADS_FSR_V = 4.096
ADS_LSB = ADS_FSR_V / 32768.0   # 125 uV per count


def convert_to_voltage(raw):
    """ADS1115 @ +/-4.096V PGA through INA159 (gain 0.2, 1.25 V ref): V = 5*(Vpin - 1.25)"""
    return 5.0 * (raw * ADS_LSB - 1.25)
ESP_USB_IDS = {
    (0x239A, 0x811B),  # Adafruit Feather ESP32-S3 2MB PSRAM
    (0x239A, 0x8113),  # Adafruit Feather ESP32-S3 No PSRAM
}
ESPRESSIF_VID = 0x303A  # any other ESP32-S3 on native USB


def find_port():
    """USB port of the ESP32-S3 if one is plugged in, else coms.py 'esp32'."""
    for p in list_ports.comports():
        if (p.vid, p.pid) in ESP_USB_IDS or p.vid == ESPRESSIF_VID:
            return p.device
    return coms_port('esp32')


class AdcReader(threading.Thread):
    def __init__(self, port=None, baud=BAUD, maxlen=20000):
        super().__init__(daemon=True)
        self.port = port or find_port()
        self.baud = baud
        self.lock = threading.Lock()
        self.t = deque(maxlen=maxlen)  # seconds, ESP clock
        self.v = [deque(maxlen=maxlen) for _ in range(3)]  # electrometer volts
        self.info = []                 # '#' lines from the most recent connect
        self.connected = False
        self.err = None
        self.n_samples = 0
        self.n_bad = 0
        self.stop_event = threading.Event()
        # Optional hook for logging later: called from this thread as
        # on_sample(t_s, v0, v1, v2, raws) with volts and the raw counts.
        self.on_sample = None
        self.on_info = None  # on_info(line) for each '#' line
        self.on_connect = None  # on_connect(ser) right after the port opens

        self._t_wrap = 0.0  # micros() wraps every ~71.6 min
        self._last_us = None

    def run(self):
        while not self.stop_event.is_set():
            try:
                # exclusive: a second program opening the port would steal
                # bytes from this one and both would see garbage
                with serial.Serial(self.port, self.baud, timeout=1, exclusive=True) as ser:
                    self.connected, self.err, self.info = True, None, []
                    ser.reset_input_buffer()
                    ser.write(b'?')  # ask for the config header
                    if self.on_connect:
                        self.on_connect(ser)
                    self._read_loop(ser)
            except (serial.SerialException, OSError) as e:
                self.err = f"{self.port}: {e}"
            self.connected = False
            if not self.stop_event.is_set():
                time.sleep(1.0)

    def _read_loop(self, ser):
        while not self.stop_event.is_set():
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode('ascii', errors='replace').strip()
            if not line:
                continue
            if line.startswith('#'):
                self.info.append(line[1:].strip())
                if self.on_info:
                    self.on_info(line)
                continue
            try:
                t_us, a, b, c = line.split(',')
                t_us = int(t_us)
                raws = (float(a), float(b), float(c))  # float() so "nan" parses
            except ValueError:
                self.n_bad += 1  # partial line after connect, noise, etc.
                continue

            if self._last_us is not None and t_us < self._last_us:
                if self._last_us - t_us > 2**31:
                    self._t_wrap += 2**32 / 1e6
                else:
                    # ESP reset: start a fresh trace rather than joining two clocks
                    self._t_wrap = 0.0
                    self.clear()
            self._last_us = t_us
            t_s = self._t_wrap + t_us / 1e6
            vals = tuple(convert_to_voltage(r) for r in raws)

            with self.lock:
                self.t.append(t_s)
                for dq, x in zip(self.v, vals):
                    dq.append(x)
            self.n_samples += 1
            if self.on_sample:
                self.on_sample(t_s, *vals, raws)

    def clear(self):
        with self.lock:
            self.t.clear()
            for dq in self.v:
                dq.clear()

    def snapshot(self):
        """Return (t, [v0, v1, v2]) as numpy arrays."""
        with self.lock:
            t = np.fromiter(self.t, float, len(self.t))
            v = [np.fromiter(dq, float, len(dq)) for dq in self.v]
        return t, v

    def stop(self):
        self.stop_event.set()


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('port', nargs='?', help='serial port (default: auto-detect)')
    ap.add_argument('--pins', action='store_true',
                    help='print A0-A3 pin voltages on every ADC, then exit')
    args = ap.parse_args()

    r = AdcReader(args.port)
    r.on_info = print
    if args.pins:
        r.on_connect = lambda ser: ser.write(b'a')
    else:
        r.on_sample = lambda t, a, b, c, raws: print(
            f"{t:10.4f}  {a:+8.4f} {b:+8.4f} {c:+8.4f} V   raw {raws}")
    print("Port:", r.port)
    r.start()
    try:
        if args.pins:
            time.sleep(2)
        else:
            while True:
                time.sleep(1)
                if r.err:
                    print("[!]", r.err)
    except KeyboardInterrupt:
        pass
    r.stop()
    if r.err:
        print("[!]", r.err)
