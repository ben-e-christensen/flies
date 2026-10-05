#!/usr/bin/env python3
"""Background reader for the xiao_max1032 firmware (MAX1032 on a XIAO ESP32-C6).

On connect it sends x (stop), then either r<hz> (set the rate) or ? (keep the
board's rate); both make the board print its header. The "# cfg ..." line gives
the rate, packet layout and scaling. Then s (start), and the board streams
packets of samples (see the .ino). Samples go into a ring buffer, and on_block
gets each decoded packet group as it arrives (for logging).

Counts -> volts: V = (code - zero) * lsb_v, both from the board's header
(zero=8192; lsb_v = 375 uV / 750 uV / 1.5 mV for the +/-3.072 / 6.144 / 12.288 V
ranges, set with range_v). The MAX1032 takes the
electrometer output directly, so there's no pedestal or gain to undo.

Run this file directly for a once-a-second summary per channel (mean, rms noise,
gaps, railed codes). Useful for the scale check and the shorted-input noise floor:
    python3 adc_serial.py
    python3 adc_serial.py --rate 10000 --range 6
"""

import sys
import threading
import time
from pathlib import Path

import numpy as np
import serial
from serial.tools import list_ports

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coms import port as coms_port

BAUD = 921600           # ignored on native USB
ESPRESSIF_VID = 0x303A  # XIAO ESP32-C6 USB-Serial/JTAG
SYNC = b"\xA5\x5A"
HDR = 10                # sync + seq + t0_us
HEADER_TIMEOUT_S = 4.0
RANGE_CODES = {3.072: 1, 6.144: 4, 12.288: 7}  # +/- volts -> MAX1032 R[2:0]


def range_code(range_v):
    """Accept 3/6/12 or the exact 3.072/6.144/12.288 (+/- volts)."""
    for v, code in RANGE_CODES.items():
        if abs(range_v - v) < 0.5:
            return code
    raise ValueError(f"range must be one of +/-{', '.join(f'{v:g}' for v in RANGE_CODES)} V")


def find_port():
    """USB port of the XIAO if one is plugged in, else coms.py 'esp32'."""
    for p in list_ports.comports():
        if p.vid == ESPRESSIF_VID:
            return p.device
    return coms_port('esp32')


def parse_cfg(lines):
    """Pull key=value pairs out of the '# cfg ...' header line."""
    for line in lines:
        if line.startswith("# cfg"):
            out = {}
            for kv in line[5:].split():
                k, _, v = kv.partition("=")
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
            return out
    return None


def decode(buf, n_ch, n_samp):
    """Decode whole packets from the start of buf.

    Returns ([(seq, t0_us, dt_us (n_samp,), codes (n_samp, n_ch) uint16), ...],
    bytes consumed, bad packets skipped). Resyncs on the A5 5A marker after any
    corruption.
    """
    size = HDR + n_samp * (1 + n_ch) * 2 + 1
    out, used, bad = [], 0, 0
    while True:
        i = buf.find(SYNC, used)
        if i < 0:
            used = max(used, len(buf) - 1)  # keep a possible half sync byte
            break
        if i > used:
            bad += 1
        used = i
        if len(buf) - used < size:
            break
        p = np.frombuffer(bytes(buf[used:used + size]), np.uint8)
        if np.bitwise_xor.reduce(p[2:-1]) != p[-1]:
            used += 1  # not a real packet start (or corrupt): look further
            bad += 1
            continue
        seq, t0 = p[2:10].view("<u4")
        body = p[HDR:-1].view("<u2").reshape(n_samp, 1 + n_ch)
        out.append((int(seq), int(t0), body[:, 0], body[:, 1:]))
        used += size
    return out, used, bad


class AdcReader(threading.Thread):
    def __init__(self, port=None, rate=None, range_v=None, baud=BAUD, buffer_s=30.0):
        """rate: per-channel samples/s to request (None = board's current rate).
        range_v: input range +/- volts, 3, 6 or 12 (None = board's current range)."""
        super().__init__(daemon=True)
        self.port = port or find_port()
        self.rate = rate
        self._range_code = range_code(range_v) if range_v else None
        self.baud = baud
        self.buffer_s = buffer_s
        self.lock = threading.Lock()
        self.cfg = None          # dict from the board's "# cfg" line
        self.n_ch = 4
        self.sample_hz = 0.0
        self.info = []           # header lines from the most recent connect
        self.connected = False
        self.err = None
        self.n_samples = 0
        self.n_gaps = 0          # times packets went missing (host or USB too slow)
        self.n_missing = 0       # samples lost in those gaps
        self.n_bad = 0           # corrupt packets / stray bytes skipped
        self.stop_event = threading.Event()
        # Optional hook for logging: on_block(host_t, esp_t, raw, volts), called
        # from this thread. host_t = perf_counter() when the data arrived,
        # esp_t (n,) seconds on the board clock, raw (n, n_ch) codes, volts (n, n_ch).
        self.on_block = None

        self._cap = 0
        self._t = self._v = None
        self._w = 0              # samples written so far (ring index = _w % _cap)
        self._next_seq = None
        self._last_t_us = None   # raw u32 of the last packet, for unwrapping
        self._wrap_us = 0

    # ---- connection ----
    def run(self):
        while not self.stop_event.is_set():
            try:
                # exclusive: a second program opening the port would steal
                # bytes from this one and both would see garbage
                with serial.Serial(self.port, self.baud, timeout=0.01, exclusive=True) as ser:
                    self._handshake(ser)
                    self.connected, self.err = True, None
                    self._stream(ser)
            except (serial.SerialException, OSError, RuntimeError) as e:
                self.err = f"{self.port}: {e}"
            self.connected = False
            if not self.stop_event.is_set():
                time.sleep(1.0)

    def _handshake(self, ser):
        ser.write(b"x")
        if self._range_code:
            ser.write(f"g{self._range_code}\n".encode())
        if self.rate:
            ser.write(f"r{int(self.rate)}\n".encode())
        time.sleep(0.1)
        ser.reset_input_buffer()  # drop any stream bytes and the headers those printed
        ser.write(b"?")
        lines, text = [], b""
        deadline = time.time() + HEADER_TIMEOUT_S
        while time.time() < deadline and "# end" not in lines:
            text += ser.read(256)
            *done, text = text.split(b"\n")
            lines += [l.decode("ascii", "replace").strip() for l in done]
            lines = [l for l in lines if l.startswith("#")]
        cfg = parse_cfg(lines)
        if cfg is None:
            raise RuntimeError("no header from the board (is xiao_max1032 flashed?)")
        self.info, self.cfg = lines, cfg
        self.n_ch = int(cfg["n_ch"])
        self.sample_hz = cfg["sample_hz"]
        self._n_samp = int(cfg["packet_samples"])
        self._lsb, self._zero = cfg["lsb_v"], cfg["zero"]
        self._reset_buffers()
        ser.write(b"s")

    def _reset_buffers(self):
        with self.lock:
            self._cap = int(self.buffer_s * self.sample_hz)
            self._t = np.zeros(self._cap)
            self._v = np.zeros((self._cap, self.n_ch))
            self._w = 0
        self._next_seq, self._last_t_us, self._wrap_us = None, None, 0

    # ---- streaming ----
    def _stream(self, ser):
        buf = bytearray()
        while not self.stop_event.is_set():
            data = ser.read(max(ser.in_waiting, 1024))
            host_t = time.perf_counter()
            if not data:
                continue
            buf += data
            packets, used, bad = decode(buf, self.n_ch, self._n_samp)
            del buf[:used]
            self.n_bad += bad
            if packets:
                self._ingest(host_t, packets)

    def _ingest(self, host_t, packets):
        """Packets -> sample times (board clock, s) and codes; track gaps."""
        ts, cs = [], []
        for seq, t_us, dt_us, codes in packets:
            if self._next_seq is not None:
                if seq < self._next_seq - self._n_samp:
                    # board restarted streaming: start a fresh trace
                    self.clear()
                    self._next_seq, self._last_t_us, self._wrap_us = None, None, 0
                elif seq > self._next_seq:
                    self.n_gaps += 1
                    self.n_missing += seq - self._next_seq
            if self._last_t_us is not None and t_us < self._last_t_us - 2**31:
                self._wrap_us += 2**32  # micros() wrapped (every ~71.6 min)
            self._last_t_us = t_us
            self._next_seq = seq + self._n_samp
            # each sample's real time = packet start + its own offset
            ts.append((self._wrap_us + t_us + dt_us.astype(np.int64)) / 1e6)
            cs.append(codes)
        esp_t = np.concatenate(ts)
        codes = np.vstack(cs)
        volts = (codes.astype(np.float64) - self._zero) * self._lsb
        with self.lock:
            n = len(esp_t)
            idx = (self._w + np.arange(n)) % self._cap
            self._t[idx] = esp_t
            self._v[idx] = volts
            self._w += n
        self.n_samples += n
        if self.on_block:
            self.on_block(host_t, esp_t, codes, volts)

    # ---- access ----
    def clear(self):
        with self.lock:
            self._w = 0

    def snapshot(self, last_s=None):
        """Return (t, [v_ch0, v_ch1, ...]) in time order, optionally only the
        last `last_s` seconds."""
        with self.lock:
            if self._t is None or self._w == 0:
                return np.empty(0), [np.empty(0) for _ in range(self.n_ch)]
            n = min(self._w, self._cap)
            if last_s is not None:
                n = min(n, int(last_s * self.sample_hz) + 1)
            idx = (self._w - n + np.arange(n)) % self._cap
            t, v = self._t[idx], self._v[idx]
        return t, [v[:, i] for i in range(v.shape[1])]

    def stop(self):
        self.stop_event.set()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("port", nargs="?", help="serial port (default: auto-detect)")
    ap.add_argument("--rate", type=float, help="samples/s per channel (default: board's)")
    ap.add_argument("--range", type=float, dest="range_v",
                    help="input range +/- volts: 3, 6 or 12 (default: board's, 12 at boot)")
    args = ap.parse_args()

    stats = []
    r = AdcReader(args.port, rate=args.rate, range_v=args.range_v)
    r.on_block = lambda h, t, raw, v: stats.append((raw, v))
    print("Port:", r.port)
    r.start()
    last_missing, shown_header = 0, False
    try:
        while True:
            time.sleep(1)
            if not r.connected:
                print("[!]", r.err or "connecting…")
                shown_header = False
                continue
            if not shown_header:
                print("\n".join(r.info))
                shown_header = True
            blocks, stats[:] = stats[:], []
            if not blocks:
                print("no data")
                continue
            raw = np.vstack([b[0] for b in blocks])
            v = np.vstack([b[1] for b in blocks])
            print(f"{len(v)} S/s per channel, lost {r.n_missing - last_missing} samples, "
                  f"bad packets {r.n_bad}")
            last_missing = r.n_missing
            for ch in range(v.shape[1]):
                railed = np.any((raw[:, ch] == 0) | (raw[:, ch] == 0x3FFF))
                print(f"  CH{ch}: {v[:, ch].mean():+9.5f} V   noise {v[:, ch].std() * 1e3:6.3f} mV rms"
                      f"   mean {raw[:, ch].mean() - 8192:+8.1f} counts"
                      f"{'  <-- RAILED' if railed else ''}")
    except KeyboardInterrupt:
        pass
    r.stop()
