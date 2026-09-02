"""Dual Brio capture via ffmpeg -- real MJPG, native JPEGs, no re-encode.

OpenCV's DSHOW backend will not negotiate MJPG on this camera, which pins it
to the YUY2 5fps ceiling at 720p. ffmpeg forces mjpeg properly AND can tell
two identically-named devices apart with -video_device_number.

Frames arrive already JPEG-compressed from the camera and are written to disk
verbatim -- no decode, no re-encode, no quality loss, almost no CPU.

Run:
    python ffmpeg_cams.py            # preview, SPACE to record
    python ffmpeg_cams.py list       # list dshow device names

Keys:  SPACE = start/stop recording,  q = quit
"""

import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ============ CONFIG ============
DEVICE_NAME = 'Brio 101'      # both cameras share this name
DEVICE_NUMBERS = [0, 1]       # -video_device_number, disambiguates them
WIDTH, HEIGHT = 1920, 1080
FPS = 15                      # camera does mjpeg 5-30 at this size
FFMPEG = r'C:\Users\Ben\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.1-full_build\bin\ffmpeg.exe'
OUT_ROOT = Path(__file__).resolve().parent / 'cam_data'
PREVIEW_W = 480
PREVIEW_EVERY = 2             # decode every Nth frame for preview only
# ================================

SOI = b'\xff\xd8'             # JPEG start of image
EOI = b'\xff\xd9'             # JPEG end of image


def list_devices():
    p = subprocess.run(
        [FFMPEG, '-hide_banner', '-list_devices', 'true', '-f', 'dshow',
         '-i', 'dummy'],
        capture_output=True, text=True
    )
    print(p.stderr)


class FFmpegCamera(threading.Thread):
    """One ffmpeg subprocess piping an MJPEG stream; splits it into frames."""

    def __init__(self, device_number, name):
        super().__init__(daemon=True)
        self.device_number = device_number
        self.name_ = name

        self.proc = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest_jpeg = None

        self.save_q = queue.Queue(maxsize=200)
        self.recording = False
        self.t0 = 0.0
        self.n = 0
        self.saved = 0
        self.dropped = 0
        self.received = 0
        self.fps = 0.0
        self._fps_t = time.perf_counter()
        self._fps_n = 0
        self.error = None

    def cmd(self):
        return [
            FFMPEG, '-hide_banner', '-loglevel', 'error',
            '-f', 'dshow',
            '-video_device_number', str(self.device_number),
            '-vcodec', 'mjpeg',                    # force MJPG, not YUY2
            '-video_size', f'{WIDTH}x{HEIGHT}',
            '-framerate', str(FPS),
            '-rtbufsize', '128M',
            '-i', f'video={DEVICE_NAME}',
            '-c:v', 'copy',                        # passthrough, no re-encode
            '-f', 'mjpeg', 'pipe:1',
        ]

    def stop(self):
        self._stop.set()
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass

    def latest_jpeg(self):
        with self._lock:
            return self._latest_jpeg

    def _drain_stderr(self):
        for line in self.proc.stderr:
            txt = line.decode('utf-8', 'replace').rstrip()
            if txt:
                print(f'[{self.name_}] {txt}')
                if self.error is None:
                    self.error = txt

    def run(self):
        try:
            self.proc = subprocess.Popen(
                self.cmd(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                bufsize=0
            )
        except FileNotFoundError:
            self.error = f'{FFMPEG} not found on PATH'
            print(f'[{self.name_}] {self.error}')
            return

        threading.Thread(target=self._drain_stderr, daemon=True).start()

        buf = bytearray()
        try:
            while not self._stop.is_set():
                chunk = self.proc.stdout.read(65536)
                if not chunk:
                    break
                buf.extend(chunk)

                # Pull out every complete JPEG sitting in the buffer.
                while True:
                    start = buf.find(SOI)
                    if start < 0:
                        buf.clear()
                        break
                    end = buf.find(EOI, start + 2)
                    if end < 0:
                        del buf[:start]        # keep the partial frame
                        break

                    jpeg = bytes(buf[start:end + 2])
                    del buf[:end + 2]
                    self._on_frame(jpeg)
        except Exception as e:
            self.error = str(e)
            print(f'[{self.name_}] {e}')
        finally:
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

        if self.recording:
            try:
                self.save_q.put_nowait((now - self.t0, jpeg, self.n))
                self.n += 1
                self.saved += 1
            except queue.Full:
                self.dropped += 1


class Writer(threading.Thread):
    """Writes the camera's own JPEG bytes straight to disk."""

    def __init__(self, cam, out_dir, index_csv):
        super().__init__(daemon=True)
        self.cam = cam
        self.out_dir = out_dir
        self.index_csv = index_csv
        self.written = 0

    def run(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.index_csv, 'w') as idx:
            idx.write('time,filename\n')
            while True:
                item = self.cam.save_q.get()
                if item is None:
                    break
                t, jpeg, n = item
                fname = f'{n:06d}.jpg'
                with open(self.out_dir / fname, 'wb') as f:
                    f.write(jpeg)
                idx.write(f'{t:.6f},{fname}\n')
                self.written += 1
                if self.written % FPS == 0:
                    idx.flush()
            idx.flush()


def main():
    cams = [FFmpegCamera(n, f'cam{n}') for n in DEVICE_NUMBERS]
    for c in cams:
        c.start()
        time.sleep(0.5)

    print(f'Waiting for {WIDTH}x{HEIGHT} mjpeg @{FPS} on both cameras...')
    deadline = time.perf_counter() + 15
    while time.perf_counter() < deadline:
        if all(c.received > 0 for c in cams):
            break
        if any(c.error and 'not found' in str(c.error) for c in cams):
            break
        time.sleep(0.2)

    for c in cams:
        state = f'{c.received} frames' if c.received else 'NO FRAMES'
        print(f'  {c.name_} (device {c.device_number}): {state}')

    if not any(c.received for c in cams):
        print('\nNothing arrived. Try:  python ffmpeg_cams.py list')
        print('and check DEVICE_NAME matches exactly.')
        for c in cams:
            c.stop()
        return

    print('\nSPACE = record, q = quit')
    win = 'ffmpeg dual cam'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    writers = []
    recording = False
    session = None
    tick = 0

    try:
        while True:
            tick += 1
            tiles = []
            for c in cams:
                jpeg = c.latest_jpeg()
                if jpeg is None:
                    continue
                if tick % PREVIEW_EVERY:
                    continue
                frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8),
                                     cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                h, w = frame.shape[:2]
                s = cv2.resize(frame, (PREVIEW_W, int(h * PREVIEW_W / w)),
                               interpolation=cv2.INTER_AREA)
                txt = f'{c.name_} {c.fps:.1f}fps saved={c.saved}'
                if c.dropped:
                    txt += f' DROP={c.dropped}'
                for col, th in (((0, 0, 0), 4),
                                ((0, 255, 0) if recording else (255, 255, 255), 1)):
                    cv2.putText(s, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, col, th, cv2.LINE_AA)
                tiles.append(s)

            if len(tiles) == len(cams):
                cv2.imshow(win, cv2.hconcat(tiles))

            key = cv2.waitKey(15) & 0xFF
            if key == ord('q'):
                break
            if key == ord(' '):
                if not recording:
                    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                    session = OUT_ROOT / f'session_{stamp}'
                    session.mkdir(parents=True, exist_ok=True)
                    t0 = time.perf_counter()
                    writers = []
                    for c in cams:
                        c.n = c.saved = c.dropped = 0
                        c.t0 = t0
                        w = Writer(c, session / c.name_,
                                   session / f'{c.name_}_frames.csv')
                        w.start()
                        writers.append(w)
                    for c in cams:
                        c.recording = True
                    recording = True
                    print(f'recording -> {session.name}')
                else:
                    for c in cams:
                        c.recording = False
                    time.sleep(0.2)
                    for c in cams:
                        c.save_q.put(None)
                    for w in writers:
                        w.join(timeout=20)
                    recording = False
                    for c in cams:
                        print(f'  {c.name_}: {c.saved} saved, {c.dropped} dropped')
                    print(f'stopped -> {session}')
    finally:
        for c in cams:
            c.recording = False
        time.sleep(0.2)
        for c in cams:
            try:
                c.save_q.put_nowait(None)
            except queue.Full:
                pass
        for w in writers:
            w.join(timeout=20)
        for c in cams:
            c.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'list':
        list_devices()
    else:
        main()