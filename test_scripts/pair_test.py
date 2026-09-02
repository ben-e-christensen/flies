"""Truthful dual-camera measurement.

grab() lies on DSHOW -- it returns True instantly whether or not a new frame
arrived. This counts only frames whose CONTENT changed, reads each camera in
its own thread (as real use would), and shows both live so you can confirm
they are two different scenes.

Run:
    python pair_test4.py

Press q to quit.
"""

import os

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import threading
import time

import cv2
import numpy as np

CAM_A, CAM_B = 0, 1
BACKEND = cv2.CAP_DSHOW
WIDTH, HEIGHT = 1280, 720
MEASURE_S = 6.0


def fourcc_str(cap):
    v = int(cap.get(cv2.CAP_PROP_FOURCC))
    s = ''.join(chr((v >> 8 * i) & 0xFF) for i in range(4)).strip()
    return s if s else '(none)'


def open_cam(index, w, h):
    cap = cv2.VideoCapture(index, BACKEND)
    if not cap.isOpened():
        cap.release()
        return None, 'open failed'

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, 30)

    deadline = time.perf_counter() + 5.0
    while time.perf_counter() < deadline:
        ret, f = cap.read()
        if ret and f is not None:
            aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap, f'{aw}x{ah} {fourcc_str(cap)}'
        time.sleep(0.05)

    cap.release()
    return None, 'no frames'


class Reader(threading.Thread):
    """Reads one camera flat out, counting only genuinely NEW frames."""

    def __init__(self, cap, name):
        super().__init__(daemon=True)
        self.cap = cap
        self.name_ = name
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest = None

        self.reads = 0          # every successful read()
        self.new_frames = 0     # reads whose content actually changed
        self._sig = None

    def stop(self):
        self._stop.set()

    def latest(self):
        with self._lock:
            return self._latest

    def run(self):
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.002)
                continue
            self.reads += 1

            # Cheap content signature: sparse sample, not a full hash.
            sig = int(frame[::37, ::37, 0].astype(np.int64).sum())
            if sig != self._sig:
                self._sig = sig
                self.new_frames += 1
                with self._lock:
                    self._latest = frame


def main():
    print(f'Opening both cameras at {WIDTH}x{HEIGHT}...')

    cap_a, info_a = open_cam(CAM_A, WIDTH, HEIGHT)
    if cap_a is None:
        print(f'cam0: {info_a}')
        return
    print(f'  cam0: {info_a}')

    time.sleep(0.5)
    cap_b, info_b = open_cam(CAM_B, WIDTH, HEIGHT)
    if cap_b is None:
        print(f'  cam1: {info_b}')
        cap_a.release()
        return
    print(f'  cam1: {info_b}')

    ra = Reader(cap_a, 'cam0')
    rb = Reader(cap_b, 'cam1')
    ra.start()
    rb.start()

    print(f'\nMeasuring for {MEASURE_S:.0f}s...')
    t0 = time.perf_counter()
    time.sleep(MEASURE_S)
    dt = time.perf_counter() - t0

    fa = ra.new_frames / dt
    fb = rb.new_frames / dt
    print(f'\n  cam0: {fa:5.1f} real fps  ({ra.reads / dt:.0f} read calls/s)')
    print(f'  cam1: {fb:5.1f} real fps  ({rb.reads / dt:.0f} read calls/s)')

    # Are the two streams actually different devices?
    A, B = ra.latest(), rb.latest()
    if A is not None and B is not None:
        d = float(np.mean(cv2.absdiff(cv2.resize(A, (160, 120)),
                                      cv2.resize(B, (160, 120)))))
        print(f'  difference between the two streams: {d:.1f}')
        if d < 8:
            print('    !! near-identical -- likely the same device twice')

    print()
    if min(fa, fb) >= 15:
        print(f'VERDICT: good. Both sustain 15fps saving.')
        print(f'    CAM_WIDTH, CAM_HEIGHT = {WIDTH}, {HEIGHT}')
    elif min(fa, fb) >= 4:
        print('VERDICT: streaming, but slow -- this is the YUY2 5fps cap.')
        print('    OpenCV is not applying MJPG. Either drop SAVE_FPS to 5,')
        print('    lower the resolution to 640x480 (YUY2 does 30 there),')
        print('    or switch to the ffmpeg capture route to force MJPG.')
    else:
        print('VERDICT: too slow to be usable.')

    print('\nLive preview -- press q to close.')
    win = 'cam0 | cam1'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    try:
        while True:
            A, B = ra.latest(), rb.latest()
            tiles = []
            for f, r in ((A, ra), (B, rb)):
                if f is None:
                    continue
                s = cv2.resize(f, (480, int(480 * f.shape[0] / f.shape[1])))
                cv2.putText(s, r.name_, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(s, r.name_, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2, cv2.LINE_AA)
                tiles.append(s)
            if len(tiles) == 2:
                cv2.imshow(win, cv2.hconcat(tiles))
            if (cv2.waitKey(30) & 0xFF) == ord('q'):
                break
    finally:
        ra.stop()
        rb.stop()
        ra.join(timeout=2)
        rb.join(timeout=2)
        cap_a.release()
        cap_b.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()