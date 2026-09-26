"""Flag continuous frames that look different from a reference frame.

Walks SEARCH_ROOT for continuous frames (p###_######.png inside flir_a_cont
folders), compares each one against REF_PATH, and flags the odd ones.

Per frame, four numbers are computed (all images scaled to 0-1 first, so
8-bit and 16-bit frames are comparable):

    mad     mean |frame - ref|                  overall difference
    frac    fraction of pixels with |diff| > PIX_THRESH   localized changes
    sharp   Laplacian variance / ref's          "pixelation" / noise / blur
    bright  mean(frame) - mean(ref)             global brightness shift

After everything is scored, each metric is compared to its own median across
ALL frames (robust z-score using MAD).  A frame is flagged if any metric is
more than Z_THRESH robust-sigmas out AND more than that metric's MIN_DELTA
from the median, so a very tight distribution can't flag trivial noise.
Unreadable or wrong-size files are always flagged.

Pass 1 (scoring) streams: RAM doesn't make it faster.  Speed is set by how
fast the drive can read and how many cores can decode PNGs in parallel.

RESUMABLE: scores are appended to scores.csv as they come in.  If you stop it
(Ctrl+C) or it crashes, just run it again and it skips what's already done.

Output (OUT_DIR):
    scores.csv          every frame: path,status,mad,frac,sharp,bright
    flagged.csv         flagged frames, sorted, with reasons and z-scores
    scores_plot.png     metrics vs frame order, flagged in red
    flagged_frames/     copies of flagged frames (first MAX_COPY) to eyeball
"""

import csv
import math
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

import cv2
import numpy as np

# ============ CONFIG ============
SEARCH_ROOT = Path(r'E:\Ben Christensen\FLIES')
REF_PATH    = Path(r'E:\Ben Christensen\FLIES\session_2026-09-21_18-08-24\flir_a_cont'
                   r'\p001_000800.png')          # <-- set the session folder
SUBDIR_NAME = 'flir_a_cont'   # only scan folders with this name (None = all)
FILE_PREFIX = 'p'             # continuous frames are p###_######.png
FILE_EXT    = '.png'

OUT_DIR     = SEARCH_ROOT / 'frame_check'
RESUME      = True            # skip frames already in scores.csv

WORKERS        = os.cpu_count() or 8   # decode threads (cv2 releases the GIL)
MAX_INFLIGHT   = WORKERS * 4           # frames queued ahead of the workers
PROGRESS_EVERY = 5000                  # console update every N frames
HEARTBEAT_S    = 60                    # also print if nothing finished in N s

# --- flagging ---
PIX_THRESH = 0.05     # |diff| > 5% of full scale counts as a changed pixel
Z_THRESH   = 6.0      # robust sigmas from the median to flag
MIN_DELTA  = {        # AND must be at least this far from the median
    'mad':    0.002,  # 0.2% of full scale average difference
    'frac':   0.001,  # 0.1% of pixels changed
    'sharp':  0.05,   # 5% change in Laplacian variance ratio
    'bright': 0.005,  # 0.5% of full scale brightness shift
}
FLAG_BRIGHTNESS = True   # False if lamp flicker makes this too noisy

# --- extras ---
COPY_FLAGGED = True
MAX_COPY     = 2000      # don't copy more than this many flagged frames
MAKE_PLOT    = True
# ================================

cv2.setNumThreads(1)     # one thread per worker; we parallelize ourselves

METRICS = ('mad', 'frac', 'sharp', 'bright')


def to_unit(img):
    """grayscale float32 in 0-1, whatever the bit depth."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint16:
        return img.astype(np.float32) * (1.0 / 65535.0)
    if img.dtype == np.uint8:
        return img.astype(np.float32) * (1.0 / 255.0)
    return img.astype(np.float32)


# --- reference (loaded once, shared read-only by all threads) ---
REF = None
REF_SHARP = 1.0
REF_MEAN = 0.0


def load_reference():
    global REF, REF_SHARP, REF_MEAN
    raw = cv2.imread(str(REF_PATH), cv2.IMREAD_UNCHANGED)
    if raw is None:
        sys.exit(f'[ref] could not read reference: {REF_PATH}')
    REF = to_unit(raw)
    REF_SHARP = float(cv2.Laplacian(REF, cv2.CV_32F).var()) or 1e-12
    REF_MEAN = float(REF.mean())
    print(f'[ref] {REF_PATH.name}  {REF.shape[1]}x{REF.shape[0]} '
          f'{raw.dtype}  mean {REF_MEAN:.4f}  lapvar {REF_SHARP:.3e}')


def score(path):
    """Returns a CSV row.  Never raises."""
    try:
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            return [path, 'unreadable', '', '', '', '']
        img = to_unit(raw)
        if img.shape != REF.shape:
            return [path, f'size_{img.shape[1]}x{img.shape[0]}',
                    '', '', '', '']
        diff = cv2.absdiff(img, REF)
        mad = float(diff.mean())
        frac = float(np.count_nonzero(diff > PIX_THRESH)) / diff.size
        sharp = float(cv2.Laplacian(img, cv2.CV_32F).var()) / REF_SHARP
        bright = float(img.mean()) - REF_MEAN
        return [path, 'ok', f'{mad:.6g}', f'{frac:.6g}',
                f'{sharp:.6g}', f'{bright:.6g}']
    except Exception as e:
        return [path, f'error:{type(e).__name__}', '', '', '', '']


# ----------------------------------------------------------- discovery

def find_frames():
    """Fast recursive scan with a running count so it never looks frozen."""
    print(f'[scan] looking for frames under {SEARCH_ROOT} ...')
    out = []
    t0 = time.perf_counter()
    last = t0
    stack = [str(SEARCH_ROOT)]
    out_dir = str(OUT_DIR)
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.is_dir(follow_symlinks=False):
                        if e.path != out_dir:
                            stack.append(e.path)
                    elif e.name.endswith(FILE_EXT) \
                            and e.name.startswith(FILE_PREFIX):
                        if SUBDIR_NAME is None \
                                or os.path.basename(d) == SUBDIR_NAME:
                            out.append(e.path)
        except OSError as err:
            print(f'[scan] skipped {d}: {err}')
        now = time.perf_counter()
        if now - last > 10:
            print(f'[scan] {len(out):,} frames found so far '
                  f'({now - t0:.0f} s)')
            last = now
    out.sort()
    print(f'[scan] {len(out):,} frames found in '
          f'{time.perf_counter() - t0:.0f} s')
    return out


# ----------------------------------------------------------- pass 1

def fmt_time(s):
    s = int(s)
    return f'{s // 3600}h{(s % 3600) // 60:02d}m{s % 60:02d}s'


def score_all(frames, scores_path):
    done = set()
    if RESUME and scores_path.exists():
        with open(scores_path, newline='') as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                if row:
                    done.add(row[0])
        print(f'[resume] {len(done):,} frames already scored, skipping')

    todo = [p for p in frames if p not in done]
    total = len(todo)
    if total == 0:
        print('[score] nothing new to score')
        return

    new_file = not scores_path.exists() or not RESUME
    f = open(scores_path, 'w' if not RESUME else 'a', newline='')
    w = csv.writer(f)
    if new_file:
        w.writerow(['path', 'status'] + list(METRICS))

    print(f'[score] {total:,} frames, {WORKERS} workers')
    t0 = time.perf_counter()
    last_done_t = t0
    n = 0
    n_bad = 0
    it = iter(todo)
    inflight = set()

    ex = ThreadPoolExecutor(max_workers=WORKERS)
    try:
        for _ in range(MAX_INFLIGHT):
            p = next(it, None)
            if p is None:
                break
            inflight.add(ex.submit(score, p))

        while inflight:
            # short timeout so Ctrl+C works on Windows and heartbeat can fire
            finished, inflight = wait(inflight, timeout=1.0,
                                      return_when=FIRST_COMPLETED)
            now = time.perf_counter()

            if not finished:
                if now - last_done_t > HEARTBEAT_S:
                    print(f'[score] still working... {n:,}/{total:,} done, '
                          f'nothing finished in {now - last_done_t:.0f} s '
                          f'(slow drive?)')
                    last_done_t = now
                continue

            for fut in finished:
                row = fut.result()
                w.writerow(row)
                n += 1
                if row[1] != 'ok':
                    n_bad += 1
                last_done_t = now

                if n % PROGRESS_EVERY == 0 or n == total:
                    f.flush()
                    el = now - t0
                    rate = n / el if el > 0 else 0
                    eta = (total - n) / rate if rate > 0 else 0
                    rel = os.path.relpath(row[0], SEARCH_ROOT)
                    print(f'[score] {n:,}/{total:,} ({100 * n / total:.1f}%)'
                          f'  {rate:.0f} img/s  elapsed {fmt_time(el)}'
                          f'  ETA {fmt_time(eta)}  bad {n_bad}'
                          f'\n        on {rel}')

                p = next(it, None)
                if p is not None:
                    inflight.add(ex.submit(score, p))

    except KeyboardInterrupt:
        print('\n[score] Ctrl+C - saving progress, rerun to resume')
        ex.shutdown(wait=False, cancel_futures=True)
        f.flush()
        f.close()
        sys.exit(1)

    ex.shutdown(wait=True)
    f.flush()
    f.close()
    print(f'[score] done: {n:,} frames in {fmt_time(time.perf_counter() - t0)}')


# ----------------------------------------------------------- pass 2

def robust(x):
    med = float(np.median(x))
    s = 1.4826 * float(np.median(np.abs(x - med)))
    return med, (s if s > 0 else 1e-12)


def flag_all(scores_path, flagged_path):
    paths, status, vals = [], [], {m: [] for m in METRICS}
    with open(scores_path, newline='') as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if not row:
                continue
            paths.append(row[0])
            status.append(row[1])
            for m, v in zip(METRICS, row[2:6]):
                vals[m].append(float(v) if v else math.nan)

    order = np.argsort(np.array(paths))          # chronological per session
    paths = [paths[i] for i in order]
    status = [status[i] for i in order]
    arr = {m: np.asarray(vals[m], dtype=np.float64)[order] for m in METRICS}
    ok = np.array([s == 'ok' for s in status])
    n = len(paths)
    print(f'[flag] {n:,} scored frames, {int(ok.sum()):,} ok')
    if not ok.any():
        print('[flag] no valid frames to analyse')
        return paths, arr, np.zeros(n, bool)

    z = {}
    print('[flag] metric      median        robust sigma')
    for m in METRICS:
        med, s = robust(arr[m][ok])
        z[m] = (arr[m] - med) / s
        z[m][~ok] = 0.0
        arr[m + '_med'] = med
        print(f'       {m:<7} {med:>12.6g}  {s:>12.4g}')

    def dev(m):
        return arr[m] - arr[m + '_med']

    reasons = [[] for _ in range(n)]
    tests = {
        'mad':    (z['mad'] > Z_THRESH) & (dev('mad') > MIN_DELTA['mad']),
        'frac':   (z['frac'] > Z_THRESH) & (dev('frac') > MIN_DELTA['frac']),
        'sharp':  (np.abs(z['sharp']) > Z_THRESH)
                  & (np.abs(dev('sharp')) > MIN_DELTA['sharp']),
    }
    if FLAG_BRIGHTNESS:
        tests['bright'] = ((np.abs(z['bright']) > Z_THRESH)
                           & (np.abs(dev('bright')) > MIN_DELTA['bright']))

    for m, hit in tests.items():
        hit = hit & ok
        print(f'[flag] {m:<7} {int(hit.sum()):,} frames')
        for i in np.nonzero(hit)[0]:
            reasons[i].append(m)
    for i in np.nonzero(~ok)[0]:
        reasons[i].append(status[i])

    flagged = np.array([bool(r) for r in reasons])
    with open(flagged_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['path', 'reasons'] + list(METRICS)
                   + [f'z_{m}' for m in METRICS])
        for i in np.nonzero(flagged)[0]:
            w.writerow([paths[i], '+'.join(reasons[i])]
                       + [f'{arr[m][i]:.6g}' for m in METRICS]
                       + [f'{z[m][i]:.2f}' for m in METRICS])

    print(f'[flag] {int(flagged.sum()):,} of {n:,} frames flagged '
          f'({100 * flagged.sum() / max(n, 1):.3f}%) -> {flagged_path}')
    return paths, arr, flagged


def copy_flagged(paths, flagged, dest):
    idx = np.nonzero(flagged)[0]
    if len(idx) == 0:
        return
    if len(idx) > MAX_COPY:
        print(f'[copy] {len(idx):,} flagged, copying only the first '
              f'{MAX_COPY:,}')
        idx = idx[:MAX_COPY]
    dest.mkdir(parents=True, exist_ok=True)
    for k, i in enumerate(idx):
        p = Path(paths[i])
        # session name in front so p001_... from different sessions don't clash
        name = f'{p.parent.parent.name}__{p.name}'
        try:
            shutil.copy2(p, dest / name)
        except OSError as e:
            print(f'[copy] {p}: {e}')
        if (k + 1) % 500 == 0:
            print(f'[copy] {k + 1:,}/{len(idx):,}')
    print(f'[copy] {len(idx):,} frames -> {dest}')


def plot(arr, flagged, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    x = np.arange(len(flagged))
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(14, 10), sharex=True)
    for ax, m in zip(axes, METRICS):
        ax.plot(x, arr[m], ',', color='0.4')
        ax.plot(x[flagged], arr[m][flagged], '.', color='red', markersize=3)
        ax.axhline(arr[m + '_med'], color='C0', lw=0.8)
        ax.set_ylabel(m)
    axes[-1].set_xlabel('frame (sorted by path)')
    axes[0].set_title(f'vs {REF_PATH.name}   red = flagged '
                      f'({int(flagged.sum()):,})')
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f'[plot] {out_png}')


# ----------------------------------------------------------- main

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores_path = OUT_DIR / 'scores.csv'
    flagged_path = OUT_DIR / 'flagged.csv'

    load_reference()
    frames = find_frames()

    # only compare frames that come AFTER the reference
    ref_key = os.path.normcase(os.path.abspath(REF_PATH))
    n_before = len(frames)
    frames = [p for p in frames
              if os.path.normcase(os.path.abspath(p)) > ref_key]
    print(f'[scan] skipping {n_before - len(frames):,} frames at or before '
          f'{REF_PATH.name}; {len(frames):,} left to compare')

    if not frames:
        sys.exit('[scan] no frames found - check SEARCH_ROOT / SUBDIR_NAME')

    score_all(frames, scores_path)
    paths, arr, flagged = flag_all(scores_path, flagged_path)

    if MAKE_PLOT and len(paths):
        plot(arr, flagged, OUT_DIR / 'scores_plot.png')
    if COPY_FLAGGED:
        copy_flagged(paths, flagged, OUT_DIR / 'flagged_frames')

    print('[done]')


if __name__ == '__main__':
    main()