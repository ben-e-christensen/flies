import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

PREFACTOR = 1e12   # C -> pC
DATA_FOLDER = Path(__file__).resolve().parent / 'Kiethley_data'
CAMERAS = ['cam0', 'cam1']

# Time window (seconds) to inspect for spikes / build a movie of.
BLOCK_START = 2750
BLOCK_END = 3800

# Spike detection: a point is a "spike" if the rate of change |dq/dt| exceeds
# median(|rate|) + SPIKE_SIGMA * std(rate). Nearby spike points within
# SPIKE_MERGE_WINDOW seconds of each other are merged into one spike event.
SPIKE_SIGMA = 10
SPIKE_MERGE_WINDOW = 1.0  # seconds

# For each spike event, grab frames within +/- FRAME_WINDOW/2 seconds of it.
FRAME_WINDOW = 1.0  # seconds

MOVIE_FPS = 5  # matches save_fps in meta.txt


def pick_session_dir():
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        return p.parent if p.is_file() else p

    sessions = sorted(
        (d for d in DATA_FOLDER.glob('session_*') if d.is_dir()),
        key=lambda p: p.stat().st_mtime,
    )
    if not sessions:
        raise SystemExit(f'No session folders found in {DATA_FOLDER} and none given as an argument.')
    return sessions[-1]


def load_electrometer(session_dir):
    filepath = session_dir / 'electrometer.csv'
    if not filepath.exists():
        raise SystemExit(f'File not found: {filepath}')

    data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    t = data[:, 0]
    q = data[:, 1] * PREFACTOR
    valid = ~np.isnan(q)
    return t[valid], q[valid]


def load_frames(session_dir, camera):
    filepath = session_dir / f'{camera}_frames.csv'
    if not filepath.exists():
        raise SystemExit(f'File not found: {filepath}')

    frame_times = []
    frame_names = []
    with open(filepath) as f:
        next(f)  # header
        for line in f:
            time_str, name = line.strip().split(',', 1)
            frame_times.append(float(time_str))
            frame_names.append(name)
    return np.array(frame_times), frame_names


def find_spikes(t, q):
    dt = np.diff(t)
    dq = np.diff(q)
    rate = dq / dt
    thresh = np.median(np.abs(rate)) + SPIKE_SIGMA * np.std(rate)

    idx = np.where(np.abs(rate) > thresh)[0]
    spike_times = t[idx]

    events = []
    for st in spike_times:
        if events and st - events[-1] <= SPIKE_MERGE_WINDOW:
            continue
        events.append(st)
    return events


def save_spike_frames(session_dir, camera, frame_times, frame_names, spike_events):
    out_root = session_dir / 'spike_frames'
    for spike_t in spike_events:
        mask = np.abs(frame_times - spike_t) <= FRAME_WINDOW / 2
        if not np.any(mask):
            continue

        out_dir = out_root / f'spike_{spike_t:.2f}s' / camera
        out_dir.mkdir(parents=True, exist_ok=True)

        for i in np.where(mask)[0]:
            src = session_dir / camera / frame_names[i]
            if src.exists():
                shutil.copy2(src, out_dir / src.name)


def build_block_movie(session_dir, camera, frame_times, frame_names):
    mask = (frame_times >= BLOCK_START) & (frame_times <= BLOCK_END)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        print(f'{camera}: no frames in {BLOCK_START}-{BLOCK_END}s, skipping movie')
        return

    idx = idx[np.argsort(frame_times[idx])]
    first_frame = cv2.imread(str(session_dir / camera / frame_names[idx[0]]))
    if first_frame is None:
        print(f'{camera}: could not read first frame, skipping movie')
        return
    height, width = first_frame.shape[:2]

    out_path = session_dir / f'{camera}_block_{BLOCK_START:g}-{BLOCK_END:g}s.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, MOVIE_FPS, (width, height))

    for i in idx:
        frame = cv2.imread(str(session_dir / camera / frame_names[i]))
        if frame is not None:
            writer.write(frame)
    writer.release()
    print(f'{camera}: saved movie ({len(idx)} frames) to {out_path}')


def main():
    session_dir = pick_session_dir()
    print(f'Session: {session_dir}')

    t, q = load_electrometer(session_dir)
    block_mask = (t >= BLOCK_START) & (t <= BLOCK_END)
    spike_events = find_spikes(t[block_mask], q[block_mask])
    print(f'Found {len(spike_events)} spike event(s) in {BLOCK_START}-{BLOCK_END}s: '
          f'{[f"{s:.2f}" for s in spike_events]}')

    for camera in CAMERAS:
        frame_times, frame_names = load_frames(session_dir, camera)
        save_spike_frames(session_dir, camera, frame_times, frame_names, spike_events)
        build_block_movie(session_dir, camera, frame_times, frame_names)


if __name__ == '__main__':
    main()
