import sys
from pathlib import Path

import cv2
import numpy as np

DATA_FOLDER = Path(__file__).resolve().parent / 'Kiethley_data'
CAMERAS = ['cam0', 'cam1']

# Time range (seconds, absolute session time) to clip.
CLIP_START = 2770
CLIP_END = 2810

CLIP_FPS = 5  # matches save_fps in meta.txt


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


def build_clip(session_dir, camera, frame_times, frame_names, start, end):
    mask = (frame_times >= start) & (frame_times <= end)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        print(f'{camera}: no frames in {start:g}-{end:g}s, skipping clip')
        return

    idx = idx[np.argsort(frame_times[idx])]
    first_frame = cv2.imread(str(session_dir / camera / frame_names[idx[0]]))
    if first_frame is None:
        print(f'{camera}: could not read first frame, skipping clip')
        return
    height, width = first_frame.shape[:2]

    out_path = session_dir / f'{camera}_clip_{start:g}-{end:g}s.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, CLIP_FPS, (width, height))

    for i in idx:
        frame = cv2.imread(str(session_dir / camera / frame_names[i]))
        if frame is not None:
            writer.write(frame)
    writer.release()
    print(f'{camera}: saved clip ({len(idx)} frames) to {out_path}')


def build_side_by_side_clip(session_dir, cam_frames, start, end):
    (t0, names0), (t1, names1) = cam_frames

    mask0 = (t0 >= start) & (t0 <= end)
    idx0 = np.where(mask0)[0]
    idx0 = idx0[np.argsort(t0[idx0])]
    if len(idx0) == 0:
        print(f'side-by-side: no cam0 frames in {start:g}-{end:g}s, skipping')
        return

    out_path = None
    writer = None
    n_written = 0

    for i in idx0:
        frame0 = cv2.imread(str(session_dir / CAMERAS[0] / names0[i]))
        if frame0 is None:
            continue

        j = np.argmin(np.abs(t1 - t0[i]))
        frame1 = cv2.imread(str(session_dir / CAMERAS[1] / names1[j]))
        if frame1 is None:
            continue

        if frame1.shape[:2] != frame0.shape[:2]:
            frame1 = cv2.resize(frame1, (frame0.shape[1], frame0.shape[0]))

        combined = cv2.hconcat([frame0, frame1])

        if writer is None:
            height, width = combined.shape[:2]
            out_path = session_dir / f'side_by_side_clip_{start:g}-{end:g}s.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(out_path), fourcc, CLIP_FPS, (width, height))

        writer.write(combined)
        n_written += 1

    if writer is not None:
        writer.release()
        print(f'side-by-side: saved clip ({n_written} frames) to {out_path}')
    else:
        print('side-by-side: no valid frame pairs found, skipping')


def main():
    session_dir = pick_session_dir()
    print(f'Session: {session_dir}')

    cam_frames = []
    for camera in CAMERAS:
        frame_times, frame_names = load_frames(session_dir, camera)
        cam_frames.append((frame_times, frame_names))
        build_clip(session_dir, camera, frame_times, frame_names, CLIP_START, CLIP_END)

    build_side_by_side_clip(session_dir, cam_frames, CLIP_START, CLIP_END)


if __name__ == '__main__':
    main()
