"""Compile per-event videos from FLIR burst frames.

Usage:
    python flir_video.py path/to/session_folder/flir
    python flir_video.py path/to/session_folder        (finds flir/ inside)

Groups frames by event number (e001_*.png, e002_*.png, …) and writes
one MP4 per event next to the flir folder.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

# ============ CONFIG ============
FPS       = 30        # playback fps for the output videos
CODEC     = 'mp4v'    # 'mp4v' is safe everywhere; 'avc1' if you want h264
NORMALIZE = True      # stretch 16-bit mono to full 8-bit range for video
# ================================


def main():
    if len(sys.argv) < 2:
        sys.exit('Usage: python flir_video.py <flir_folder or session_folder>')

    path = Path(sys.argv[1])
    flir_dir = path / 'flir' if (path / 'flir').is_dir() else path
    if not flir_dir.is_dir():
        sys.exit(f'Not a directory: {flir_dir}')

    out_dir = flir_dir.parent
    frames = sorted(flir_dir.glob('e*_*.*'))
    if not frames:
        sys.exit(f'No eNNN_NNNN.* frames found in {flir_dir}')

    # group by event prefix
    events = {}
    for f in frames:
        event_tag = f.stem.split('_')[0]   # 'e001'
        events.setdefault(event_tag, []).append(f)

    print(f'Found {len(frames)} frames across {len(events)} events')

    for tag in sorted(events):
        paths = sorted(events[tag])
        sample = cv2.imread(str(paths[0]), cv2.IMREAD_UNCHANGED)
        if sample is None:
            print(f'  [{tag}] could not read {paths[0].name}, skipping')
            continue

        h, w = sample.shape[:2]
        is_16bit = sample.dtype == np.uint16

        out_path = out_dir / f'{tag}.mp4'
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*CODEC),
                                 FPS, (w, h), isColor=False)

        for p in paths:
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if is_16bit:
                if NORMALIZE:
                    lo, hi = img.min(), img.max()
                    if hi > lo:
                        img = ((img - lo) / (hi - lo) * 255).astype(np.uint8)
                    else:
                        img = np.zeros_like(img, dtype=np.uint8)
                else:
                    img = (img >> 8).astype(np.uint8)
            writer.write(img)

        writer.release()
        dur = len(paths) / FPS
        print(f'  [{tag}] {len(paths)} frames -> {out_path.name}  ({dur:.1f}s @ {FPS} fps)')

    print('Done.')


if __name__ == '__main__':
    main()