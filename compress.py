"""Compress a FLIES folder for transfer: image sequences -> videos,
everything else copied.  The original folder is never modified.

    python compress_flies.py                         # SRC/DST from config
    python compress_flies.py "E:/Ben Christensen/FLIES" "F:/FLIES_compressed"

WHAT IT DOES
    Walks every folder under SRC and mirrors it into DST.
    - Image sequences named like the acquisition scripts write them
          e001_0000.png ...     (bursts: FLIR, Brio)
          p003_000000.jpg ...   (continuous phases)
      become ONE video per sequence, next to a manifest:
          flir/e001.mp4   + flir/e001_frames.csv
          brio_cont/p003.mp4 + brio_cont/p003_frames.csv
      Frame i of the video is exactly the i-th original image, in order.
      The manifest lists frame_index, original filename, frame number and
      timestamp (from <folder>_frames.csv), so every frame can be traced
      back.  Gaps in the numbering (dropped continuous frames) are kept
      as gaps in the manifest, not filled.
    - Everything else (CSVs, meta.txt, analysis PNGs, ...) is copied as-is.

CODECS
    16-bit mono (FLIR):  FLIR_MODE = 'h264'  -> 8-bit H.264, visually
                                                 lossless, ~30-100x smaller
                         FLIR_MODE = 'ffv1'  -> exact 16-bit lossless (.mkv),
                                                 but barely smaller than PNG
    8-bit / colour (Brio): H.264.  They were JPEGs already.

SAFETY
    Every video's frame count is checked with ffprobe before it is marked
    done.  A video only counts as finished once its manifest exists, so
    re-running skips finished work and redoes anything interrupted.
    Problems are listed in DST/compress_log.csv.

GETTING FRAMES BACK
    python compress_flies.py --restore "F:/FLIES_compressed/session_X"
        decodes every video in that session back into its own folder
        (flir/e001_0000.png ... next to flir/e001.mp4), original names, so
        event_video.py, blip_viewer.py etc. run on it unchanged.
    python compress_flies.py --extract "F:/.../flir/e001.mp4" [outdir]
        just one video (default outdir: F:/.../flir/e001_frames/)
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

import cv2
import numpy as np

# ============ CONFIG ============
SRC = Path(r'E:\Ben Christensen\FLIES')
DST = None                    # None = sibling folder "FLIES_compressed"

FFMPEG = r'E:\Ben Christensen\ffmpeg-2026-09-21-git-a9cbcc2bbb-essentials_build\ffmpeg-2026-09-21-git-a9cbcc2bbb-essentials_build\bin\ffmpeg.exe'

FLIR_MODE  = 'h264'           # 'h264' (8-bit, small) or 'ffv1' (16-bit exact)
MONO16_CRF = 16               # H.264 quality for FLIR; lower = better/bigger
COLOR_CRF  = 20               # H.264 quality for the Brios
PRESET     = 'medium'         # x264 speed: faster -> quicker, a bit bigger

N_JOBS     = max(1, (os.cpu_count() or 4) // 4)   # videos encoded at once
MIN_FRAMES = 2                # shorter "sequences" are just copied
VERIFY     = True             # ffprobe frame count on every video
HEARTBEAT_S = 60
# ================================

SEQ_RE = re.compile(r'^([ep]\d{3,})_(\d+)\.(png|jpe?g)$', re.I)
PRINT_LOCK = threading.Lock()
INDEX_CACHE = {}
INDEX_LOCK = threading.Lock()


def log(msg):
    with PRINT_LOCK:
        print(msg, flush=True)


def tool(name):
    """ffmpeg / ffprobe: from the FFMPEG path, else PATH."""
    p = Path(FFMPEG)
    if name != 'ffmpeg':
        p = p.with_name(p.name.replace('ffmpeg', name))
    if p.exists():
        return str(p)
    found = shutil.which(name)
    if not found:
        sys.exit(f'[error] {name} not found - set FFMPEG in the config')
    return found


def gb(n):
    return f'{n / 1e9:.2f} GB'


def fmt_time(s):
    s = int(s)
    return f'{s // 3600}h{(s % 3600) // 60:02d}m{s % 60:02d}s'


# ------------------------------------------------------------- scanning

def scan(src, dst):
    """Returns (sequences, copies).  Uses scandir so file sizes are free."""
    seqs, copies = [], []
    stack = [src]
    n_dirs = 0
    last = time.perf_counter()
    while stack:
        d = stack.pop()
        if d == dst or dst in d.parents:
            continue
        rel = d.relative_to(src)
        groups = {}
        try:
            entries = list(os.scandir(d))
        except OSError as e:
            log(f'[scan] skipped {d}: {e}')
            continue
        for e in entries:
            if e.is_dir(follow_symlinks=False):
                stack.append(Path(e.path))
                continue
            size = e.stat().st_size
            m = SEQ_RE.match(e.name)
            if m:
                groups.setdefault(m.group(1).lower(), []).append(
                    (int(m.group(2)), e.name, size))
            else:
                copies.append((Path(e.path), dst / rel / e.name, size))
        for prefix, items in groups.items():
            if len(items) < MIN_FRAMES:
                copies += [(d / fn, dst / rel / fn, sz) for _, fn, sz in items]
                continue
            items.sort()
            seqs.append({'dir': d, 'rel': rel, 'prefix': prefix,
                         'nums': [n for n, _, _ in items],
                         'files': [fn for _, fn, _ in items],
                         'bytes': sum(sz for _, _, sz in items)})
        n_dirs += 1
        if time.perf_counter() - last > 10:
            log(f'[scan] {n_dirs:,} folders, {len(seqs):,} sequences so far')
            last = time.perf_counter()
    seqs.sort(key=lambda s: (str(s['rel']), s['prefix']))
    return seqs, copies


def frame_times(folder):
    """{filename: time} from <parent>/<folder name>_frames.csv, cached."""
    with INDEX_LOCK:
        if folder in INDEX_CACHE:
            return INDEX_CACHE[folder]
    idx = folder.parent / f'{folder.name}_frames.csv'
    times = {}
    if idx.exists():
        try:
            with open(idx, newline='') as f:
                for row in csv.DictReader(f):
                    times[row['filename']] = row['time']
        except Exception as e:
            log(f'[warn] could not read {idx}: {e}')
    with INDEX_LOCK:
        INDEX_CACHE[folder] = times
    return times


# ------------------------------------------------------------- encoding

def encode(seq, dst, ffmpeg, ffprobe):
    d, prefix, files = seq['dir'], seq['prefix'], seq['files']
    out_dir = dst / seq['rel']
    out_dir.mkdir(parents=True, exist_ok=True)

    first = cv2.imread(str(d / files[0]), cv2.IMREAD_UNCHANGED)
    if first is None:
        return {'status': 'error', 'why': f'cannot read {files[0]}'}
    mono16 = first.dtype == np.uint16 and first.ndim == 2

    lossless = mono16 and FLIR_MODE == 'ffv1'
    ext = '.mkv' if lossless else '.mp4'
    out = out_dir / f'{prefix}{ext}'
    manifest = out_dir / f'{prefix}_frames.csv'
    if out.exists() and manifest.exists():
        return {'status': 'skipped', 'out_bytes': out.stat().st_size}

    # nominal fps from the timestamps; real times live in the manifest
    times = frame_times(d)
    ts = [times.get(fn) for fn in files]
    tv = np.array([float(t) for t in ts if t not in (None, '')])
    if len(tv) > 2 and np.median(np.diff(tv)) > 0:
        fps = 1.0 / float(np.median(np.diff(tv)))
    else:
        fps = 60.0 if mono16 else (30.0 if prefix.startswith('e') else 20.0)

    listfile = out_dir / f'.{prefix}.ffconcat'
    with open(listfile, 'w', encoding='utf-8') as f:
        f.write('ffconcat version 1.0\n')
        for fn in files:
            p = (d / fn).as_posix().replace("'", "'\\''")
            f.write(f"file '{p}'\nduration {1.0 / fps:.6f}\n")

    tmp = out_dir / f'{prefix}.partial{ext}'
    cmd = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
           '-f', 'concat', '-safe', '0', '-i', str(listfile)]
    if lossless:
        cmd += ['-c:v', 'ffv1', '-level', '3', '-g', '1', '-slices', '16',
                '-slicecrc', '1', '-pix_fmt', 'gray16le']
    else:
        cmd += ['-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                '-c:v', 'libx264', '-preset', PRESET,
                '-crf', str(MONO16_CRF if mono16 else COLOR_CRF),
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart']
    cmd += ['-fps_mode', 'passthrough', '-an', str(tmp)]

    r = subprocess.run(cmd, capture_output=True, text=True)
    listfile.unlink(missing_ok=True)
    if r.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        return {'status': 'error', 'why': (r.stderr or 'ffmpeg failed')
                .strip().splitlines()[-1][:300]}

    if VERIFY:
        pr = subprocess.run(
            [ffprobe, '-v', 'error', '-select_streams', 'v:0',
             '-count_packets', '-show_entries', 'stream=nb_read_packets',
             '-of', 'csv=p=0', str(tmp)], capture_output=True, text=True)
        try:
            n_vid = int(pr.stdout.strip().split(',')[0])
        except ValueError:
            n_vid = -1
        if n_vid != len(files):
            return {'status': 'error',
                    'why': f'frame count {n_vid} != {len(files)} images '
                           f'(left as {tmp.name})'}

    os.replace(tmp, out)
    with open(manifest, 'w', newline='') as f:        # written last = done
        w = csv.writer(f)
        w.writerow(['frame_index', 'filename', 'number', 'time'])
        for i, (fn, n, t) in enumerate(zip(files, seq['nums'], ts)):
            w.writerow([i, fn, n, t if t is not None else ''])
    return {'status': 'ok', 'out_bytes': out.stat().st_size, 'fps': fps,
            'codec': 'ffv1' if lossless else 'h264',
            'kind': 'mono16' if mono16 else 'color/8-bit'}


# ------------------------------------------------------------- main run

def compress(src, dst):
    ffmpeg, ffprobe = tool('ffmpeg'), tool('ffprobe')
    print(f'[src] {src}\n[dst] {dst}\n[ffmpeg] {ffmpeg}')
    print(f'[mode] FLIR {FLIR_MODE}, {N_JOBS} videos at a time, '
          f'preset {PRESET}\n')
    dst.mkdir(parents=True, exist_ok=True)

    print('[scan] walking the folder tree ...')
    seqs, copies = scan(src, dst)
    seq_bytes = sum(s['bytes'] for s in seqs)
    copy_bytes = sum(c[2] for c in copies)
    n_frames = sum(len(s['files']) for s in seqs)
    print(f'[scan] {len(seqs):,} image sequences ({n_frames:,} frames, '
          f'{gb(seq_bytes)}), {len(copies):,} other files ({gb(copy_bytes)})\n')

    # --- plain copies first (fast, small) ---
    t0 = time.perf_counter()
    n_copied = 0
    for i, (s, dd, size) in enumerate(copies):
        if dd.exists() and dd.stat().st_size == size:
            continue
        dd.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, dd)
        n_copied += 1
        if (i + 1) % 500 == 0:
            log(f'[copy] {i + 1:,}/{len(copies):,}')
    print(f'[copy] {n_copied:,} copied, {len(copies) - n_copied:,} already '
          f'there ({time.perf_counter() - t0:.0f} s)\n')

    # --- videos ---
    log_rows = []
    done_bytes = 0
    out_total = 0
    t0 = time.perf_counter()
    last_done = t0
    work_bytes = 0          # bytes actually encoded (not skipped), for ETA
    ex = ThreadPoolExecutor(max_workers=N_JOBS)
    futs = {ex.submit(encode, s, dst, ffmpeg, ffprobe): s for s in seqs}
    pending = set(futs)
    n_done = 0
    try:
        while pending:
            finished, pending = wait(pending, timeout=5,
                                     return_when=FIRST_COMPLETED)
            now = time.perf_counter()
            if not finished:
                if now - last_done > HEARTBEAT_S:
                    log(f'[video] still working ... {n_done}/{len(seqs)} '
                        f'done, {len(pending)} left')
                    last_done = now
                continue
            for fut in finished:
                s = futs[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    res = {'status': 'error', 'why': repr(e)}
                n_done += 1
                last_done = now
                done_bytes += s['bytes']
                ob = res.get('out_bytes', 0)
                out_total += ob
                if res['status'] == 'ok':
                    work_bytes += s['bytes']
                name = f'{s["rel"].as_posix()}/{s["prefix"]}'
                el = now - t0
                rate = work_bytes / el if el > 0 and work_bytes else 0
                remaining = seq_bytes - done_bytes
                eta = remaining / rate if rate else 0
                if res['status'] == 'ok':
                    ratio = s['bytes'] / ob if ob else 0
                    log(f'[{n_done}/{len(seqs)}] {name}: '
                        f'{len(s["files"])} frames {res["kind"]} '
                        f'{s["bytes"] / 1e6:.0f} -> {ob / 1e6:.1f} MB '
                        f'({ratio:.0f}x)   {gb(done_bytes)}/{gb(seq_bytes)}'
                        f'  ETA {fmt_time(eta)}')
                elif res['status'] == 'skipped':
                    log(f'[{n_done}/{len(seqs)}] {name}: already done')
                else:
                    log(f'[{n_done}/{len(seqs)}] {name}: ERROR {res["why"]}')
                log_rows.append({
                    'sequence': name, 'status': res['status'],
                    'frames': len(s['files']), 'in_bytes': s['bytes'],
                    'out_bytes': ob, 'codec': res.get('codec', ''),
                    'fps': f'{res["fps"]:.3f}' if 'fps' in res else '',
                    'note': res.get('why', '')})
    except KeyboardInterrupt:
        print('\n[stop] Ctrl+C - finished videos are kept; rerun to resume')
        ex.shutdown(wait=False, cancel_futures=True)
        sys.exit(1)
    ex.shutdown(wait=True)

    with open(dst / 'compress_log.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys())
                           if log_rows else ['sequence'])
        w.writeheader()
        w.writerows(log_rows)
    write_readme(dst)

    errs = [r for r in log_rows if r['status'] == 'error']
    total_in = seq_bytes + copy_bytes
    total_out = out_total + copy_bytes
    print(f'\n[done] {len(seqs) - len(errs)}/{len(seqs)} sequences OK in '
          f'{fmt_time(time.perf_counter() - t0)}')
    print(f'[done] {gb(total_in)} -> {gb(total_out)} '
          f'({total_in / max(total_out, 1):.1f}x smaller)')
    if errs:
        print(f'[done] {len(errs)} FAILED - see {dst / "compress_log.csv"}; '
              f'their images were NOT converted, so copy those folders over '
              f'raw or rerun after fixing')
        for r in errs[:10]:
            print(f'         {r["sequence"]}: {r["note"]}')


def write_readme(dst):
    (dst / 'README_compressed.txt').write_text(f"""\
Compressed copy of a FLIES folder, made by compress_flies.py.

Every image sequence (e###_####.png/.jpg bursts, p###_######.jpg
continuous phases) is one video: <prefix>.mp4 (or .mkv for lossless
16-bit FLIR).  Next to it, <prefix>_frames.csv maps

    frame_index -> original filename, frame number, timestamp

Frame i of the video IS the i-th original image, one-to-one, in order.
Timestamps come from the session's <folder>_frames.csv, which is also
copied unchanged.  Everything that was not an image sequence is an exact
copy.

FLIR mode used: {FLIR_MODE}
    h264 = 8-bit (16-bit FLIR images scaled to 8 bits), visually lossless
    ffv1 = exact 16-bit, lossless

To get the frames back with their original names:
    python compress_flies.py --extract path/to/e001.mp4
""")


# ------------------------------------------------------------- extract

def extract(video, out_dir=None):
    video = Path(video)
    manifest = video.with_name(f'{video.stem}_frames.csv')
    if not manifest.exists():
        sys.exit(f'[error] no manifest {manifest}')
    with open(manifest, newline='') as f:
        names = [row['filename'] for row in csv.DictReader(f)]
    out_dir = Path(out_dir) if out_dir else \
        video.with_name(f'{video.stem}_frames')
    out_dir.mkdir(parents=True, exist_ok=True)

    is_jpg = names[0].lower().endswith(('.jpg', '.jpeg'))
    tmp_ext = '.jpg' if is_jpg else '.png'
    ffmpeg = tool('ffmpeg')
    cmd = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
           '-i', str(video), '-fps_mode', 'passthrough',
           '-enc_time_base', '-1']      # keep 1 frame per frame, no pts clash
    if is_jpg:
        cmd += ['-q:v', '2']
    elif video.suffix == '.mkv':
        cmd += ['-pix_fmt', 'gray16be']          # keep 16-bit PNGs
    else:
        cmd += ['-pix_fmt', 'gray']              # FLIR h264 -> 8-bit PNG
    cmd += ['-start_number', '0', str(out_dir / f'__tmp_%07d{tmp_ext}')]
    print(f'[extract] {video.name} -> {out_dir}')
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f'[error] ffmpeg: {r.stderr.strip()[-300:]}')

    tmp = sorted(out_dir.glob(f'__tmp_*{tmp_ext}'))
    if len(tmp) != len(names):
        print(f'[warn] {len(tmp)} frames decoded, manifest has {len(names)}')
    for p, name in zip(tmp, names):
        os.replace(p, out_dir / name)
    print(f'[extract] {min(len(tmp), len(names))} frames with original names')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src', nargs='?', default=str(SRC))
    ap.add_argument('dst', nargs='?', default=None)
    ap.add_argument('--extract', metavar='VIDEO',
                    help='decode one video back to its original filenames')
    ap.add_argument('--restore', metavar='FOLDER',
                    help='decode every video under FOLDER back into its own '
                         'folder (flir/, brio1/ ...) with original names, so '
                         'the analysis scripts run on it')
    args = ap.parse_args()

    if args.extract:
        # "--extract video outdir": outdir arrives as the first positional
        out = args.dst or (args.src if args.src != str(SRC) else None)
        extract(args.extract, out)
        return
    if args.restore:
        vids = [m.with_name(m.name[:-len('_frames.csv')] + ext)
                for m in sorted(Path(args.restore).rglob('*_frames.csv'))
                for ext in ('.mp4', '.mkv')]
        vids = [v for v in vids if v.exists()]
        print(f'[restore] {len(vids)} videos under {args.restore}')
        for i, v in enumerate(vids, 1):
            print(f'[{i}/{len(vids)}] ', end='')
            extract(v, v.parent)
        return

    src = Path(args.src).resolve()
    dst = Path(args.dst) if args.dst else \
        (Path(DST) if DST else src.parent / f'{src.name}_compressed')
    dst = dst.resolve()
    if dst == src:
        sys.exit('[error] destination must differ from source')
    compress(src, dst)


if __name__ == '__main__':
    main()