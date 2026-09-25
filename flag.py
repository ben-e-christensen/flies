"""Summarize what flag_odd_frames.py flagged.

Reads frame_check/flagged.csv (and scores.csv for totals) and prints:
  - how many frames were flagged, and by which reason
  - the breakdown per session and per lamp phase
  - contiguous RUNS of flagged frames (blocks, not scattered singles)
  - the most extreme frames for each metric
  - whether the flagged files still exist on disk (explains copy errors)

Then optionally copies the MOST EXTREME frames (not the first N) so what you
flip through is actually representative.
"""

import csv
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

# ============ CONFIG ============
OUT_DIR = Path(r'E:\Ben Christensen\FLIES\frame_check')

CHECK_EXISTS  = True     # stat every flagged file (slow-ish on a big list)
COPY_EXTREME  = True     # copy the worst offenders
COPY_N        = 60       # per metric
COPY_DEST     = OUT_DIR / 'flagged_extreme'
# ================================

METRICS = ('mad', 'frac', 'sharp', 'bright')
FRAME_RE = re.compile(r'p(\d+)_(\d+)\.png$', re.I)


def long_path(p):
    """Windows \\\\?\\ prefix so >260 char paths and odd names still work."""
    p = os.path.abspath(str(p))
    if os.name == 'nt' and not p.startswith('\\\\?\\'):
        return '\\\\?\\' + p
    return p


def load(path):
    rows = []
    with open(path, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def key_of(p):
    """(session, phase, frame_number) from a frame path."""
    p = Path(p)
    m = FRAME_RE.search(p.name)
    phase, num = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    return p.parent.parent.name, phase, num


def runs(nums, gap=1):
    """Collapse sorted ints into (start, end, count) blocks."""
    out = []
    start = prev = None
    for n in nums:
        if start is None:
            start = prev = n
        elif n - prev <= gap:
            prev = n
        else:
            out.append((start, prev, prev - start + 1))
            start = prev = n
    if start is not None:
        out.append((start, prev, prev - start + 1))
    return out


def main():
    flagged_path = OUT_DIR / 'flagged.csv'
    scores_path = OUT_DIR / 'scores.csv'
    flagged = load(flagged_path)
    n_total = sum(1 for _ in open(scores_path)) - 1

    print(f'{len(flagged):,} flagged out of {n_total:,} scored '
          f'({100 * len(flagged) / max(n_total, 1):.2f}%)\n')

    # --- reasons ---
    print('reasons (a frame can have several):')
    c = Counter()
    for row in flagged:
        for r in row['reasons'].split('+'):
            c[r] += 1
    for r, n in c.most_common():
        print(f'  {r:<20} {n:>8,}')

    # --- per session / phase ---
    by_sess = defaultdict(list)
    for row in flagged:
        sess, phase, num = key_of(row['path'])
        by_sess[(sess, phase)].append(num)

    print('\nper session / phase:')
    for (sess, phase), nums in sorted(by_sess.items()):
        nums.sort()
        blocks = runs(nums)
        big = [b for b in blocks if b[2] >= 5]
        print(f'  {sess}  p{phase:03d}  {len(nums):>7,} flagged, '
              f'{len(blocks):>5,} blocks, longest {max(b[2] for b in blocks):,}')
        for s, e, n in sorted(big, key=lambda b: -b[2])[:5]:
            print(f'      block {s:06d}-{e:06d}  ({n:,} frames)')

    # --- most extreme per metric ---
    print('\nmost extreme frames:')
    for m in METRICS:
        rows = [r for r in flagged if r.get(f'z_{m}') not in (None, '', 'nan')]
        if not rows:
            continue
        rows.sort(key=lambda r: -abs(float(r[f'z_{m}'])))
        print(f'  by {m}:')
        for r in rows[:5]:
            print(f'    z={float(r[f"z_{m}"]):>12.1f}  {r[m]:>12}  '
                  f'{Path(r["path"]).parent.parent.name}/{Path(r["path"]).name}')

    # --- do the files still exist? ---
    if CHECK_EXISTS:
        missing = [r['path'] for r in flagged
                   if not os.path.exists(long_path(r['path']))]
        print(f'\n{len(missing):,} of {len(flagged):,} flagged files are '
              f'NOT readable right now')
        for p in missing[:5]:
            print(f'    {p}')
        if missing:
            print('    (drive dropped out, or the files moved since the scan)')

    # --- copy the worst, not the first ---
    if COPY_EXTREME:
        picks = {}
        for m in METRICS:
            rows = [r for r in flagged
                    if r.get(f'z_{m}') not in (None, '', 'nan')]
            rows.sort(key=lambda r: -abs(float(r[f'z_{m}'])))
            for r in rows[:COPY_N]:
                picks.setdefault(r['path'], []).append(m)
        COPY_DEST.mkdir(parents=True, exist_ok=True)
        ok = bad = 0
        for p, ms in picks.items():
            src = Path(p)
            name = f'{"-".join(ms)}__{src.parent.parent.name}__{src.name}'
            try:
                shutil.copy2(long_path(src), long_path(COPY_DEST / name))
                ok += 1
            except OSError:
                bad += 1
        print(f'\ncopied {ok:,} extreme frames -> {COPY_DEST}'
              + (f'  ({bad:,} failed)' if bad else ''))


if __name__ == '__main__':
    main()