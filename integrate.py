"""Integrate electrometer current data to recover charge vs time.

Usage:
    python integrate_current.py path/to/session_folder
    python integrate_current.py path/to/electrometer.csv

Reads the electrometer.csv (time,current in amps), numerically integrates
with the trapezoidal rule, and produces:

  1. A two-panel plot: current vs time (top) and cumulative charge vs time
     (bottom), saved as charge_integrated.png alongside the CSV.

  2. A new CSV (charge_integrated.csv) with columns:
        time, current, charge
     where charge is in coulombs.

NaN samples are interpolated before integration so gaps don't break the
cumulative sum.  The plot labels use pC/nA for readability but the CSV
stays in SI (amps / coulombs).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_session(path: Path):
    """Return (times, currents) arrays in SI from an electrometer.csv."""
    if path.is_dir():
        csv = path / 'electrometer.csv'
    else:
        csv = path
    if not csv.exists():
        sys.exit(f'Not found: {csv}')

    data = np.genfromtxt(csv, delimiter=',', skip_header=1, filling_values=np.nan)
    if data.ndim != 2 or data.shape[1] < 2:
        sys.exit(f'Unexpected shape in {csv}: {data.shape}')

    t = data[:, 0]
    i = data[:, 1]
    return t, i, csv.parent


def integrate(t, i):
    """Trapezoidal integration of current -> charge.

    NaNs are linearly interpolated first so the cumulative integral
    isn't broken by occasional bad readings.
    """
    i_clean = i.copy()
    nans = np.isnan(i_clean)
    if nans.any():
        good = ~nans
        if good.sum() >= 2:
            i_clean[nans] = np.interp(t[nans], t[good], i_clean[good])
        else:
            i_clean[nans] = 0.0

    # cumulative trapezoidal: Q(k) = sum of 0.5*(I[j]+I[j+1])*(t[j+1]-t[j])
    dt = np.diff(t)
    avg_i = 0.5 * (i_clean[:-1] + i_clean[1:])
    dq = avg_i * dt
    charge = np.concatenate(([0.0], np.cumsum(dq)))
    return charge, i_clean


def main():
    if len(sys.argv) < 2:
        sys.exit('Usage: python integrate_current.py <session_folder or electrometer.csv>')

    path = Path(sys.argv[1])
    t, i_raw, out_dir = load_session(path)
    charge, i_clean = integrate(t, i_raw)

    n_nan = np.isnan(i_raw).sum()
    print(f'Loaded {len(t)} samples, {n_nan} NaN interpolated')
    print(f'Time span: {t[-1] - t[0]:.3f} s')
    print(f'Peak current: {np.nanmax(np.abs(i_raw))*1e9:.3f} nA')
    print(f'Final integrated charge: {charge[-1]*1e12:.3f} pC')

    # ---- save CSV ----
    out_csv = out_dir / 'charge_integrated.csv'
    with open(out_csv, 'w') as f:
        f.write('time,current,charge\n')
        for tj, ij, qj in zip(t, i_raw, charge):
            f.write(f'{tj},{ij},{qj}\n')
    print(f'Wrote {out_csv}')

    # ---- plot ----
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

    ax1.plot(t, i_raw * 1e9, 'b-', linewidth=0.8, alpha=0.7, label='raw')
    if n_nan:
        ax1.plot(t, i_clean * 1e9, 'r-', linewidth=0.5, alpha=0.4,
                 label='interpolated')
        ax1.legend(fontsize=8)
    ax1.set_ylabel('Current [nA]')
    ax1.axhline(0, color='k', linewidth=0.3)
    ax1.set_title('Electrometer current')

    ax2.plot(t, charge * 1e12, 'g-', linewidth=1)
    ax2.set_ylabel('Charge [pC]')
    ax2.set_xlabel('Time [s]')
    ax2.axhline(0, color='k', linewidth=0.3)
    ax2.set_title('Cumulative charge (trapezoidal integration)')

    fig.tight_layout()
    out_png = out_dir / 'charge_integrated.png'
    fig.savefig(out_png, dpi=150)
    print(f'Wrote {out_png}')
    plt.show()


if __name__ == '__main__':
    main()