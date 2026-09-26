"""Algorithmic error across the full phantom battery, written to CSV.

Each (activity, attenuation) pair is evaluated twice on the same grid: once by
the multilevel schedule and once by direct evaluation (n_theta_min = n_theta,
which performs no interpolation).  The difference is what the schedule costs.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from tomogrid import Options, forward, phantoms as ph
from tomogrid.image import sample_function, square_grid

OUT = Path(__file__).resolve().parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-grid", type=int, default=33)
    ap.add_argument("--n-theta-min", type=int, default=8)
    args = ap.parse_args()

    n, n_theta, n_s = args.n_grid, args.n_grid - 1, args.n_grid
    xa, ya = square_grid(n)
    ml_opt = Options(n_theta_min=args.n_theta_min, gl_order=6)
    dir_opt = Options(n_theta_min=n_theta, gl_order=6)

    rows = []
    t0 = time.perf_counter()
    for case in ph.battery():
        f_name, mu_name = case.name.split("|")
        f = sample_function(case.f, xa, ya)
        mu = sample_function(case.mu, xa, ya)
        kw = dict(n_theta=n_theta, n_s=n_s, support_radius=case.support_radius)
        a = forward(f, mu, options=ml_opt, **kw)
        b = forward(f, mu, options=dir_opt, **kw)
        scale = float(np.abs(b.S).max())
        rows.append(dict(
            activity=f_name, attenuation=mu_name,
            e_S=float(np.abs(a.S - b.S).max() / scale),
            e_E=float(np.abs(a.E - b.E).max() / max(np.abs(b.E).max(), 1e-300)),
            e_I=float(np.abs(a.I - b.I).max() / scale),
            peak_E=float(b.E.max()),
        ))
        print(f"{case.name:38s} e_I={rows[-1]['e_I']:.2e} "
              f"peak_E={rows[-1]['peak_E']:.2f}", flush=True)

    OUT.mkdir(exist_ok=True)
    path = OUT / "battery.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    worst = max(rows, key=lambda r: max(r["e_S"], r["e_E"], r["e_I"]))
    print(f"\nworst: {worst['activity']}|{worst['attenuation']} "
          f"{max(worst['e_S'], worst['e_E'], worst['e_I']):.2e}")
    print(f"wrote {path}  ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
