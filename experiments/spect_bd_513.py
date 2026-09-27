"""One more 2-D refinement point at n = 513.

The reference is subsampled: at this size the four families carry 8.4 million
lines, and a dense-trapezoid check on all of them costs far more than the
transform it is checking.  Every 4th offset and every 8th slope is enough to
bound the sup error.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from tomogrid import phantoms as ph
from tomogrid.bd import build_grid, forward
from tomogrid.image import sample_function, square_grid
from tomogrid.reference import naive_segments
from experiments.spect_bd import F, MU, work_count

OUT = Path(__file__).resolve().parent / "results"
SUB = (slice(None, None, 4), slice(None, None, 8))


def main(n_grid=513, sigma=2, order=4, gl_order=8, img_order=4, n_samples=1025):
    xa, ya = square_grid(n_grid)
    f, mu = sample_function(F, xa, ya), sample_function(MU, xa, ya)
    t0 = time.perf_counter()
    sweeps = forward(f, mu, sigma=sigma, order=order, gl_order=gl_order,
                     img_order=img_order)
    secs = time.perf_counter() - t0
    grid = build_grid(half_width=1.0, image_h=xa.h, sigma=sigma)
    w = work_count(grid, order, gl_order, img_order)
    print(f"transform: {secs:.1f}s  {w['ops']/1e6:.1f} Mflop", flush=True)

    err = {"S": 0.0, "I": 0.0}
    for name, sw in sweeps.items():
        t = sw.integrals()
        p0, p1 = sw.p0[SUB], sw.p1[SUB]
        S, _, I = naive_segments(F, MU, p0, p1, n_samples=n_samples)
        scale = float(np.abs(S).max())
        err["S"] = max(err["S"], float(np.abs(t.S[SUB].ravel() - S).max() / scale))
        err["I"] = max(err["I"], float(np.abs(t.I[SUB].ravel() - I).max() / scale))
        print(f"  {name}: err_I={err['I']:.3e}", flush=True)

    row = dict(n_grid=n_grid, sigma=sigma, order=order, seconds=secs,
               n_lines=4 * sweeps["+x"].triple.S.size, ops=w["ops"],
               entries=w["entries"], base_ops=w["base_ops"],
               doubling_ops=w["doubling_ops"],
               err_S=err["S"], err_E=float("nan"), err_I=err["I"])
    OUT.mkdir(exist_ok=True)
    path = OUT / "spect_bd_513.csv"
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(row))
        wr.writeheader(); wr.writerow(row)
    print(f"\nn={n_grid}: err_I={err['I']:.3e} lines={row['n_lines']} "
          f"ops={w['ops']/1e6:.1f}M {secs:.2f}s\nwrote {path}")


if __name__ == "__main__":
    main()
