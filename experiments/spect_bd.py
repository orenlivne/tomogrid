"""SPECT/PET on the Brandt--Dym indexing: accuracy against work, 2-D.

Mirrors the structure of the Brandt--Dym experiments (bdlines/) but for the
attenuated transform, where the merge carries ``exp(-E)`` and the interpolation
must therefore be applied to the halves *before* concatenating, not after.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from tomogrid import phantoms as ph
from tomogrid.bd import build_grid, forward
from tomogrid.image import sample_function, square_grid
from tomogrid.reference import naive_segments

OUT = Path(__file__).resolve().parent / "results"
F = ph.ACTIVITIES["three_blobs"][0]
MU = ph.ATTENUATIONS["high_contrast"][0]


def work_count(grid, order: int, gl_order: int, img_order: int) -> dict:
    """Base-level gathers plus doubling multiply--adds, for all four sweeps."""
    lv0 = grid.levels[0]
    base = lv0.n_x * grid.n_y * lv0.n_m
    base_ops = base * gl_order * (2 * img_order + gl_order)
    ops = 0
    entries = 0
    for k in range(len(grid.levels) - 1):
        nxt = grid.levels[k + 1]
        n_even = grid.levels[k].n_m
        n_odd = nxt.n_m - n_even
        per = nxt.n_x * grid.n_y
        entries += per * nxt.n_m
        # 3 fields; even: add + halve, odd: 2*order multiply-adds; plus one exp
        ops += per * (3 * (2 * n_even + n_odd * 4 * order) + nxt.n_m * 6)
    total = 4 * (base_ops + ops)  # four direction families
    return {"entries": 4 * (base + entries), "ops": total,
            "base_ops": 4 * base_ops, "doubling_ops": 4 * ops}


def measure(n_grid, sigma, order, *, n_samples=1025, gl_order=8, img_order=4):
    xa, ya = square_grid(n_grid)
    f, mu = sample_function(F, xa, ya), sample_function(MU, xa, ya)
    t0 = time.perf_counter()
    sweeps = forward(f, mu, sigma=sigma, order=order, gl_order=gl_order,
                     img_order=img_order)
    secs = time.perf_counter() - t0
    grid = build_grid(half_width=1.0, image_h=xa.h, sigma=sigma)
    w = work_count(grid, order, gl_order, img_order)

    errs = {"S": 0.0, "E": 0.0, "I": 0.0}
    for sw in sweeps.values():
        t = sw.integrals()
        S, E, I = naive_segments(F, MU, sw.p0, sw.p1, n_samples=n_samples)
        S, E, I = (a.reshape(t.S.shape) for a in (S, E, I))
        scale = float(np.abs(S).max())
        errs["S"] = max(errs["S"], float(np.abs(t.S - S).max() / scale))
        errs["E"] = max(errs["E"], float(np.abs(t.E - E).max()
                                         / max(float(np.abs(E).max()), 1e-30)))
        errs["I"] = max(errs["I"], float(np.abs(t.I - I).max() / scale))
    n_lines = 4 * sweeps["+x"].triple.S.size
    return dict(n_grid=n_grid, sigma=sigma, order=order, seconds=secs,
                n_lines=n_lines, ops=w["ops"], entries=w["entries"],
                base_ops=w["base_ops"], doubling_ops=w["doubling_ops"],
                **{f"err_{k}": v for k, v in errs.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", type=int, nargs="+", default=[33, 65, 129, 257])
    ap.add_argument("--sigmas", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--orders", type=int, nargs="+", default=[2, 4, 6])
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    print("=== accuracy vs the integration mesh and the interpolation order "
          "(n = 129) ===", flush=True)
    knob = []
    for order in args.orders:
        for sigma in args.sigmas:
            r = measure(129, sigma, order)
            knob.append(r)
            print(f"  order={order} sigma={sigma}: err_I={r['err_I']:.3e} "
                  f"ops={r['ops']/1e6:9.1f} M  {r['seconds']:.2f}s", flush=True)

    print("\n=== refinement at fixed parameters (order 4, sigma 2) ===",
          flush=True)
    ref = []
    for n in args.grids:
        r = measure(n, 2, 4)
        ref.append(r)
        print(f"  n={n:4d}: err_I={r['err_I']:.3e}  lines={r['n_lines']:8d}  "
              f"ops={r['ops']/1e6:9.1f} M  {r['seconds']:6.2f}s", flush=True)

    for name, rows in (("spect_bd_knobs.csv", knob),
                       ("spect_bd_refine.csv", ref)):
        with open(OUT / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"wrote {OUT / name}")


if __name__ == "__main__":
    main()
