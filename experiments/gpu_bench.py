"""Measure the transform on whatever device is present.

This is the script that turns the bounds in docs/gpu.md into measurements.  It
has never been run on a GPU; on a host without one it runs under NumPy and
says so, which is also the way to check that it works before renting anything.

  python -m experiments.gpu_bench --grids 129 257 513

Reports, per grid: wall time for the whole four-family transform, wall time
for the recursion alone, the traffic the counts say was moved, and the
effective bandwidth that implies.  Compare the last column with the device's
published figure: that ratio, not the speed-up against NumPy, is what says
whether the kernels are any good.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from tomogrid import phantoms as ph
from tomogrid.backend import SinglePrecision, is_gpu, resolve, sync, to_host
from tomogrid.bd import build_grid, build_level0, forward, refine
from tomogrid.gpucost import forward_counts, refine_counts
from tomogrid.image import sample_function, square_grid

F = ph.ACTIVITIES["three_blobs"][0]
MU = ph.ATTENUATIONS["high_contrast"][0]


def describe(xp) -> str:
    if not is_gpu(xp):
        return "numpy (no device)"
    props = xp.cuda.runtime.getDeviceProperties(0)
    name = props["name"].decode() if isinstance(props["name"], bytes) \
        else props["name"]
    free, total = xp.cuda.runtime.memGetInfo()
    return f"cupy on {name}, {total / 1e9:.0f} GB, {free / 1e9:.0f} GB free"


def run(n, sigma, order, xp, repeats):
    xa, ya = square_grid(n)
    f, mu = sample_function(F, xa, ya), sample_function(MU, xa, ya)
    grid = build_grid(half_width=1.0, image_h=xa.h, sigma=sigma)

    forward(f, mu, sigma=sigma, order=order, xp=xp)  # warm up
    sync(xp)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        sweeps = forward(f, mu, sigma=sigma, order=order, xp=xp)
        sync(xp)
        best = min(best, time.perf_counter() - t0)

    fo = f.to(xp)
    mo = mu.to(xp)
    tri = build_level0(fo, mo, grid, xp=xp)
    sync(xp)
    t0 = time.perf_counter()
    for k in range(grid.levels[-1].index):
        tri = refine(tri, grid, k, order=order, xp=xp)
    sync(xp)
    rec_secs = time.perf_counter() - t0

    counts = forward_counts(grid, order=order)
    rec = None
    for k in range(len(grid.levels) - 1):
        c = refine_counts(grid, k, order)
        rec = c if rec is None else rec + c
    nbytes = 8 if not isinstance(xp, SinglePrecision) else 4
    return {
        "lines": 4 * grid.n_y * grid.top.n_m,
        "secs": best,
        "rec_secs": rec_secs,
        "total_GB": counts.bytes_moved(nbytes) / 1e9,
        "rec_GB": rec.bytes_moved(nbytes) / 1e9,
        "rec_BW": rec.bytes_moved(nbytes) / rec_secs / 1e9,
        "checksum": float(np.abs(to_host(sweeps["+x"].triple.I)).max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", type=int, nargs="+", default=[129, 257])
    ap.add_argument("--sigma", type=int, default=2)
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()

    xp = resolve(args.backend)
    if args.fp32:
        xp = SinglePrecision(xp)
    print(describe(resolve(args.backend))
          + (", single precision" if args.fp32 else ", double precision"))
    print(f"\n{'n':>5} {'lines':>12} {'total s':>9} {'recursion s':>12} "
          f"{'rec GB':>8} {'GB/s':>8} {'max |I|':>12}")
    for n in args.grids:
        r = run(n, args.sigma, args.order, xp, args.repeats)
        print(f"{n:5d} {r['lines']:12,d} {r['secs']:9.3f} "
              f"{r['rec_secs']:12.3f} {r['rec_GB']:8.3f} {r['rec_BW']:8.1f} "
              f"{r['checksum']:12.6f}")
    print("\nGB/s is the recursion's traffic over its wall time.  Against a "
          "device's published bandwidth it says how much of the hardware the "
          "kernels reach; the NumPy path reaches under 1 GB/s of ~100.")


if __name__ == "__main__":
    main()
