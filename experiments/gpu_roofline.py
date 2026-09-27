"""What the doubling recursion would cost on a GPU, and whether fp32 is enough.

Three separate things, none of which needs a GPU to be worth knowing:

1. ``--counts``   traffic and arithmetic intensity of the whole transform, from
   array shapes alone, with the roofline they imply on published peak figures.
   The recursion turns out to be bandwidth-bound by a wide margin, so the
   projection is essentially "bytes / HBM bandwidth" and does not depend on how
   well a kernel uses the arithmetic units.
2. ``--host``     the same traffic divided by the measured NumPy time, i.e. the
   effective bandwidth the host implementation achieves.  This is the number a
   device implementation has to be compared against, and it is measured.
3. ``--fp32``     the whole pipeline re-run in single precision against the
   double-precision answer.  The merge damps (its weights lie in ``(0, 1]``),
   so roundoff should not grow with the number of levels; if it does not, the
   effective bandwidth doubles and consumer cards become usable.

Nothing here is a GPU benchmark.  The device port is the ``xp`` argument
already threaded through :mod:`tomogrid.bd`; see docs/gpu.md.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from tomogrid import phantoms as ph
from tomogrid.backend import SinglePrecision
from tomogrid.bd import build_grid, forward, refine, build_level0
from tomogrid.bd3d import build_grid3d
from tomogrid.gpucost import (DEVICES, base_counts, direct_counts,
                              forward_counts, peak_table_entries,
                              peak_table_entries3d, refine_counts, roofline,
                              sweep3d_counts)
from tomogrid.image import sample_function, square_grid

F = ph.ACTIVITIES["three_blobs"][0]
MU = ph.ATTENUATIONS["high_contrast"][0]
GB = 1e9


def _images(n):
    xa, ya = square_grid(n)
    return sample_function(F, xa, ya), sample_function(MU, xa, ya), xa.h


def counts_table(grids, sigma, order):
    print(f"\n2-D, sigma={sigma}, order={order}, four direction families")
    print("counted from array shapes; interpolation taps on the image are "
          "charged as cache hits, one exp costs 20 flops in fp64\n")
    head = (f"{'n':>5} {'lines':>12} {'base GB':>8} {'rec GB':>8} "
            f"{'base f/B':>9} {'rec f/B':>8} {'live GB':>8} "
            + " ".join(f"{d.name.split()[0]:>9}" for d in DEVICES))
    print(head)
    print("-" * len(head))
    for n in grids:
        grid = build_grid(half_width=1.0, image_h=2.0 / (n - 1), sigma=sigma)
        base = base_counts(grid)
        rec = _recursion_counts(grid, order)
        c = forward_counts(grid, order=order)
        live = 4 * peak_table_entries(grid) * 8 / GB
        lines = 4 * grid.n_y * grid.top.n_m
        times = [roofline(c, d, 8) for d in DEVICES]
        print(f"{n:5d} {lines:12,d} {4*base.bytes_moved()/GB:8.3f} "
              f"{4*rec.bytes_moved()/GB:8.3f} {base.intensity():9.1f} "
              f"{rec.intensity():8.2f} {live:8.3f} "
              + " ".join(f"{1e3*t['t']:7.2f}m{t['bound'][0]}" for t in times))
    print("\nf/B is flops per byte of traffic.  The base level is "
          "arithmetic-dense, the recursion is not: a doubling moves three "
          "tables and does a few dozen flops per entry.")
    print("device columns: roofline ms in fp64, suffixed m (memory-bound) or "
          "c (compute-bound).  A bound, not a measurement.")
    against_direct(grids, sigma, order)


def _recursion_counts(grid, order):
    total = None
    for k in range(len(grid.levels) - 1):
        c = refine_counts(grid, k, order)
        total = c if total is None else total + c
    return total


def against_direct(grids, sigma, order, device=DEVICES[2]):
    """Roofline of the recursion against direct quadrature of the same lines.

    The direct method is charged nothing for its interpolation taps beyond
    reading the two fields once, so it comes out arithmetic-bound.  That is
    deliberately generous: the question is how much of the arithmetic saving
    survives on a machine where arithmetic is the cheap resource.
    """
    print(f"\nagainst direct quadrature of the same lines, on {device.name}")
    print(f"{'n':>5} " + " ".join(
        f"{p:>9} {'direct':>9} {'ratio':>6}" for p in ("fp64 ML", "fp32 ML")))
    for n in grids:
        grid = build_grid(half_width=1.0, image_h=2.0 / (n - 1), sigma=sigma)
        ml = forward_counts(grid, order=order)
        dr = direct_counts(grid, n)
        cells = []
        for db in (8, 4):
            a, b = roofline(ml, device, db), roofline(dr, device, db)
            cells.append(f"{1e3*a['t']:8.2f}m {1e3*b['t']:8.2f}m "
                         f"{b['t']/a['t']:5.1f}x")
        print(f"{n:5d} " + " ".join(cells))
    print("the recursion is bandwidth-bound, direct quadrature is "
          "arithmetic-bound, so the ratio is smaller than the flop ratio -- "
          "and still doubles with every refinement.")


def counts_table3d(grids, sigma, order):
    print(f"\n3-D, sigma={sigma}, order={order}, one direction family of six")
    head = (f"{'n':>5} {'entries':>14} {'traffic GB':>11} {'flop/byte':>10} "
            f"{'live GB':>9} " + " ".join(f"{d.name.split()[0]:>9}"
                                          for d in DEVICES))
    print(head)
    print("-" * len(head))
    for n in grids:
        grid = build_grid3d(half_width=1.0, image_h=2.0 / (n - 1), sigma=sigma)
        c = sweep3d_counts(grid, order=order)
        live = peak_table_entries3d(grid) * 8 / GB
        times = [roofline(c, d, 8)["t"] for d in DEVICES]
        print(f"{n:5d} {int(c.points):14,d} {c.bytes_moved()/GB:11.3f} "
              f"{c.intensity():10.3f} {live:9.3f} "
              + " ".join(f"{1e3*t:8.1f}m" for t in times))
    print("\n'live GB' is one family's three tables in fp64; it is the "
          "constraint in 3-D, not the arithmetic.")


def host_bandwidth(n, sigma, order):
    """Effective bandwidth the host implementation reaches, level by level."""
    f, mu, h = _images(n)
    grid = build_grid(half_width=1.0, image_h=h, sigma=sigma)
    tri = build_level0(f, mu, grid)
    print(f"\nhost, n={n}, sigma={sigma}, order={order}, one family")
    print(f"{'level':>6} {'entries':>12} {'traffic MB':>11} {'seconds':>9} "
          f"{'GB/s':>7}")
    tot_b = tot_t = 0.0
    for k in range(len(grid.levels) - 1):
        c = refine_counts(grid, k, order)
        t0 = time.perf_counter()
        tri = refine(tri, grid, k, order=order)
        dt = time.perf_counter() - t0
        b = c.bytes_moved()
        tot_b += b
        tot_t += dt
        print(f"{k:6d} {int(c.points):12,d} {b/1e6:11.1f} {dt:9.4f} "
              f"{b/dt/GB:7.1f}")
    print(f"{'all':>6} {'':12} {tot_b/1e6:11.1f} {tot_t:9.4f} "
          f"{tot_b/tot_t/GB:7.1f}")
    base = base_counts(grid)
    print(f"base level: {int(base.points):,} entries, "
          f"{base.total_flops()/1e9:.3f} Gflop, "
          f"{base.bytes_moved()/1e6:.1f} MB")
    return tot_b / tot_t / GB


def fp32_check(grids, sigma, order):
    print(f"\nsingle vs double precision, sigma={sigma}, order={order}")
    print(f"{'n':>5} {'levels':>7} {'max |fp32-fp64|':>17} {'relative':>10}")
    for n in grids:
        f, mu, h = _images(n)
        a = forward(f, mu, sigma=sigma, order=order)
        b = forward(f, mu, sigma=sigma, order=order, xp=SinglePrecision())
        dev = rel = 0.0
        for name in a:
            x = a[name].integrals().I
            y = b[name].integrals().I
            scale = float(np.abs(x).max())
            d = float(np.abs(x - y).max())
            dev, rel = max(dev, d), max(rel, d / scale)
        levels = len(build_grid(half_width=1.0, image_h=h,
                                sigma=sigma).levels) - 1
        print(f"{n:5d} {levels:7d} {dev:17.3e} {rel:10.2e}")
    print("fp32 roundoff is ~1e-7 relative and does not grow with the level "
          "count: the merge damps, so the recursion is not error-amplifying.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", type=int, nargs="+", default=[65, 129, 257, 513])
    ap.add_argument("--grids3d", type=int, nargs="+", default=[17, 33, 65])
    ap.add_argument("--sigma", type=int, default=2)
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--host-grid", type=int, default=257)
    ap.add_argument("--counts", action="store_true")
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()
    everything = not (args.counts or args.host or args.fp32)
    if args.counts or everything:
        counts_table(args.grids, args.sigma, args.order)
        counts_table3d(args.grids3d, args.sigma, args.order)
    if args.host or everything:
        host_bandwidth(args.host_grid, args.sigma, args.order)
    if args.fp32 or everything:
        fp32_check([n for n in args.grids if n <= 257], args.sigma, args.order)


if __name__ == "__main__":
    main()
