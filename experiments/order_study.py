"""Work against accuracy: how the cost scales with log(1/epsilon).

The stencil widths are the parameters that trade work for accuracy at a fixed
grid.  With an interpolation of order p the error falls geometrically,
eps ~ c^p, so p ~ log(1/eps); the work per interpolated point is q*p_s*p_t.
Raising all three together therefore tests directly which power of log(1/eps)
the schedule pays.

The sampling parameters are set high (n_theta_min, sigma) so that the stencil
order, not the sampling, is the binding constraint.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from tomogrid import Options, forward, phantoms as ph
from tomogrid.cost import estimate_work
from tomogrid.image import sample_function, square_grid

OUT = Path(__file__).resolve().parent / "results"
F = ph.ACTIVITIES["three_blobs"][0]
MU = ph.ATTENUATIONS["high_contrast"][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-grid", type=int, default=129)
    ap.add_argument("--n-theta-min", type=int, default=32)
    ap.add_argument("--s-oversample", type=int, default=4)
    ap.add_argument("--orders", type=int, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--sampling-only", action="store_true")
    args = ap.parse_args()

    n = args.n_grid
    if args.sampling_only:
        print("=== sampling sweep ===", flush=True)
        rows = sampling_sweep(n, [(4, 1), (8, 2), (16, 4), (32, 8)])
        OUT.mkdir(exist_ok=True)
        path = OUT / "sampling.csv"
        with open(path, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader(); wr.writerows(rows)
        print(f"wrote {path}")
        e = np.array([r["e_alg"] for r in rows])
        W = np.array([r["flops"] for r in rows], float)
        print(f"\nfitted  W ~ eps^{np.polyfit(np.log(e), np.log(W), 1)[0]:.2f}")
        print(f"fitted  W ~ (log 1/eps)^"
              f"{np.polyfit(np.log(np.log(1/e)), np.log(W), 1)[0]:.2f}")
        return
    n_theta, n_s = n - 1, n
    xa, ya = square_grid(n)
    f, mu = sample_function(F, xa, ya), sample_function(MU, xa, ya)
    kw = dict(n_theta=n_theta, n_s=n_s, support_radius=0.6)

    # One direct reference: with n_theta_min = n_theta nothing is interpolated,
    # so the stencil widths do not enter and it is the same for every order.
    base = dict(s_oversample=args.s_oversample)
    print("direct reference ...", flush=True)
    t0 = time.perf_counter()
    ref = forward(f, mu, options=Options(n_theta_min=n_theta, **base), **kw)
    print(f"   {time.perf_counter() - t0:.0f}s", flush=True)
    scale = float(np.abs(ref.S).max())

    rows = []
    for p in args.orders:
        opt = Options(n_theta_min=args.n_theta_min, theta_order=p,
                      s_order=p, t_order=p, **base)
        t0 = time.perf_counter()
        sg = forward(f, mu, options=opt, **kw)
        secs = time.perf_counter() - t0
        w = estimate_work(sg.hierarchy, gl_order=opt.gl_order,
                          img_order=opt.img_order, theta_order=p,
                          s_order=p, t_order=p)
        e = float(np.abs(sg.I - ref.I).max() / scale)
        rows.append(dict(order=p, e_alg=e, flops=w["flops"],
                         interp_gathers=w["interp_gathers"], seconds=secs,
                         n_grid=n, n_theta_min=args.n_theta_min,
                         s_oversample=args.s_oversample))
        print(f"   p={p}: e_alg={e:.3e}  {w['flops']/1e9:.2f} Gflop  {secs:.0f}s",
              flush=True)

    OUT.mkdir(exist_ok=True)
    path = OUT / "order.csv"
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {path}")

    e = np.array([r["e_alg"] for r in rows])
    W = np.array([r["flops"] for r in rows], dtype=float)
    ok = e > 0
    if ok.sum() >= 3:
        L = np.log(1.0 / e[ok])
        slope = np.polyfit(np.log(L), np.log(W[ok]), 1)[0]
        print(f"\nfitted  W ~ (log 1/eps)^{slope:.2f}")




def sampling_sweep(n_grid, pairs, order=4):
    """Accuracy against cost when the *sampling* parameters are raised together.

    The stencil order is fixed: on equispaced nodes a wider Lagrange stencil
    stops paying once the table spacing is not well inside the field's Nyquist
    rate, because the Lebesgue constant grows like 2^p.  What does keep buying
    accuracy is refining the tables, so this sweep raises n_theta_min and sigma
    in lockstep and records the achieved error against the work.
    """
    n_theta, n_s = n_grid - 1, n_grid
    xa, ya = square_grid(n_grid)
    f, mu = sample_function(F, xa, ya), sample_function(MU, xa, ya)
    rows = []
    for n_min, sigma in pairs:
        kw = dict(n_theta=n_theta, n_s=n_s, support_radius=0.6)
        common = dict(s_oversample=sigma, theta_order=order,
                      s_order=order, t_order=order)
        ref = forward(f, mu, options=Options(n_theta_min=n_theta, **common), **kw)
        t0 = time.perf_counter()
        sg = forward(f, mu, options=Options(n_theta_min=n_min, **common), **kw)
        secs = time.perf_counter() - t0
        w = estimate_work(sg.hierarchy, gl_order=8, img_order=4,
                          theta_order=order, s_order=order, t_order=order)
        e = float(np.abs(sg.I - ref.I).max() / np.abs(ref.S).max())
        rows.append(dict(n_theta_min=n_min, s_oversample=sigma, order=order,
                         e_alg=e, flops=w["flops"], seconds=secs, n_grid=n_grid))
        print(f"   n_theta_min={n_min:3d} sigma={sigma}: e_alg={e:.3e}  "
              f"{w['flops']/1e9:7.2f} Gflop  {secs:.0f}s", flush=True)
    return rows

if __name__ == "__main__":
    main()
