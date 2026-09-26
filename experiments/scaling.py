"""Scaling study: accuracy and operation count under uniform grid refinement.

For a sequence of uniform grids the script measures, on the same phantom,

    e_alg   multilevel vs the direct method on the *same* discretisation
            (this is the error the multilevel schedule introduces)
    e_disc  direct vs the continuum
            (the error of the discretisation alone)
    e_tot   multilevel vs the continuum

together with wall time and the modelled operation count for both methods.
The continuum reference is a dense trapezoidal evaluation of the *analytic*
phantom; on the C-infinity compactly supported fields used here that rule is
spectrally accurate, and the script verifies this by refining it.

    python -m experiments.scaling --quick
    python -m experiments.scaling
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from tomogrid import Options, forward, phantoms as ph
from tomogrid.axis import Axis
from tomogrid.cost import estimate_work
from tomogrid.hierarchy import build_hierarchy
from tomogrid.image import sample_function, square_grid
from tomogrid.multilevel import default_n_levels
from tomogrid.reference import naive_transform

OUT = Path(__file__).resolve().parent / "results"
N_THETA_MIN = 16
PHANTOM = dict(
    f=ph.ACTIVITIES["three_blobs"][0],
    mu=ph.ATTENUATIONS["high_contrast"][0],
    support=0.6,
)


def sup_err(got, ref, scale):
    return float(np.abs(np.asarray(got) - np.asarray(ref)).max() / scale)


def continuum(thetas, s_nodes, n_samples):
    budget = 1 << 22
    chunk = max(1, min(8, budget // max(1, s_nodes.size * n_samples)))
    return naive_transform(
        PHANTOM["f"], PHANTOM["mu"], thetas, s_nodes,
        half_chord=1.0, n_samples=n_samples, theta_chunk=chunk,
    )


def run_once(n_grid, n_theta_min, opts_extra=None):
    xa, ya = square_grid(n_grid)
    f = sample_function(PHANTOM["f"], xa, ya)
    mu = sample_function(PHANTOM["mu"], xa, ya)
    opt = Options(n_theta_min=n_theta_min, **(opts_extra or {}))
    t0 = time.perf_counter()
    sg = forward(f, mu, n_theta=n_grid - 1, n_s=n_grid, options=opt,
                 support_radius=PHANTOM["support"])
    return sg, time.perf_counter() - t0, estimate_work(
        sg.hierarchy, gl_order=opt.gl_order, img_order=opt.img_order,
        theta_order=opt.theta_order, s_order=opt.s_order, t_order=opt.t_order,
    )


def refinement_study(grids, direct_max, n_samples):
    rows = []
    for n_grid in grids:
        n_theta = n_grid - 1
        print(f"[grid {n_grid}] n_theta={n_theta} n_s={n_grid}", flush=True)

        ml, t_ml, w_ml = run_once(n_grid, N_THETA_MIN)
        print(f"   multilevel {t_ml:8.2f} s  {w_ml['flops']/1e9:7.2f} Gflop", flush=True)

        ref = continuum(ml.thetas, ml.s_axis.nodes, n_samples)
        scale = float(np.abs(ref[0]).max())
        row = dict(
            n_grid=n_grid, h=2.0 / (n_grid - 1), n_theta=n_theta, n_s=n_grid,
            n_levels=default_n_levels(2.0 / (n_grid - 1), 1.0),
            t_ml=t_ml, flops_ml=w_ml["flops"], gathers_ml=w_ml["gathers"],
            e_tot=sup_err(ml.I, ref[2], scale),
            e_tot_S=sup_err(ml.S, ref[0], scale),
        )

        if n_grid <= direct_max:
            dr, t_dir, w_dir = run_once(n_grid, n_theta)
            print(f"   direct     {t_dir:8.2f} s  {w_dir['flops']/1e9:7.2f} Gflop", flush=True)
            row.update(
                t_direct=t_dir, flops_direct=w_dir["flops"],
                gathers_direct=w_dir["gathers"],
                e_alg=sup_err(ml.I, dr.I, float(np.abs(dr.S).max())),
                e_disc=sup_err(dr.I, ref[2], scale),
            )
        else:
            n_levels = default_n_levels(2.0 / (n_grid - 1), 1.0)
            s_axis = Axis.from_endpoints(-1.0, 1.0, 2 * (n_grid - 1) + 1)
            hier = build_hierarchy(n_theta_out=n_theta, n_theta_min=n_theta,
                                   n_levels=n_levels, s_axis=s_axis, t_oversample=2)
            w = estimate_work(hier, gl_order=8, img_order=4, theta_order=4,
                              s_order=4, t_order=4)
            row.update(t_direct=float("nan"), flops_direct=w["flops"],
                       gathers_direct=w["gathers"],
                       e_alg=float("nan"), e_disc=float("nan"))
        print(f"   e_tot={row['e_tot']:.3e}  e_alg={row.get('e_alg', float('nan')):.3e}"
              f"  e_disc={row.get('e_disc', float('nan')):.3e}", flush=True)
        rows.append(row)
    return rows


def knob_study(n_grid, n_samples):
    """Accuracy vs the two knobs at fixed resolution."""
    n_theta = n_grid - 1
    direct, _, _ = run_once(n_grid, n_theta)
    scale = float(np.abs(direct.S).max())
    rows = []
    for name, kwargs_list in (
        ("n_theta_min", [dict(n_theta_min=v) for v in (4, 8, 16, 32, 64)]),
        ("s_oversample", [dict(n_theta_min=N_THETA_MIN, s_oversample=v) for v in (1, 2, 4)]),
        ("t_oversample", [dict(n_theta_min=N_THETA_MIN, t_oversample=v) for v in (2, 4, 8)]),
        ("theta_order", [dict(n_theta_min=N_THETA_MIN, theta_order=v) for v in (2, 4, 6)]),
        ("s_order", [dict(n_theta_min=N_THETA_MIN, s_order=v) for v in (2, 4, 6)]),
        ("t_order", [dict(n_theta_min=N_THETA_MIN, t_order=v) for v in (2, 4, 6)]),
    ):
        for kw in kwargs_list:
            nmin = kw.pop("n_theta_min")
            sg, t, w = run_once(n_grid, nmin, kw)
            value = nmin if name == "n_theta_min" else kw[name]
            rows.append(dict(
                knob=name, value=value, n_grid=n_grid,
                e_alg=sup_err(sg.I, direct.I, scale),
                flops=w["flops"], seconds=t,
                dtheta_l_over_h=sg.hierarchy.angular_criterion(2.0 / (n_grid - 1)),
            ))
            print(f"   {name}={value}: e_alg={rows[-1]['e_alg']:.3e} "
                  f"{w['flops']/1e9:.2f} Gflop  {t:.1f}s", flush=True)
    return rows


def model_extrapolation(grids):
    rows = []
    for n_grid in grids:
        n_theta = n_grid - 1
        n_levels = default_n_levels(2.0 / (n_grid - 1), 1.0)
        s_axis = Axis.from_endpoints(-1.0, 1.0, 2 * (n_grid - 1) + 1)
        out = {"n_grid": n_grid, "N": n_grid**2}
        for tag, nmin in (("ml", N_THETA_MIN), ("direct", n_theta)):
            hier = build_hierarchy(n_theta_out=n_theta, n_theta_min=nmin,
                                   n_levels=n_levels, s_axis=s_axis, t_oversample=2)
            out[f"flops_{tag}"] = estimate_work(
                hier, gl_order=8, img_order=4, theta_order=4, s_order=4, t_order=4
            )["flops"]
        out["speedup"] = out["flops_direct"] / out["flops_ml"]
        rows.append(out)
    return rows


def verify_reference(n_grid, n_samples):
    """The continuum reference must be converged: refining it changes nothing."""
    xa, ya = square_grid(n_grid)
    thetas = 2 * np.pi * np.arange(n_grid - 1) / (n_grid - 1)
    s_nodes = np.linspace(-1.0, 1.0, n_grid)
    a = continuum(thetas, s_nodes, n_samples)
    b = continuum(thetas, s_nodes, 2 * n_samples - 1)
    scale = float(np.abs(b[0]).max())
    return sup_err(a[2], b[2], scale)


def write_csv(path, rows):
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--knobs-only", action="store_true")
    ap.add_argument("--skip-knobs", action="store_true",
                    help="leave results/knobs.csv untouched")
    ap.add_argument("--grids", type=int, nargs="+",
                    help="override the refinement grid sequence")
    ap.add_argument("--direct-max", type=int,
                    help="largest grid on which to run direct evaluation; above "
                         "this the operation count comes from the schedule")
    args = ap.parse_args()

    if args.quick:
        grids, direct_max, n_samples, knob_n = [33, 65], 65, 2049, 33
        model_grids = [65, 129, 257, 513]
    else:
        grids, direct_max, n_samples, knob_n = [33, 65, 129, 257, 513], 257, 4097, 129
        model_grids = [65, 129, 257, 513, 1025, 2049, 4097]
    if args.grids:
        grids = args.grids
    if args.direct_max is not None:
        direct_max = args.direct_max

    OUT.mkdir(exist_ok=True)
    t0 = time.perf_counter()

    if args.knobs_only:
        write_csv(OUT / "knobs.csv", knob_study(knob_n, n_samples))
        print(f"\ntotal {time.perf_counter() - t0:.0f}s")
        return

    drift = verify_reference(grids[0], n_samples)
    print(f"continuum reference self-consistency: {drift:.2e}\n", flush=True)

    print("=== refinement study ===", flush=True)
    ref_rows = refinement_study(grids, direct_max, n_samples)
    write_csv(OUT / "refinement.csv", ref_rows)

    if not args.skip_knobs:
        print("\n=== knob study ===", flush=True)
        write_csv(OUT / "knobs.csv", knob_study(knob_n, n_samples))

    print("\n=== cost model extrapolation ===", flush=True)
    model_rows = model_extrapolation(model_grids)
    for r in model_rows:
        print(f"   n={r['n_grid']:5d}  ml {r['flops_ml']/1e9:10.2f} G   "
              f"direct {r['flops_direct']/1e9:12.2f} G   speedup {r['speedup']:8.1f}x")
    write_csv(OUT / "model.csv", model_rows)

    (OUT / "meta.json").write_text(json.dumps(
        dict(n_theta_min=N_THETA_MIN, n_samples=n_samples,
             reference_drift=drift, seconds=time.perf_counter() - t0,
             phantom="three_blobs x high_contrast"), indent=2))
    print(f"\ntotal {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
