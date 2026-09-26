"""Three-dimensional scaling study: accuracy and operation count.

Sizes are much smaller than in 2-D: the tables are four-dimensional and the
frame-rotation resample costs ``q^2 p_s^2 p_t`` per target, so a single run at
33^3 already takes tens of minutes.  The cost model is exercised well beyond
what is executed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from tomogrid import phantoms as ph
from tomogrid.axis import Axis
from tomogrid.image import cube_grid, sample_function3d
from tomogrid.multilevel3d import Options3D, build_hierarchy3d, forward3d
from tomogrid.reference3d import naive_transform3d

OUT = Path(__file__).resolve().parent / "results"
F = ph.ACTIVITIES_3D["three_blobs"][0]
MU = ph.ATTENUATIONS_3D["high_contrast"][0]
SUPPORT = 0.6


def work3d(hier, opt: Options3D) -> dict:
    """Operation count for the 3-D schedule (mirrors tomogrid.cost for 2-D)."""
    n_s = hier.s_axis.n
    l0 = hier.levels[0]
    level0_nodes = l0.dirs.n_dir * n_s * n_s * l0.t_axis.n * opt.gl_order
    table_points = sum(l.dirs.n_dir * n_s * n_s * l.t_axis.n for l in hier.levels)
    gathers = 2 * level0_nodes * opt.img_order**3
    for i in hier.refine_levels:
        prev, cur = hier.levels[i - 1], hier.levels[i]
        q = min(opt.dir_order, prev.dirs.m)
        fresh = cur.dirs.n_dir - prev.dirs.n_dir
        targets = 3 * fresh * n_s * n_s * prev.t_axis.n
        gathers += targets * q * q * opt.s_order**2 * opt.t_order
    level0_quad = level0_nodes * opt.gl_order
    return dict(level0_nodes=level0_nodes, table_points=table_points,
                gathers=gathers, flops=2 * (gathers + level0_quad))


def hierarchy_for(n_grid, m_out, m_min, s_axis, opt):
    h = 2.0 / (n_grid - 1)
    n_levels = max(0, int(math.ceil(math.log2(2.0 * opt.half_chord / h))))
    return build_hierarchy3d(m_out=m_out, m_min=min(m_min, m_out),
                             n_levels=n_levels, s_axis=s_axis,
                             half_chord=opt.half_chord,
                             t_oversample=opt.t_oversample)


def run_once(n_grid, m_out, m_min, s_oversample, gl_order=6):
    xa, ya, za = cube_grid(n_grid)
    f = sample_function3d(F, xa, ya, za)
    mu = sample_function3d(MU, xa, ya, za)
    n_s = s_oversample * (n_grid - 1) + 1
    opt = Options3D(m_min=m_min, gl_order=gl_order)
    t0 = time.perf_counter()
    sg = forward3d(f, mu, m_out=m_out, n_s=n_s, options=opt,
                   support_radius=SUPPORT)
    return sg, time.perf_counter() - t0, work3d(sg.hierarchy, opt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", type=int, nargs="+", default=[9, 17, 33])
    ap.add_argument("--direct-max", type=int, default=17)
    ap.add_argument("--m-out", type=int, default=9)
    ap.add_argument("--m-min", type=int, default=5)
    ap.add_argument("--s-oversample", type=int, default=2)
    ap.add_argument("--n-samples", type=int, default=2049)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    rows = []
    t_start = time.perf_counter()
    for n in args.grids:
        print(f"[grid {n}^3] m_out={args.m_out} m_min={args.m_min} "
              f"sigma={args.s_oversample}", flush=True)
        ml, t_ml, w_ml = run_once(n, args.m_out, args.m_min, args.s_oversample)
        print(f"   multilevel {t_ml:8.1f} s  {w_ml['flops']/1e9:8.2f} Gflop",
              flush=True)
        _, _, I_ref = naive_transform3d(F, MU, ml.dirs, ml.s_axis.nodes, 1.0,
                                        n_samples=args.n_samples, dir_chunk=4)
        S_ref, _, _ = naive_transform3d(F, ph.zero3d(), ml.dirs, ml.s_axis.nodes,
                                        1.0, n_samples=args.n_samples, dir_chunk=4)
        scale = float(np.abs(S_ref).max())
        row = dict(n_grid=n, h=2.0 / (n - 1), n_dir=ml.dirs.n_dir,
                   n_s=ml.s_axis.n, t_ml=t_ml, flops_ml=w_ml["flops"],
                   e_tot=float(np.abs(ml.I - I_ref).max() / scale))
        if n <= args.direct_max:
            dr, t_dir, w_dir = run_once(n, args.m_out, args.m_out, args.s_oversample)
            print(f"   direct     {t_dir:8.1f} s  {w_dir['flops']/1e9:8.2f} Gflop",
                  flush=True)
            row.update(t_direct=t_dir, flops_direct=w_dir["flops"],
                       e_alg=float(np.abs(ml.I - dr.I).max()
                                   / float(np.abs(dr.S).max())),
                       e_disc=float(np.abs(dr.I - I_ref).max() / scale))
        else:
            s_axis = Axis.from_endpoints(-1.0, 1.0,
                                         args.s_oversample * (n - 1) + 1)
            opt = Options3D(m_min=args.m_out, gl_order=6)
            w = work3d(hierarchy_for(n, args.m_out, args.m_out, s_axis, opt), opt)
            row.update(t_direct=float("nan"), flops_direct=w["flops"],
                       e_alg=float("nan"), e_disc=float("nan"))
        print(f"   e_tot={row['e_tot']:.3e} e_alg={row.get('e_alg', float('nan')):.3e}"
              f" e_disc={row.get('e_disc', float('nan')):.3e}", flush=True)
        rows.append(row)

    path = OUT / "refinement3d.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {path}")

    # Cost model well beyond what can be executed.
    model = []
    for n in (17, 33, 65, 129, 257):
        s_axis = Axis.from_endpoints(-1.0, 1.0, args.s_oversample * (n - 1) + 1)
        m_out = args.m_out
        out = {"n_grid": n, "N": n**3, "n_dir": 6 * m_out * m_out}
        for tag, mm in (("ml", args.m_min), ("direct", m_out)):
            opt = Options3D(m_min=mm, gl_order=6)
            out[f"flops_{tag}"] = work3d(
                hierarchy_for(n, m_out, mm, s_axis, opt), opt)["flops"]
        out["speedup"] = out["flops_direct"] / out["flops_ml"]
        model.append(out)
        print(f"   n={n:4d}  ml {out['flops_ml']/1e9:10.2f} G  "
              f"direct {out['flops_direct']/1e9:12.2f} G  "
              f"speedup {out['speedup']:6.2f}x")
    path = OUT / "model3d.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(model[0]))
        w.writeheader(); w.writerows(model)
    print(f"wrote {path}")
    (OUT / "meta3d.json").write_text(json.dumps(
        dict(m_out=args.m_out, m_min=args.m_min, s_oversample=args.s_oversample,
             n_samples=args.n_samples, seconds=time.perf_counter() - t_start),
        indent=2))


if __name__ == "__main__":
    main()
