"""Reproduction of the numerical experiments of Brandt & Dym, section 4.

Table 1: relative approximation error ``|(I_a - I_d)/I_d|``, average and worst
case over all directions ``0 <= theta <= pi/4``, for sinusoidal data
``g(x,y) = 1 + sin(y/w)`` at three integration lengths and three integration
meshes, with the discretisation by cubic spline and the doubling by linear
interpolation.  The published table also carries the discretisation error
``|(I_t - I_d)/I_t|`` for comparison.

Table 2: the same at ``L = 128``, ``w = 2``, crossing three discretisation
interpolants with two doubling interpolants.
"""

from __future__ import annotations

import numpy as np

from .discrete import _spline_second_derivatives, discrete_integral
from .mesh import build_mesh
from .recursion import run_recursion
from .truth import true_integral

W_VALUES = (4, 2, 1)
LENGTHS = (8, 32, 128)
SIGMAS = (1, 2, 4)


def sinusoid(w):
    """The paper's target: constant along columns, sinusoidal down rows."""
    return lambda x, y: 1.0 + np.sin(np.asarray(y) / w)


def sample(w, n_rows, n_cols):
    col = 1.0 + np.sin(np.arange(n_rows) / w)
    return np.tile(col[:, None], (1, n_cols))


def _errors(approx, exact):
    rel = np.abs((approx - exact) / exact)
    return rel.mean(), rel.max()


def run_case(w, sigma, *, n_rows=257, n_cols=257, max_length=128,
             disc_kind="spline", order=2, x0=0, truth_nodes=200):
    """One (w, sigma) cell: approximation and discretisation errors per length."""
    g = sample(w, n_rows, n_cols)
    fun = sinusoid(w)
    mesh = build_mesh(n_rows, n_cols, sigma=sigma, max_length=max_length)
    tables = run_recursion(g, mesh, disc_kind=disc_kind, order=order,
                           keep_all=True)
    spline_m = _spline_second_derivatives(g) if disc_kind == "spline" else None

    y0 = (n_rows - 1) / 2.0
    jy = int(round(y0 / mesh.h))
    out = {}
    for lv, table in zip(mesh.levels, tables):
        if lv.length not in LENGTHS:
            continue
        m = np.arange(0, lv.rise + 1)  # slopes 0 .. 1, i.e. 0 <= theta <= pi/4
        rise = m * mesh.h
        approx = table[x0 // lv.length, jy, m + lv.rise]
        disc = discrete_integral(g, np.asarray(x0), np.asarray(y0), lv.length,
                                 rise, disc_kind, spline_m)
        truth = true_integral(fun, x0, y0, lv.length, rise, truth_nodes)
        out[lv.length] = {
            "approx_mean": _errors(approx, disc)[0],
            "approx_worst": _errors(approx, disc)[1],
            "disc_mean": _errors(disc, truth)[0],
            "n_dir": m.size,
        }
    return out


def table1(**kw):
    rows = []
    for w in W_VALUES:
        per_sigma = {s: run_case(w, s, **kw) for s in SIGMAS}
        for L in LENGTHS:
            rows.append({
                "w": w, "L": L,
                "disc": per_sigma[SIGMAS[0]][L]["disc_mean"],
                **{f"sigma{s}_mean": per_sigma[s][L]["approx_mean"]
                   for s in SIGMAS},
                **{f"sigma{s}_worst": per_sigma[s][L]["approx_worst"]
                   for s in SIGMAS},
            })
    return rows


def table2(**kw):
    rows = []
    for disc_kind in ("linear", "cubic", "spline"):
        for order, name in ((2, "linear"), (4, "cubic")):
            per_sigma = {s: run_case(2, s, disc_kind=disc_kind, order=order, **kw)
                         for s in SIGMAS}
            rows.append({
                "disc_kind": disc_kind, "doubling": name,
                "disc": per_sigma[SIGMAS[0]][128]["disc_mean"],
                **{f"sigma{s}_mean": per_sigma[s][128]["approx_mean"]
                   for s in SIGMAS},
                **{f"sigma{s}_worst": per_sigma[s][128]["approx_worst"]
                   for s in SIGMAS},
            })
    return rows


def _pct(x):
    return f"{100 * x:.3g}%"


def print_table1(rows):
    print(f"{'w':>2} {'L':>4} {'disc err':>10} "
          + " ".join(f"{'H/h=' + str(s):>18}" for s in SIGMAS))
    for r in rows:
        cells = " ".join(
            f"{_pct(r[f'sigma{s}_mean']) + ' (' + _pct(r[f'sigma{s}_worst']) + ')':>18}"
            for s in SIGMAS)
        print(f"{r['w']:>2} {r['L']:>4} {_pct(r['disc']):>10} {cells}")


def print_table2(rows):
    print(f"{'disc':>8} {'doubling':>9} {'disc err':>10} "
          + " ".join(f"{'H/h=' + str(s):>18}" for s in SIGMAS))
    for r in rows:
        cells = " ".join(
            f"{_pct(r[f'sigma{s}_mean']) + ' (' + _pct(r[f'sigma{s}_worst']) + ')':>18}"
            for s in SIGMAS)
        print(f"{r['disc_kind']:>8} {r['doubling']:>9} {_pct(r['disc']):>10} {cells}")


if __name__ == "__main__":
    import sys

    which = sys.argv[1] if len(sys.argv) > 1 else "1"
    if which == "1":
        print("Table 1 (cf. Brandt & Dym p. 1425)\n")
        print_table1(table1())
    else:
        print("Table 2 (cf. Brandt & Dym p. 1426)\n")
        print_table2(table2())
