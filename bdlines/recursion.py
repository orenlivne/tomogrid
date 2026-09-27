"""Base level and length doubling (paper, section 3).

Tables are ``(n_x, n_y, n_slope)`` arrays of *normalised* integrals: entry
``[i, j, m]`` is the mean of the data along the segment from
``(x_i, y_j)`` to ``(x_i + L, y_j + (m - rise) h)``.

Doubling, for a target with rise ``M`` over length ``2L``:

* ``M`` even -- the direction exists at level ``L``; the two halves are the
  level-``L`` entries ``[2i, j, M/2]`` and ``[2i+1, j + M/2, M/2]``, the second
  starting exactly where the first ends.  **Exact.**
* ``M`` odd -- interpolate the *slope index* over ``p`` neighbours, separately
  for each half, at the same starting point; the second half's start row moves
  with the slope index, which is what removes any need for spatial
  interpolation.

Then average the two halves.  Cost is ``O(p)`` per integral per level.
"""

from __future__ import annotations

import numpy as np

from .discrete import _spline_second_derivatives, discrete_integral
from .mesh import BDLevel, BDMesh


def base_level(g: np.ndarray, mesh: BDMesh, kind: str = "cubic") -> np.ndarray:
    """Level-0 table, the only stage that touches the data."""
    lv = mesh.levels[0]
    spline_m = _spline_second_derivatives(g) if kind == "spline" else None
    x0 = mesh.x_nodes(lv)[:, None, None].astype(np.int64)
    y0 = mesh.y_nodes()[None, :, None]
    rise = (np.arange(-lv.rise, lv.rise + 1) * mesh.h)[None, None, :]
    return discrete_integral(g, x0, y0, lv.length, rise, kind, spline_m)


def _shear(table: np.ndarray) -> np.ndarray:
    """``G[i, j, m] = table[i, j + (m - rise), m]``, zero outside the mesh.

    Where the second half of each segment is read: it begins where the first
    ended, so its row index moves with the slope index.  An integer shear.
    """
    n_y, n_m = table.shape[1], table.shape[2]
    shift = np.arange(n_m) - (n_m - 1) // 2
    rows = np.arange(n_y)[:, None] + shift[None, :]
    ok = (rows >= 0) & (rows < n_y)
    cols = np.arange(n_m)[None, :]
    out = table[:, np.clip(rows, 0, n_y - 1), cols]
    return np.where(ok[None, :, :], out, 0.0)


def _odd_stencil(n_odd: int, n_in: int, order: int):
    """Slope-index stencil for the half-rises ``m + 1/2``.

    The stencil slides inside the slope range at the ends, so it is one-sided
    there but never extrapolates: the targets are midpoints of the input grid.
    """
    q = min(order, n_in)
    m_idx = np.arange(n_odd)
    base = np.clip(m_idx - (q // 2 - 1), 0, n_in - q)
    idx = base[None, :] + np.arange(q)[:, None]
    u = (m_idx + 0.5) - base
    w = np.empty((q, n_odd))
    for k in range(q):
        col = np.ones(n_odd)
        for j in range(q):
            if j != k:
                col = col * (u - j) / (k - j)
        w[k] = col
    return idx, w


def double(table: np.ndarray, mesh: BDMesh, k: int, order: int = 2) -> np.ndarray:
    """One length doubling, level ``k`` -> ``k+1``.  ``order`` is the paper's ``p``."""
    lv, nxt = mesh.levels[k], mesh.levels[k + 1]
    if table.shape != (lv.n_x, mesh.n_y, lv.n_slope):
        raise ValueError(f"table {table.shape} != "
                         f"{(lv.n_x, mesh.n_y, lv.n_slope)}")
    sheared = _shear(table)
    lo = slice(0, 2 * nxt.n_x, 2)
    hi = slice(1, 2 * nxt.n_x, 2)
    out = np.empty((nxt.n_x, mesh.n_y, nxt.n_slope))

    # Even targets: the direction already exists, so this is exact.
    out[:, :, 0::2] = 0.5 * (table[lo] + sheared[hi])

    # Odd targets: interpolate the slope index of each half, then average.
    n_odd = nxt.n_slope - lv.n_slope
    idx, w = _odd_stencil(n_odd, lv.n_slope, order)
    first = np.zeros((nxt.n_x, mesh.n_y, n_odd))
    second = np.zeros_like(first)
    for a in range(idx.shape[0]):
        wa = w[a][None, None, :]
        first += wa * table[lo][:, :, idx[a]]
        second += wa * sheared[hi][:, :, idx[a]]
    out[:, :, 1::2] = 0.5 * (first + second)
    return out


def run_recursion(g: np.ndarray, mesh: BDMesh, *, disc_kind: str = "cubic",
                  order: int = 2, keep_all: bool = False):
    """Base level then every doubling.  Returns the top table, or all of them."""
    table = base_level(g, mesh, disc_kind)
    tables = [table]
    for k in range(len(mesh.levels) - 1):
        table = double(table, mesh, k, order)
        if keep_all:
            tables.append(table)
    return tables if keep_all else table
