"""Tensor-product Lagrange interpolation with exact zero-fill outside the grid.

The only interpolation used anywhere in the solver.  ``order`` is the number of
stencil points, so ``order=4`` is cubic.  Out-of-range stencil entries are given
weight zero, i.e. the field is treated as identically zero outside the grid.
That is *exact* for the tables in this code (see docs/formulation.md, sec. 4),
and it is the caller's job to guarantee it; :func:`support_is_safe` helps.
"""

from __future__ import annotations

import numpy as np

from .axis import Axis


def lagrange_weights(u: np.ndarray, order: int) -> np.ndarray:
    """Lagrange basis weights at local coordinate ``u`` for nodes ``0..order-1``.

    Returns an array of shape ``(order,) + u.shape``.
    """
    u = np.asarray(u, dtype=float)
    w = np.ones((order,) + u.shape, dtype=float)
    for k in range(order):
        for m in range(order):
            if m != k:
                w[k] *= (u - m) / (k - m)
    return w


def stencil(x: np.ndarray, axis: Axis, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Zero-filled interpolation stencil for targets ``x`` on ``axis``.

    Returns ``(idx, w)`` each of shape ``(order,) + x.shape``; the interpolant is
    ``sum_k w[k] * values[idx[k]]``.  Indices are clamped into range and the
    corresponding weights zeroed, so out-of-range contributions vanish.
    """
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    x = np.asarray(x, dtype=float)
    tau = (x - axis.origin) / axis.h
    # Left-most stencil node, chosen so the stencil straddles x.
    i0 = np.floor(tau).astype(np.int64) - (order // 2 - 1 if order > 1 else 0)
    if axis.n < order:
        # Degenerate axis: fall back to what fits, anchored at 0.
        i0 = np.zeros_like(i0)
    u = tau - i0
    w = lagrange_weights(u, order)
    k = np.arange(order).reshape((order,) + (1,) * x.ndim)
    idx = i0[None, ...] + k
    oob = (idx < 0) | (idx >= axis.n)
    w = np.where(oob, 0.0, w)
    idx = np.clip(idx, 0, axis.n - 1)
    return idx, w


def interp1d(values: np.ndarray, axis: Axis, x: np.ndarray, order: int) -> np.ndarray:
    idx, w = stencil(x, axis, order)
    return np.einsum("k...,k...->...", w, values[idx])


def interp2d(
    values: np.ndarray,
    axis0: Axis,
    axis1: Axis,
    x0: np.ndarray,
    x1: np.ndarray,
    order0: int,
    order1: int,
) -> np.ndarray:
    """Interpolate a 2-D table ``values[i0, i1]`` at scattered ``(x0, x1)``."""
    if values.shape != (axis0.n, axis1.n):
        raise ValueError(f"values shape {values.shape} != ({axis0.n}, {axis1.n})")
    i0, w0 = stencil(x0, axis0, order0)
    i1, w1 = stencil(x1, axis1, order1)
    out = np.zeros(np.broadcast(x0, x1).shape, dtype=float)
    for a in range(order0):
        for b in range(order1):
            out += w0[a] * w1[b] * values[i0[a], i1[b]]
    return out


def interp2d_stacked(
    values: np.ndarray,
    axis0: Axis,
    axis1: Axis,
    lead: np.ndarray,
    x0: np.ndarray,
    x1: np.ndarray,
    order0: int,
    order1: int,
) -> np.ndarray:
    """Like :func:`interp2d` but ``values`` has a leading index selected per target.

    ``values`` has shape ``(n_lead, axis0.n, axis1.n)`` and ``lead`` broadcasts
    against ``x0``/``x1`` selecting which slab each target reads from.
    """
    i0, w0 = stencil(x0, axis0, order0)
    i1, w1 = stencil(x1, axis1, order1)
    out = np.zeros(np.broadcast(x0, x1).shape, dtype=float)
    for a in range(order0):
        for b in range(order1):
            out += w0[a] * w1[b] * values[lead, i0[a], i1[b]]
    return out


def periodic_midpoint_weights(order: int) -> np.ndarray:
    """Lagrange weights for the midpoint of a uniform grid, ``order`` points wide.

    The stencil is ``j - order//2 + 1 ... j + order//2`` and the target sits at
    ``j + 1/2``, i.e. local coordinate ``order/2 - 1/2``.
    """
    return lagrange_weights(np.asarray(order / 2 - 0.5), order)


def support_is_safe(support_radius: float, table_half_extent: float, order: int, h: float) -> bool:
    """Whether zero-fill can never clip a nonzero value.

    The outermost ``order//2`` nodes of a table are only ever read as part of a
    stencil, so the table must extend at least that far beyond the support.
    """
    return table_half_extent >= support_radius + (order // 2) * h
