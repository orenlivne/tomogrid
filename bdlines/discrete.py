"""The discretised line integral of the paper's Figure 1, and the 1-D
interpolation along grid lines that defines it.

    "The data value at points where the integration line intersects a vertical
    grid line will be computed by interpolation from the data points on that
    line.  The integral will then be computed using the trapezoid rule."

The result is *normalised*: the mean of the (interpolated) data along the
segment, which is what the paper's ``F`` denotes and what makes the doubling
step a plain average.  Being a mean, it is the same whether the parameter is
arc length or horizontal extent.
"""

from __future__ import annotations

import numpy as np

KINDS = ("linear", "cubic", "spline")


def _spline_second_derivatives(values: np.ndarray) -> np.ndarray:
    """Natural cubic-spline second derivatives down axis 0, unit spacing."""
    n = values.shape[0]
    if n < 3:
        return np.zeros_like(values)
    rhs = 6.0 * (values[:-2] - 2.0 * values[1:-1] + values[2:])
    lower = np.full(n - 2, 1.0)
    diag = np.full(n - 2, 4.0)
    # Thomas algorithm on the constant tridiagonal (1, 4, 1).
    c = np.empty(n - 2)
    d = np.empty((n - 2,) + values.shape[1:])
    c[0] = 1.0 / diag[0]
    d[0] = rhs[0] / diag[0]
    for i in range(1, n - 2):
        denom = diag[i] - lower[i] * c[i - 1]
        c[i] = 1.0 / denom
        d[i] = (rhs[i] - lower[i] * d[i - 1]) / denom
    m = np.zeros_like(values)
    m[n - 2] = d[n - 3]
    for i in range(n - 4, -1, -1):
        m[i + 1] = d[i] - c[i] * m[i + 2]
    return m


def sample_columns(g: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                   kind: str = "cubic",
                   spline_m: np.ndarray | None = None) -> np.ndarray:
    """Interpolate ``g`` at fractional ``rows`` on the integer ``cols``.

    ``rows`` and ``cols`` broadcast together.  Out-of-range rows give zero, so
    the caller must keep segments inside the image.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    n_rows = g.shape[0]
    rows = np.asarray(rows, dtype=float)
    cols = np.asarray(cols)
    i0 = np.floor(rows).astype(np.int64)
    t = rows - i0

    if kind == "linear":
        out = np.zeros(np.broadcast(rows, cols).shape)
        for k, w in ((0, 1.0 - t), (1, t)):
            idx = i0 + k
            ok = (idx >= 0) & (idx < n_rows)
            out += np.where(ok, w * g[np.clip(idx, 0, n_rows - 1), cols], 0.0)
        return out

    if kind == "cubic":
        out = np.zeros(np.broadcast(rows, cols).shape)
        w = (
            -t * (t - 1) * (t - 2) / 6.0,
            (t + 1) * (t - 1) * (t - 2) / 2.0,
            -(t + 1) * t * (t - 2) / 2.0,
            (t + 1) * t * (t - 1) / 6.0,
        )
        for k in range(4):
            idx = i0 + k - 1
            ok = (idx >= 0) & (idx < n_rows)
            out += np.where(ok, w[k] * g[np.clip(idx, 0, n_rows - 1), cols], 0.0)
        return out

    m = _spline_second_derivatives(g) if spline_m is None else spline_m
    idx = np.clip(i0, 0, n_rows - 2)
    t = rows - idx
    a, b = 1.0 - t, t
    lo = g[idx, cols]
    hi = g[idx + 1, cols]
    mlo = m[idx, cols]
    mhi = m[idx + 1, cols]
    out = a * lo + b * hi + ((a**3 - a) * mlo + (b**3 - b) * mhi) / 6.0
    inside = (rows >= 0) & (rows <= n_rows - 1)
    return np.where(inside, out, 0.0)


def discrete_integral(g: np.ndarray, x0, y0, length: int, rise,
                      kind: str = "cubic",
                      spline_m: np.ndarray | None = None) -> np.ndarray:
    """Normalised discrete integral over the segment ``(x0,y0) -> (x0+L, y0+rise)``.

    Trapezoid over the ``length + 1`` vertical grid lines the segment crosses,
    with the data interpolated along each.  ``x0``, ``y0`` and ``rise`` broadcast.
    """
    x0 = np.asarray(x0)
    y0 = np.asarray(y0, dtype=float)
    rise = np.asarray(rise, dtype=float)
    if kind == "spline" and spline_m is None:
        spline_m = _spline_second_derivatives(g)
    u = np.arange(length + 1) / length
    shape = np.broadcast(x0, y0, rise).shape + (length + 1,)
    cols = np.broadcast_to(x0[..., None], shape) + np.arange(length + 1)
    rows = (np.broadcast_to(y0[..., None], shape)
            + np.broadcast_to(rise[..., None], shape) * u)
    vals = sample_columns(g, rows, cols, kind, spline_m)
    weights = np.full(length + 1, 1.0)
    weights[0] = weights[-1] = 0.5
    return (vals * weights).sum(axis=-1) / length
