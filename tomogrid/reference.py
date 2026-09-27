"""Independent reference evaluators, for testing only.

:func:`naive_transform` shares *no* code with the solver: it samples each line on
a dense uniform grid and uses nothing but the trapezoidal rule (plus a reversed
cumulative sum for the attenuation).  Second order in the sample spacing, with
:func:`richardson_transform` extrapolating to fourth.  Slow and obviously
correct, which is the point.

:func:`analytic_gaussian` gives closed-form ground truth for Gaussian activity
with uniform attenuation, independent of any quadrature at all.
"""

from __future__ import annotations

import numpy as np

from .image import Image


def image_evaluator(img: Image, order: int = 4):
    """Adapt an :class:`Image` to the ``fun(x, y)`` interface, using the *same*
    off-grid rule as the solver so that discretisation error cancels."""

    def ev(x, y):
        return img.sample(x, y, order=order)

    return ev


def naive_transform(
    f_eval,
    mu_eval,
    thetas: np.ndarray,
    s_nodes: np.ndarray,
    half_chord: float = 1.0,
    n_samples: int = 2049,
    theta_chunk: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense-trapezoid evaluation of ``(S, E, I)``.  Returns arrays ``(n_th, n_s)``."""
    thetas = np.asarray(thetas, dtype=float)
    s_nodes = np.asarray(s_nodes, dtype=float)
    t = np.linspace(-half_chord, half_chord, n_samples)
    dt = t[1] - t[0]

    S = np.empty((thetas.size, s_nodes.size))
    E = np.empty_like(S)
    I = np.empty_like(S)

    for lo in range(0, thetas.size, theta_chunk):
        th = thetas[lo : lo + theta_chunk]
        c, sn = np.cos(th)[:, None, None], np.sin(th)[:, None, None]
        s = s_nodes[None, :, None]
        tt = t[None, None, :]
        x = s * (-sn) + tt * c
        y = s * c + tt * sn
        fv = np.asarray(f_eval(x, y), dtype=float) + np.zeros(x.shape)
        mv = np.asarray(mu_eval(x, y), dtype=float) + np.zeros(x.shape)

        # A[..., i] = int_{t_i}^{+half_chord} mu  (trapezoid, reversed cumsum)
        panel = 0.5 * (mv[..., :-1] + mv[..., 1:]) * dt
        tail = np.cumsum(panel[..., ::-1], axis=-1)[..., ::-1]
        A = np.concatenate([tail, np.zeros(tail.shape[:-1] + (1,))], axis=-1)

        def trapz(g):
            return dt * (g[..., 1:-1].sum(axis=-1) + 0.5 * (g[..., 0] + g[..., -1]))

        sl = slice(lo, lo + th.size)
        S[sl] = trapz(fv)
        E[sl] = A[..., 0]
        I[sl] = trapz(fv * np.exp(-A))
    return S, E, I


def richardson_transform(
    f_eval, mu_eval, thetas, s_nodes, half_chord=1.0, n_samples=1025, **kw
):
    """Richardson-extrapolated :func:`naive_transform` (fourth order).

    Uses ``n_samples`` and ``2*n_samples - 1`` points, both with the same
    endpoints, and combines as ``(4*fine - coarse)/3``.
    """
    coarse = naive_transform(f_eval, mu_eval, thetas, s_nodes, half_chord, n_samples, **kw)
    fine = naive_transform(
        f_eval, mu_eval, thetas, s_nodes, half_chord, 2 * n_samples - 1, **kw
    )
    return tuple((4.0 * fi - co) / 3.0 for co, fi in zip(coarse, fine))


def analytic_gaussian(
    thetas: np.ndarray,
    s_nodes: np.ndarray,
    *,
    amp: float,
    center: tuple[float, float],
    sigma: float,
    mu_const: float,
    half_chord: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed form for ``f`` Gaussian and ``mu`` constant on the whole plane.

    With ``c_perp = c.omega_perp`` and ``c_par = c.omega``, integrating over the
    *whole* line (the truncation to ``[-H, H]`` is negligible when the Gaussian is
    well inside),

        S = amp * sigma * sqrt(2 pi) * exp(-(s - c_perp)^2 / (2 sigma^2))
        E = mu_const * 2 * H
        I = S * exp(-mu_const * H) * exp(mu_const * c_par + mu_const^2 sigma^2 / 2)

    The last factor is ``int exp(-(t-c_par)^2/2sigma^2) exp(mu t) dt`` normalised
    by ``S``, i.e. attenuation referenced to the exit point ``t = +H``.
    """
    th = np.asarray(thetas, dtype=float)[:, None]
    s = np.asarray(s_nodes, dtype=float)[None, :]
    cx, cy = center
    c_perp = -np.sin(th) * cx + np.cos(th) * cy
    c_par = np.cos(th) * cx + np.sin(th) * cy
    S = amp * sigma * np.sqrt(2.0 * np.pi) * np.exp(-((s - c_perp) ** 2) / (2 * sigma**2))
    E = np.full(S.shape, mu_const * 2.0 * half_chord)
    I = S * np.exp(-mu_const * half_chord) * np.exp(
        mu_const * c_par + 0.5 * mu_const**2 * sigma**2
    )
    return S, E, I + np.zeros(S.shape)


def naive_segments(f_eval, mu_eval, p0: np.ndarray, p1: np.ndarray,
                   n_samples: int = 2049, chunk: int = 4096):
    """``(S, E, I)`` over arbitrary segments ``p0 -> p1`` by dense trapezoid.

    ``p0`` and ``p1`` are ``(..., 2)`` arrays of endpoints; integrals are with
    respect to arc length and the attenuation is referenced to ``p1``.  Used to
    check the Brandt--Dym tables, whose segments are given by their endpoints
    rather than by an angle and an offset.
    """
    p0 = np.asarray(p0, dtype=float).reshape(-1, 2)
    p1 = np.asarray(p1, dtype=float).reshape(-1, 2)
    arc = np.linalg.norm(p1 - p0, axis=-1)
    u = np.linspace(0.0, 1.0, n_samples)
    du = u[1] - u[0]
    S = np.empty(p0.shape[0])
    E = np.empty_like(S)
    I = np.empty_like(S)
    for lo in range(0, p0.shape[0], chunk):
        sl = slice(lo, lo + chunk)
        a, b = p0[sl], p1[sl]
        x = a[:, 0, None] + (b[:, 0] - a[:, 0])[:, None] * u[None, :]
        y = a[:, 1, None] + (b[:, 1] - a[:, 1])[:, None] * u[None, :]
        fv = np.asarray(f_eval(x, y), dtype=float) + np.zeros(x.shape)
        mv = np.asarray(mu_eval(x, y), dtype=float) + np.zeros(x.shape)
        ds = arc[sl][:, None] * du
        panel = 0.5 * (mv[:, :-1] + mv[:, 1:]) * ds
        tail = np.cumsum(panel[:, ::-1], axis=1)[:, ::-1]
        A = np.concatenate([tail, np.zeros((tail.shape[0], 1))], axis=1)

        def trapz(g):
            return (arc[sl] * du) * (g[:, 1:-1].sum(axis=1)
                                     + 0.5 * (g[:, 0] + g[:, -1]))

        S[sl], E[sl], I[sl] = trapz(fv), A[:, 0], trapz(fv * np.exp(-A))
    return S, E, I
