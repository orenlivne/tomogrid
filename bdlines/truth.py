"""The true integral ``I_t`` of the underlying continuous function."""

from __future__ import annotations

import numpy as np

from .quadrature import gauss_legendre_1d


def true_integral(fun, x0, y0, length: int, rise, n_nodes: int = 200):
    """Normalised integral of ``fun(x, y)`` over ``(x0,y0) -> (x0+L, y0+rise)``.

    Gauss--Legendre in the horizontal parameter; ``fun`` is the continuum
    function, not the sampled image, so this is the paper's ``I_t``.
    """
    x0 = np.asarray(x0, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    rise = np.asarray(rise, dtype=float)
    u, w = gauss_legendre_1d(n_nodes)  # on [0, 1], weights summing to 1
    shape = np.broadcast(x0, y0, rise).shape + (n_nodes,)
    x = np.broadcast_to(x0[..., None], shape) + length * u
    y = (np.broadcast_to(y0[..., None], shape)
         + np.broadcast_to(rise[..., None], shape) * u)
    return (np.asarray(fun(x, y), dtype=float) * w).sum(axis=-1)
