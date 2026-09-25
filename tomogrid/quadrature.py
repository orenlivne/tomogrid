"""Gauss-Legendre rule plus the *tail-integral* matrix.

The level-0 kernel needs, from the same ``m`` samples of ``mu`` on a segment,
both the full integral and every partial integral ``int_{t_i}^{b} mu``.  Taking
the Lagrange interpolant of ``mu`` through the Gauss-Legendre nodes and
integrating it exactly gives both, at order ``m``.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=32)
def gauss_legendre(m: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights of the ``m``-point Gauss-Legendre rule on ``[-1, 1]``."""
    if m < 1:
        raise ValueError(f"need m >= 1, got {m}")
    x, w = np.polynomial.legendre.leggauss(m)
    return x.copy(), w.copy()


@lru_cache(maxsize=32)
def tail_integral_matrix(m: int) -> np.ndarray:
    """``C[i, j] = int_{x_i}^{1} L_j(x) dx`` on ``[-1, 1]``.

    ``L_j`` is the Lagrange basis on the ``m`` Gauss-Legendre nodes, so
    ``sum_j C[i,j] g(x_j)`` is an order-``m`` accurate ``int_{x_i}^1 g``, exact for
    polynomials of degree ``< m``.
    """
    x, _ = gauss_legendre(m)
    # L_j(x) = sum_k A[k, j] x^k
    vander = np.vander(x, m, increasing=True)
    coeff = np.linalg.inv(vander)
    k = np.arange(m)
    # int_{x_i}^1 x^k dx = (1 - x_i^{k+1}) / (k+1)
    mono = (1.0 - x[:, None] ** (k + 1)) / (k + 1)
    return mono @ coeff


@lru_cache(maxsize=32)
def head_integral_matrix(m: int) -> np.ndarray:
    """``C[i, j] = int_{-1}^{x_i} L_j(x) dx`` on ``[-1, 1]`` (the reverse direction)."""
    _, w = gauss_legendre(m)
    return w[None, :] - tail_integral_matrix(m)
