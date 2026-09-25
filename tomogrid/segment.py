"""Segment triples ``(S, E, I)`` and the exact merge that drives the hierarchy.

For a segment ``[a, b]`` of a ray (traversed towards increasing ``t``):

    E = int_a^b mu
    S = int_a^b f
    I = int_a^b f(t) exp(-int_t^b mu) dt      (attenuation referenced to b)

See docs/formulation.md sec. 2.  All arrays are elementwise; any common shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .quadrature import gauss_legendre, tail_integral_matrix


@dataclass
class Triple:
    """The three segment integrals, as arrays of identical shape."""

    S: np.ndarray
    E: np.ndarray
    I: np.ndarray

    def __post_init__(self) -> None:
        if not (self.S.shape == self.E.shape == self.I.shape):
            raise ValueError(
                f"shape mismatch: S{self.S.shape} E{self.E.shape} I{self.I.shape}"
            )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.S.shape

    def pet(self) -> np.ndarray:
        """The PET forward model for this segment: ``exp(-E) * S``."""
        return np.exp(-self.E) * self.S

    def __getitem__(self, key) -> "Triple":
        return Triple(S=self.S[key], E=self.E[key], I=self.I[key])

    def copy(self) -> "Triple":
        return Triple(S=self.S.copy(), E=self.E.copy(), I=self.I.copy())


def merge(left: Triple, right: Triple) -> Triple:
    """Concatenate ``left = [a,c]`` with ``right = [c,b]`` into ``[a,b]``.

    Exact -- no quadrature, no interpolation.  The factor ``exp(-right.E) <= 1``
    also damps any pre-existing error in ``left.I``, so the recursion is stable
    under strong attenuation.
    """
    return Triple(
        S=left.S + right.S,
        E=left.E + right.E,
        I=np.exp(-right.E) * left.I + right.I,
    )


def merge_level(tri: Triple, t_oversample: int) -> Triple:
    """Merge level ``l-1`` into level ``l`` along the last (``t``) axis.

    With node grid ``t_j = -H + j*dt`` and ``dt_{l-1} = l_{l-1}/nu``, the level-``l``
    segment centred at level-``(l-1)`` index ``2j`` is the union of the two
    level-``(l-1)`` segments centred at ``2j -/+ nu/2``.  Out-of-range neighbours are
    zero-padded, which is exact whenever ``half_chord`` exceeds the support radius.

    Still exact: no interpolation, only eq. (4).
    """
    nu = int(t_oversample)
    if nu < 2 or nu % 2 != 0:
        raise ValueError(f"t_oversample must be even >= 2, got {t_oversample}")
    n_old = tri.shape[-1]
    if (n_old - 1) % 2 != 0:
        raise ValueError(f"t axis length {n_old} must be odd (node grid)")
    n_new = (n_old - 1) // 2 + 1
    half = nu // 2

    pad = [(0, 0)] * tri.S.ndim
    pad[-1] = (half, half)
    left_idx = slice(0, 2 * n_new, 2)
    right_idx = slice(nu, nu + 2 * n_new, 2)
    parts = []
    for arr in (tri.S, tri.E, tri.I):
        padded = np.pad(arr, pad, mode="constant")
        parts.append((padded[..., left_idx], padded[..., right_idx]))
    left = Triple(S=parts[0][0], E=parts[1][0], I=parts[2][0])
    right = Triple(S=parts[0][1], E=parts[1][1], I=parts[2][1])
    return merge(left, right)


def quad_spectral(f_nodes: np.ndarray, mu_nodes: np.ndarray, length: float) -> Triple:
    """Level-0 kernel: the triple of one segment from ``m`` Gauss-Legendre samples.

    ``f_nodes`` and ``mu_nodes`` have shape ``(..., m)``, sampled at the
    Gauss-Legendre nodes of the segment ordered from ``a`` to ``b``.  Order ``m``
    accurate in ``length``.
    """
    m = f_nodes.shape[-1]
    if mu_nodes.shape != f_nodes.shape:
        raise ValueError(f"shape mismatch {f_nodes.shape} vs {mu_nodes.shape}")
    _, w = gauss_legendre(m)
    scale = 0.5 * length
    E = scale * (f_nodes * 0.0 + mu_nodes) @ w
    S = scale * f_nodes @ w
    # tail[..., i] ~= int_{t_i}^{b} mu
    tail = scale * mu_nodes @ tail_integral_matrix(m).T
    I = scale * np.einsum("...i,...i->...", w * f_nodes, np.exp(-tail))
    return Triple(S=S, E=E, I=I)


def quad_linear(f_ends: np.ndarray, mu_mid: np.ndarray, length: float) -> Triple:
    """Second-order alternative level-0 kernel, in closed form.

    ``f`` linear across the segment (values ``f_ends[..., 0]`` at ``a`` and
    ``f_ends[..., 1]`` at ``b``), ``mu`` constant at ``mu_mid``.  Then
    ``I = f1*A0 - (f1-f0)/d * A1`` with ``A0 = int_0^d e^{-m u} du`` and
    ``A1 = int_0^d u e^{-m u} du``, both evaluated stably at small ``m``.
    """
    f0 = f_ends[..., 0]
    f1 = f_ends[..., 1]
    d = float(length)
    m = np.asarray(mu_mid, dtype=float)
    md = m * d
    # A0 = (1 - exp(-md)) / m  -> d * (-expm1(-md)/md), stable via expm1.
    small = np.abs(md) < 1e-8
    md_safe = np.where(small, 1.0, md)
    a0 = d * np.where(small, 1.0 - 0.5 * md, -np.expm1(-md_safe) / md_safe)
    # A1 = (1 - exp(-md)*(1+md)) / m^2 -> d^2 * (1 - e^{-md}(1+md)) / md^2
    a1 = d * d * np.where(
        small,
        0.5 - md / 3.0,
        (1.0 - np.exp(-md_safe) * (1.0 + md_safe)) / (md_safe * md_safe),
    )
    return Triple(
        S=0.5 * d * (f0 + f1),
        E=d * m,
        I=f1 * a0 - (f1 - f0) / d * a1,
    )
