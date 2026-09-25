"""The level hierarchy: segment lengths, angular grids, and ``(s, t)`` grids.

Level ``l`` holds segments of length ``l_l = 2^l * l_0`` with

    dt = l_l / nu             (segments OVERLAP; nu even, nested across levels)
    ds = image spacing / sigma
    n_theta = min(n_theta_out, n_theta_min * 2^l)

``nu = t_oversample`` matters more than it looks.  A segment integral is a box
average of length ``l`` along the ray, so its spectrum carries the factor
``sinc(k*l/2)``, which is still ``0.64`` at the Nyquist frequency of a grid with
``dt = l``.  Sampling the tables at ``dt = l`` therefore aliases badly and caps
the accuracy of the ``(s,t)`` interpolation at ~1e-3 no matter how fine the
angular grid is.  Taking ``nu = 2`` puts the grid Nyquist exactly on the first
*zero* of that sinc, which is the natural choice; ``nu = 4`` buys a little more.

``nu`` must be even: the two half-segments of a level-``l`` segment centred at
``t`` sit at ``t -/+ l_{l-1}/2 = t -/+ (nu/2)*dt_{l-1}``, which is an integer
number of level-``(l-1)`` grid steps only when ``nu`` is even.  That is what keeps
the merge exact -- no interpolation along ``t`` ever happens.

The last line is the scheduling rule of docs/formulation.md eq. (7): angular
refinement happens at the *bottom* levels where segments are short and the
criterion ``dtheta * l <~ h`` is cheap to satisfy; the top levels coast at full
angular resolution doing nothing but exact merges.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .axis import Axis


def thetas_of(n: int) -> np.ndarray:
    """The uniform periodic direction grid ``2*pi*i/n``, ``i = 0..n-1``.

    Nested by construction: the grid of ``n`` directions is the even-indexed
    subset of the grid of ``2n``.
    """
    return 2.0 * np.pi * np.arange(n) / n


@dataclass(frozen=True)
class Level:
    index: int
    seg_len: float
    n_theta: int
    t_axis: Axis

    @property
    def thetas(self) -> np.ndarray:
        return thetas_of(self.n_theta)

    @property
    def table_size(self) -> int:
        return self.n_theta * self.t_axis.n


@dataclass(frozen=True)
class Hierarchy:
    levels: tuple[Level, ...]
    s_axis: Axis
    half_chord: float
    n_theta_out: int
    n_theta_min: int
    t_oversample: int = 2

    @property
    def n_levels(self) -> int:
        return len(self.levels) - 1

    @property
    def top(self) -> Level:
        return self.levels[-1]

    @property
    def refine_levels(self) -> tuple[int, ...]:
        """Levels at which the angular grid is actually refined (the only levels
        that introduce any error at all)."""
        return tuple(
            l.index
            for l in self.levels[1:]
            if l.n_theta != self.levels[l.index - 1].n_theta
        )

    @property
    def max_refine_seg_len(self) -> float:
        """Longest segment involved in an angular interpolation."""
        idx = self.refine_levels
        return max((self.levels[i - 1].seg_len for i in idx), default=0.0)

    def max_angular_displacement(self) -> float:
        """``max_l dtheta_l * l_l`` over the refinement levels, in *length* units.

        This is how far a segment endpoint moves between neighbouring directions.
        Eq. (6) asks for it to stay below about one image cell; divide by the image
        spacing with :meth:`angular_criterion` to get the dimensionless constant."""
        out = 0.0
        for i in self.refine_levels:
            prev = self.levels[i - 1]
            out = max(out, (2 * np.pi / prev.n_theta) * prev.seg_len)
        return out

    def angular_criterion(self, image_h: float) -> float:
        """The constant ``c`` of eq. (6), measured against the *image* spacing."""
        return self.max_angular_displacement() / image_h

    def total_table_points(self) -> int:
        return sum(l.table_size for l in self.levels) * self.s_axis.n

    def describe(self) -> str:
        rows = [
            f"  L{l.index}: seg_len={l.seg_len:.5g} n_t={l.t_axis.n:5d} "
            f"n_theta={l.n_theta:5d} table={l.table_size * self.s_axis.n:9d}"
            for l in self.levels
        ]
        return "\n".join(
            [
                f"Hierarchy: {self.n_levels} levels, half_chord={self.half_chord}, "
                f"n_s={self.s_axis.n}, ds={self.s_axis.h:.5g}",
                f"  refine at levels {self.refine_levels}, "
                f"max dtheta*seg_len={self.max_angular_displacement():.4g}",
                *rows,
            ]
        )


def build_hierarchy(
    *,
    n_theta_out: int,
    n_theta_min: int,
    n_levels: int,
    s_axis: Axis,
    half_chord: float = 1.0,
    t_oversample: int = 2,
) -> Hierarchy:
    """Assemble the level hierarchy.

    ``n_theta_out`` must be ``n_theta_min * 2**k`` for some ``0 <= k <= n_levels``
    so that the angular grids nest exactly and the top level really does reach the
    requested resolution.
    """
    if n_levels < 0:
        raise ValueError(f"n_levels must be >= 0, got {n_levels}")
    if n_theta_min < 1 or n_theta_out < 1:
        raise ValueError("angle counts must be positive")
    if n_theta_out < n_theta_min:
        raise ValueError(
            f"n_theta_out={n_theta_out} < n_theta_min={n_theta_min}; "
            "lower n_theta_min or raise n_theta_out"
        )
    ratio = n_theta_out / n_theta_min
    k = round(math.log2(ratio))
    if abs(2.0**k - ratio) > 1e-12:
        raise ValueError(
            f"n_theta_out/n_theta_min = {ratio} is not a power of two "
            f"(n_theta_out={n_theta_out}, n_theta_min={n_theta_min})"
        )
    if k > n_levels:
        raise ValueError(
            f"need n_theta_out <= n_theta_min * 2**n_levels: "
            f"{n_theta_out} > {n_theta_min * 2 ** n_levels}"
        )
    if not half_chord > 0:
        raise ValueError("half_chord must be positive")
    nu = int(t_oversample)
    if nu < 2 or nu % 2 != 0:
        raise ValueError(
            f"t_oversample must be an even integer >= 2, got {t_oversample}; "
            "odd values put the half-segment centres off-grid and would force an "
            "interpolation into the (otherwise exact) merge"
        )

    levels = []
    for l in range(n_levels + 1):
        n_seg = 2 ** (n_levels - l)
        seg_len = 2.0 * half_chord / n_seg
        # Nodes t = -H + j*(seg_len/nu), covering [-H, H] inclusive.  Level l is
        # every 2nd node of level l-1, so the grids nest exactly.
        t_axis = Axis(origin=-half_chord, h=seg_len / nu, n=nu * n_seg + 1)
        n_theta = min(n_theta_out, n_theta_min * 2**l)
        levels.append(Level(index=l, seg_len=seg_len, n_theta=n_theta, t_axis=t_axis))
    return Hierarchy(
        levels=tuple(levels),
        s_axis=s_axis,
        half_chord=half_chord,
        n_theta_out=n_theta_out,
        n_theta_min=n_theta_min,
        t_oversample=nu,
    )
