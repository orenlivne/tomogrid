"""Uniform 1-D axes.  Every grid in the code is a uniform axis, so this is the
single place where the origin/spacing/count convention lives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Axis:
    """Uniform grid of ``n`` nodes: ``x_i = origin + i*h``, ``i = 0..n-1``."""

    origin: float
    h: float
    n: int

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError(f"Axis needs n >= 1, got {self.n}")
        if not self.h > 0:
            raise ValueError(f"Axis needs h > 0, got {self.h}")

    @property
    def nodes(self) -> np.ndarray:
        return self.origin + self.h * np.arange(self.n)

    @property
    def lo(self) -> float:
        return self.origin

    @property
    def hi(self) -> float:
        return self.origin + self.h * (self.n - 1)

    @classmethod
    def from_endpoints(cls, lo: float, hi: float, n: int) -> "Axis":
        """``n`` nodes with ``x_0 = lo`` and ``x_{n-1} = hi``."""
        if n < 2:
            raise ValueError("from_endpoints needs n >= 2")
        return cls(origin=lo, h=(hi - lo) / (n - 1), n=n)

    @classmethod
    def cell_centers(cls, lo: float, hi: float, n: int) -> "Axis":
        """``n`` cell centres of a uniform partition of ``[lo, hi]``."""
        h = (hi - lo) / n
        return cls(origin=lo + 0.5 * h, h=h, n=n)
