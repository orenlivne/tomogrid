"""The integration mesh and the level hierarchy (paper, section 2).

Three resolution rules, each derived in the paper and each visible here:

* angular resolution proportional to the integration length -- so the slope set
  at level ``k`` has spacing ``h / L_k`` and doubles in size when ``L`` doubles;
* spacing of evaluation points *along* the direction inversely proportional to
  ``L`` -- so the x starts are spaced ``L_k`` and halve in number;
* spacing *transverse* to the direction independent of ``L``, set by the data
  bandwidth -- so the y starts are spaced ``h`` at every level.

Consequently the number of integrals per level does not depend on ``L``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BDLevel:
    index: int
    length: int  # horizontal extent, in pixels
    n_x: int
    rise: int  # rises run -rise .. rise, in units of h

    @property
    def n_slope(self) -> int:
        return 2 * self.rise + 1


@dataclass(frozen=True)
class BDMesh:
    n_rows: int  # image height in pixels
    n_cols: int  # image width in pixels
    sigma: int  # h = 1 / sigma
    levels: tuple[BDLevel, ...]

    @property
    def h(self) -> float:
        return 1.0 / self.sigma

    @property
    def n_y(self) -> int:
        """Number of integration-mesh rows spanning the image."""
        return (self.n_rows - 1) * self.sigma + 1

    @property
    def top(self) -> BDLevel:
        return self.levels[-1]

    def y_nodes(self) -> np.ndarray:
        """Integration-mesh row coordinates, in pixel units."""
        return np.arange(self.n_y) * self.h

    def x_nodes(self, lv: BDLevel) -> np.ndarray:
        return np.arange(lv.n_x) * lv.length

    def slopes(self, lv: BDLevel) -> np.ndarray:
        return np.arange(-lv.rise, lv.rise + 1) * (self.h / lv.length)

    def table_size(self, lv: BDLevel) -> int:
        return lv.n_x * self.n_y * lv.n_slope

    def total_table_size(self) -> int:
        return sum(self.table_size(lv) for lv in self.levels)

    def describe(self) -> str:
        rows = [f"  L{lv.index}: length={lv.length:4d} n_x={lv.n_x:5d} "
                f"n_slope={lv.n_slope:6d} table={self.table_size(lv):10d}"
                for lv in self.levels]
        return "\n".join([
            f"BDMesh {self.n_rows}x{self.n_cols}, H/h = {self.sigma}, "
            f"n_y = {self.n_y}", *rows])


def build_mesh(n_rows: int, n_cols: int, *, sigma: int = 1,
               length0: int = 2, max_length: int | None = None) -> BDMesh:
    """Levels from ``length0`` up to ``max_length`` (default: the image width)."""
    if sigma < 1:
        raise ValueError(f"sigma must be >= 1, got {sigma}")
    if length0 < 1 or (length0 & (length0 - 1)):
        raise ValueError(f"length0 must be a power of two, got {length0}")
    top = max_length if max_length is not None else n_cols - 1
    if top < length0:
        raise ValueError(f"max_length {top} below length0 {length0}")
    n_levels = int(round(np.log2(top / length0)))
    if length0 * 2**n_levels != top:
        raise ValueError(f"max_length {top} is not {length0} * 2^k")

    levels = []
    for k in range(n_levels + 1):
        length = length0 * 2**k
        levels.append(BDLevel(index=k, length=length,
                              n_x=max(1, (n_cols - 1) // length),
                              rise=length * sigma))
    return BDMesh(n_rows=n_rows, n_cols=n_cols, sigma=sigma,
                  levels=tuple(levels))
