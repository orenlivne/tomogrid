"""Discrete 2-D fields and off-grid sampling.

An :class:`Image` is ``values[i, j]`` at ``(x_axis.nodes[i], y_axis.nodes[j])``.
Sampling off-grid uses tensor-product Lagrange interpolation with zero-fill, so
the *continuum* object the solver actually transforms is the Lagrange interpolant
of the samples, extended by zero.  All reference evaluators use the same rule, so
that discretisation error and algorithmic error can be separated in the tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .axis import Axis
from .interp import interp2d


@dataclass(frozen=True)
class Image:
    values: np.ndarray
    x_axis: Axis
    y_axis: Axis

    def __post_init__(self) -> None:
        if self.values.shape != (self.x_axis.n, self.y_axis.n):
            raise ValueError(
                f"values shape {self.values.shape} != "
                f"({self.x_axis.n}, {self.y_axis.n})"
            )

    @property
    def h(self) -> float:
        """Nominal spacing (the two axes are required to match)."""
        if not np.isclose(self.x_axis.h, self.y_axis.h):
            raise ValueError("anisotropic image grids are not supported")
        return self.x_axis.h

    def sample(self, x: np.ndarray, y: np.ndarray, order: int = 4) -> np.ndarray:
        return interp2d(self.values, self.x_axis, self.y_axis, x, y, order, order)


def sample_function(fun, x_axis: Axis, y_axis: Axis) -> Image:
    """Tabulate a callable ``fun(x, y)`` on the grid."""
    xx = x_axis.nodes[:, None]
    yy = y_axis.nodes[None, :]
    return Image(values=np.asarray(fun(xx, yy), dtype=float) + np.zeros((x_axis.n, y_axis.n)),
                 x_axis=x_axis, y_axis=y_axis)


def square_grid(n: int, half_width: float = 1.0) -> tuple[Axis, Axis]:
    """``n x n`` node grid on ``[-half_width, half_width]^2``."""
    ax = Axis.from_endpoints(-half_width, half_width, n)
    return ax, ax


@dataclass(frozen=True)
class Image3D:
    """``values[i, j, k]`` at ``(x_axis[i], y_axis[j], z_axis[k])``."""

    values: np.ndarray
    x_axis: Axis
    y_axis: Axis
    z_axis: Axis

    def __post_init__(self) -> None:
        want = (self.x_axis.n, self.y_axis.n, self.z_axis.n)
        if self.values.shape != want:
            raise ValueError(f"values shape {self.values.shape} != {want}")

    @property
    def h(self) -> float:
        hs = [self.x_axis.h, self.y_axis.h, self.z_axis.h]
        if not np.allclose(hs, hs[0]):
            raise ValueError("anisotropic image grids are not supported")
        return hs[0]

    def sample(self, x, y, z, order: int = 4) -> np.ndarray:
        from .interp import interp3d

        return interp3d(self.values, self.x_axis, self.y_axis, self.z_axis,
                        x, y, z, order)


def sample_function3d(fun, x_axis: Axis, y_axis: Axis, z_axis: Axis) -> Image3D:
    xx = x_axis.nodes[:, None, None]
    yy = y_axis.nodes[None, :, None]
    zz = z_axis.nodes[None, None, :]
    shape = (x_axis.n, y_axis.n, z_axis.n)
    vals = np.asarray(fun(xx, yy, zz), dtype=float) + np.zeros(shape)
    return Image3D(values=vals, x_axis=x_axis, y_axis=y_axis, z_axis=z_axis)


def cube_grid(n: int, half_width: float = 1.0) -> tuple[Axis, Axis, Axis]:
    ax = Axis.from_endpoints(-half_width, half_width, n)
    return ax, ax, ax
