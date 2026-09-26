"""Direction grids on the sphere, for the three-dimensional transform.

In 2-D the directions form a uniform periodic grid and nesting is trivial.  On
the sphere it is not, and the parameterisation has to satisfy three things:

1. **nested refinement** -- every direction of level ``l-1`` is a direction of
   level ``l``, so half the tables are copied rather than interpolated;
2. **smooth local coordinates** -- the segment tables must be smooth functions
   of the two angular parameters for the interpolation of eq. (11) to converge;
3. **a smooth frame** -- the table of a direction ``d`` is indexed by ray
   coordinates ``(s1, s2, t)`` against an orthonormal frame ``(e1, e2, d)``, and
   that frame must itself vary smoothly with ``d``, or the tables are not smooth
   in direction no matter how smooth the fields are.

The gnomonic cube-sphere supplies all three.  Each of the six faces carries a
uniform ``(a, b)`` grid, ``d = v/|v|`` with ``v`` the face vector, and the
angular spacing is uniform up to a factor ``sqrt(3)`` between face centre and
corner.  Refinement inserts midpoints, so grids nest exactly.  Interpolation
stays inside one face and uses the ``shift`` stencil mode at face edges, which
is one-sided but not extrapolating: the new directions are midpoints and so lie
strictly inside the old grid's span.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .axis import Axis

# Face k: v = FACE_U[k]*a + FACE_V[k]*b + FACE_N[k].  TANGENT[k] is a fixed
# vector, never parallel to any direction of the face, used to build the frame.
FACE_N = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
FACE_U = np.array([[0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]], float)
FACE_V = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0]], float)
TANGENT = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0]], float)

N_FACES = 6


@dataclass(frozen=True)
class DirectionGrid:
    """``6 * m^2`` directions: face index plus a uniform ``(a, b)`` grid."""

    m: int  # nodes per face per axis
    half_width: float = 1.0

    @property
    def axis(self) -> Axis:
        return Axis.from_endpoints(-self.half_width, self.half_width, self.m)

    @property
    def n_dir(self) -> int:
        return N_FACES * self.m * self.m

    def refine(self) -> "DirectionGrid":
        """Insert midpoints: ``m -> 2m - 1``, so the old grid is the even nodes."""
        return DirectionGrid(m=2 * self.m - 1, half_width=self.half_width)

    def indices(self):
        """``(face, ia, ib)`` of every direction, in storage order."""
        f, ia, ib = np.meshgrid(
            np.arange(N_FACES), np.arange(self.m), np.arange(self.m), indexing="ij"
        )
        return f.ravel(), ia.ravel(), ib.ravel()

    def vectors(self) -> np.ndarray:
        """Unit directions, shape ``(n_dir, 3)``."""
        f, ia, ib = self.indices()
        a = self.axis.nodes[ia][:, None]
        b = self.axis.nodes[ib][:, None]
        v = FACE_N[f] + FACE_U[f] * a + FACE_V[f] * b
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    def frames(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Orthonormal frames ``(e1, e2, d)``, each ``(n_dir, 3)``.

        ``e1`` is the face tangent projected off ``d`` and normalised, so it is a
        smooth function of ``(a, b)`` within a face; ``e2 = d x e1``.
        """
        f, _, _ = self.indices()
        d = self.vectors()
        t = TANGENT[f]
        e1 = t - (np.sum(t * d, axis=-1, keepdims=True)) * d
        norm = np.linalg.norm(e1, axis=-1, keepdims=True)
        if norm.min() < 1e-6:
            raise AssertionError("frame degenerate: tangent parallel to a direction")
        e1 = e1 / norm
        e2 = np.cross(d, e1)
        return e1, e2, d

    def nested_parent_slice(self):
        """Indices of this grid's directions that belong to the coarser grid.

        Valid only when ``m`` came from :meth:`refine`, i.e. ``m`` is odd.
        """
        if self.m % 2 != 1:
            raise ValueError(f"m={self.m} is not a refinement of a coarser grid")
        keep = np.zeros((N_FACES, self.m, self.m), dtype=bool)
        keep[:, 0::2, 0::2] = True
        return keep.ravel()

    def max_angular_step(self) -> float:
        """Largest angle between neighbouring directions, in radians."""
        d = self.vectors().reshape(N_FACES, self.m, self.m, 3)
        best = 0.0
        for arr in (d[:, 1:, :, :] - d[:, :-1, :, :], d[:, :, 1:, :] - d[:, :, :-1, :]):
            best = max(best, float(np.linalg.norm(arr, axis=-1).max()))
        return 2.0 * np.arcsin(min(1.0, best / 2.0))


def grid_sequence(m0: int, n_levels: int) -> list[DirectionGrid]:
    """``n_levels + 1`` nested grids starting from ``m0`` nodes per face axis."""
    grids = [DirectionGrid(m=m0)]
    for _ in range(n_levels):
        grids.append(grids[-1].refine())
    return grids
