"""A faithful reproduction of Brandt & Dym (SIAM J. Sci. Comput. 20:1417, 1999).

Plain line integrals, no attenuation.  Kept separate from :mod:`tomogrid` so
that each ingredient of the original method -- the discretisation, the recursive
merging, the choice of interpolation, the balance between approximation and
discretisation error, and the resulting operation count -- can be exercised and
checked against the numbers published in the paper, before any of it is carried
over to the attenuated case.

Everything is in *pixel units*: the data mesh is ``H = 1``, the integration mesh
is ``h = 1/sigma``, and lengths ``L`` are horizontal extents (the paper's
``L_infinity`` measure).
"""

from .mesh import BDMesh, build_mesh
from .discrete import discrete_integral, sample_columns
from .recursion import base_level, double, run_recursion
from .truth import true_integral
from .work import operation_count, optimal_order

__all__ = [
    "BDMesh", "build_mesh", "discrete_integral", "sample_columns",
    "base_level", "double", "run_recursion", "true_integral",
    "operation_count", "optimal_order",
]
