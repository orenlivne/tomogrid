"""Operation counts (paper, section 3, "Accuracy analysis").

Per level the table size is independent of ``L``, so with ``l ~ log n`` levels
and ``O(p)` operations per integral the work is ``O(p n log n)`` when ``h = H``.
Refining the integration mesh multiplies the table size by ``sigma^2``, so

    W = O( p sigma^2 n log n ),

while the error per interpolation falls as ``(h/H)^p = sigma^-p``.  Requiring
``l sigma^-p ~ q`` and optimising ``p`` gives the paper's
``O(n log n log log n)``.
"""

from __future__ import annotations

import numpy as np

from .mesh import BDMesh


def operation_count(mesh: BDMesh, order: int = 2) -> dict:
    """Multiply--adds for the doubling stages, and entries touched."""
    entries = 0
    ops = 0
    for k in range(len(mesh.levels) - 1):
        nxt = mesh.levels[k + 1]
        n_even = mesh.levels[k].n_slope
        n_odd = nxt.n_slope - n_even
        per_x_y = nxt.n_x * mesh.n_y
        entries += per_x_y * nxt.n_slope
        # even: one add and one halving; odd: 2 * order multiply-adds
        ops += per_x_y * (2 * n_even + n_odd * (4 * order + 1))
    return {"entries": entries, "ops": ops,
            "levels": len(mesh.levels) - 1,
            "table_total": mesh.total_table_size()}


def optimal_order(n: int) -> int:
    """The paper's ``p = 2 ln log n``, which minimises ``p n (log n)^(1+2/p)``."""
    return max(2, int(round(2.0 * np.log(np.log2(n)))))
