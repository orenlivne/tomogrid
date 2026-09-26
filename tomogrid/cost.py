"""Analytic operation counts, so complexity can be asserted without timing.

:func:`estimate_work` reproduces the counters that :func:`tomogrid.forward`
accumulates at run time (the tests check that they agree), which makes it cheap
to compare the ``O(N log N)`` multilevel schedule against the ``O(N^1.5)`` direct
one at sizes far too large to actually run.
"""

from __future__ import annotations

from .hierarchy import Hierarchy


def estimate_work(hier: Hierarchy, *, gl_order: int, img_order: int,
                  theta_order: int, s_order: int, t_order: int) -> dict:
    """Operation counts for one :func:`tomogrid.forward` call.

    ``gathers`` is the dominant cost: image lookups at level 0 plus table
    lookups in every angular prolongation.
    """
    n_s = hier.s_axis.n
    level0_nodes = hier.levels[0].n_theta * n_s * hier.levels[0].t_axis.n * gl_order
    table_points = sum(lv.n_theta * n_s * lv.t_axis.n for lv in hier.levels)

    interp_targets = 0
    interp_gathers = 0
    for l in hier.refine_levels:
        prev, cur = hier.levels[l - 1], hier.levels[l]
        q = min(theta_order, prev.n_theta)
        targets = 3 * (cur.n_theta // 2) * n_s * prev.t_axis.n
        interp_targets += targets
        interp_gathers += targets * q * s_order * t_order

    # Level-0 also applies the m x m tail-integral matrix once per segment,
    # i.e. gl_order multiply-adds per node.
    level0_quad = level0_nodes * gl_order
    gathers = 2 * level0_nodes * img_order**2 + interp_gathers
    return {
        "level0_nodes": level0_nodes,
        "table_points": table_points,
        "interp_targets": interp_targets,
        "interp_gathers": interp_gathers,
        # 2 fields (f and mu) sampled with an img_order^2 stencil at level 0.
        "gathers": gathers,
        "level0_quad": level0_quad,
        # Each gather is one multiply and one add.
        "flops": 2 * (gathers + level0_quad),
    }
