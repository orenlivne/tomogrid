"""The operation counts, and the precision the recursion needs."""

from __future__ import annotations

import numpy as np
import pytest

from tomogrid.backend import SinglePrecision
from tomogrid.bd import build_grid, forward
from tomogrid.bd3d import build_grid3d
from tomogrid.gpucost import (DEVICES, Counts, base_counts, forward_counts,
                              peak_table_entries, refine3d_counts,
                              refine_counts, roofline, sweep_counts)
from tomogrid.image import sample_function, square_grid


def _grid(n=129, sigma=2):
    return build_grid(half_width=1.0, image_h=2.0 / (n - 1), sigma=sigma)


def test_counts_add():
    a = Counts(1, 2, 3, 4, 5)
    b = a + a
    assert (b.points, b.flops, b.exps, b.reads, b.writes) == (2, 4, 6, 8, 10)


def test_refine_counts_match_the_table_shapes():
    grid = _grid()
    for k in range(len(grid.levels) - 1):
        nxt = grid.levels[k + 1]
        c = refine_counts(grid, k)
        assert c.points == nxt.n_x * grid.n_y * nxt.n_m
        assert c.writes == 3 * c.points
        assert c.exps == c.points


def test_doubling_cost_is_flat_across_levels():
    """The whole point of the indexing: centres halve as directions double."""
    grid = _grid(257)
    per = [refine_counts(grid, k).bytes_moved()
           for k in range(len(grid.levels) - 1)]
    assert max(per) / min(per) < 1.15


def test_three_dimensional_table_doubles_each_level():
    """Directions quadruple while segment centres halve, so tables double.

    This is why three dimensions is dominated by its top level -- the output
    of the 3-D transform is itself four-dimensional -- while in two dimensions
    the tables are the same size at every level.
    """
    grid = build_grid3d(half_width=1.0, image_h=2.0 / 32, sigma=1)
    per = [refine3d_counts(grid, k).points
           for k in range(len(grid.levels) - 1)]
    for a, b in zip(per, per[1:]):
        assert 1.7 < b / a <= 2.0


def test_the_recursion_is_memory_bound_and_the_base_level_is_not():
    """The two halves of the algorithm sit on opposite sides of the roofline.

    A doubling moves three tables and does a few dozen flops per entry, so it
    is bandwidth-bound on anything.  The base level interpolates the image at
    every Gauss point of every short segment out of a few megabytes that stay
    in cache, so it is arithmetic-dense.  A device implementation has to be
    written for both, and it is the recursion that sets the floor.
    """
    grid = _grid(257)
    rec = None
    for k in range(len(grid.levels) - 1):
        c = refine_counts(grid, k)
        rec = c if rec is None else rec + c
    assert rec.intensity() < 1.0
    for d in DEVICES:
        assert roofline(rec, d, 8)["bound"] == "memory"
    assert base_counts(grid).intensity() > 10 * rec.intensity()


def test_base_level_is_the_arithmetic_heavy_part():
    grid = _grid(257)
    base = base_counts(grid)
    doubling = sweep_counts(grid) + Counts(*(-v for v in (
        base.points, base.flops, base.exps, base.reads, base.writes)))
    assert base.total_flops() > doubling.total_flops()


def test_live_footprint_counts_three_tables_of_three():
    grid = _grid(129)
    biggest = max(lv.n_x * grid.n_y * lv.n_m for lv in grid.levels)
    assert peak_table_entries(grid) == 9 * biggest


def test_unknown_dtype_is_rejected():
    with pytest.raises(KeyError):
        forward_counts(_grid(65)).total_flops(2)


@pytest.mark.parametrize("n", [33, 65, 129])
def test_single_precision_does_not_amplify(n):
    """fp32 must cost only fp32 roundoff, however many levels there are.

    The merge is a convex-ish combination with a damping factor in ``(0, 1]``,
    so errors are carried, not grown.  This is what makes single precision --
    and so consumer hardware, where fp64 runs at a 64th of the rate -- usable.
    """
    xa, ya = square_grid(n)
    f = sample_function(lambda x, y: np.exp(-6 * (x**2 + y**2)), xa, ya)
    mu = sample_function(lambda x, y: 0.5 / (1 + 4 * (x**2 + y**2)), xa, ya)
    a = forward(f, mu, sigma=2)
    b = forward(f, mu, sigma=2, xp=SinglePrecision())
    for name in a:
        assert b[name].triple.I.dtype == np.float32
        x, y = a[name].integrals().I, b[name].integrals().I
        rel = np.abs(x - y).max() / np.abs(x).max()
        assert rel < 1e-6, f"{name}: {rel:.2e}"


def test_direct_is_arithmetic_bound_and_grows_faster():
    """Direct quadrature costs a factor n more arithmetic, and it shows."""
    from tomogrid.gpucost import direct_counts

    ratios = []
    for n in (129, 257, 513):
        grid = _grid(n)
        ml, dr = forward_counts(grid), direct_counts(grid, n)
        a = roofline(ml, DEVICES[2], 8)
        b = roofline(dr, DEVICES[2], 8)
        assert a["bound"] == "memory" and b["bound"] == "compute"
        ratios.append(b["t"] / a["t"])
    assert ratios == sorted(ratios)
    assert ratios[-1] > 2 * ratios[0]
