"""Operation and memory-traffic counts for the doubling recursion, and the
roofline they imply on a given device.

Everything here is counted from array shapes and the stencil width, not
measured and not fitted, so the same numbers can be checked by hand against
:mod:`tomogrid.bd` and :mod:`tomogrid.bd3d`.  The only device-specific input is
a :class:`Device`'s peak bandwidth and peak rate, so a projection can be read as
what it is: a bound, not a benchmark.

Per output entry a doubling costs

* ``2`` flops each for ``S`` and ``E`` (average the two halves),
* ``4`` flops and one ``exp`` for ``I`` (damp the first half, add, halve),
* ``2 q`` flops for each of the three quantities on each of the two halves,
  when the target direction is new and the slope index has to be interpolated
  with a ``q``-point stencil.

The traffic is the three input tables, read once as first halves and once
sheared as second halves, plus the three output tables written once.  A fused
kernel would do better on the second read; nothing here assumes it does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bd import BDGrid
from .bd3d import BD3Grid

#: Flops charged for one exponential.  Single precision has a hardware
#: approximation; double precision is a polynomial in software.
EXP_FLOPS = {4: 6.0, 8: 20.0}


@dataclass(frozen=True)
class Counts:
    """Work and traffic for a piece of the algorithm, per quantity triple."""

    points: float  # output table entries produced
    flops: float
    exps: float
    reads: float  # array *entries* read (not bytes)
    writes: float

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(*(a + b for a, b in zip(
            (self.points, self.flops, self.exps, self.reads, self.writes),
            (other.points, other.flops, other.exps, other.reads,
             other.writes))))

    def bytes_moved(self, dtype_bytes: int = 8) -> float:
        return dtype_bytes * (self.reads + self.writes)

    def total_flops(self, dtype_bytes: int = 8) -> float:
        return self.flops + self.exps * EXP_FLOPS[dtype_bytes]

    def intensity(self, dtype_bytes: int = 8) -> float:
        """Flops per byte of traffic -- where this sits on a roofline."""
        return self.total_flops(dtype_bytes) / self.bytes_moved(dtype_bytes)


_MERGE_FLOPS = 2 + 2 + 4  # S, E, I


def refine_counts(grid: BDGrid, k: int, order: int = 4) -> Counts:
    """One 2-D doubling, level ``k`` -> ``k+1``."""
    lv, nxt = grid.levels[k], grid.levels[k + 1]
    n_y = grid.n_y
    t_in = lv.n_x * n_y * lv.n_m
    n_even = nxt.n_x * n_y * lv.n_m
    n_odd = nxt.n_x * n_y * (nxt.n_m - lv.n_m)
    q = min(order, lv.n_m)
    flops = (n_even + n_odd) * _MERGE_FLOPS + n_odd * 3 * 2 * 2 * q
    return Counts(points=n_even + n_odd, flops=flops, exps=n_even + n_odd,
                  reads=3 * 2 * t_in, writes=3 * (n_even + n_odd))


def base_counts(grid: BDGrid, gl_order: int = 8, img_order: int = 4) -> Counts:
    """Direct quadrature of the shortest segments.

    Two fields sampled at ``gl_order`` Gauss points per segment by
    tensor-product Lagrange interpolation (``img_order**2`` taps), then the
    Gauss sums: ``S`` and ``E`` are one contraction each, the attenuation tail
    is a ``gl_order``-square triangular solve done as a small matrix product,
    and ``I`` needs one exponential per Gauss point.

    As in :func:`direct_counts`, the interpolation taps are charged as cache
    hits: the two fields are counted as read once.  The base level is
    therefore the arithmetic-dense part of the algorithm, which is what it is.
    """
    lv = grid.levels[0]
    seg = lv.n_x * grid.n_y * lv.n_m
    nodes = seg * gl_order
    sample = 2 * nodes * (2 * img_order**2 + 4 * img_order)  # f and mu
    sums = seg * (2 * 2 * gl_order + 2 * gl_order**2 + 3 * gl_order)
    return Counts(points=seg, flops=sample + sums, exps=nodes,
                  reads=2 * grid.n_y**2, writes=3 * seg)


def sweep_counts(grid: BDGrid, *, order: int = 4, gl_order: int = 8,
                 img_order: int = 4) -> Counts:
    """Base level plus every doubling, for one direction family."""
    total = base_counts(grid, gl_order, img_order)
    for k in range(len(grid.levels) - 1):
        total = total + refine_counts(grid, k, order)
    return total


def forward_counts(grid: BDGrid, *, families: int = 4, **kw) -> Counts:
    one = sweep_counts(grid, **kw)
    return Counts(*(families * v for v in (one.points, one.flops, one.exps,
                                           one.reads, one.writes)))


def peak_table_entries(grid: BDGrid) -> int:
    """Largest number of entries live at once, per quantity.

    A doubling holds the level-``k`` table, its sheared copy and the level-
    ``k+1`` table, so three tables of three quantities.
    """
    return 3 * 3 * max(lv.n_x * grid.n_y * lv.n_m for lv in grid.levels)


# --- three dimensions ------------------------------------------------------


def refine3d_counts(grid: BD3Grid, k: int, order: int = 4) -> Counts:
    """One 3-D doubling.  Both slope axes are refined, one at a time."""
    lv, nxt = grid.levels[k], grid.levels[k + 1]
    n_t = grid.n_t
    t_in = lv.n_x * n_t * n_t * lv.n_m**2
    t_out = nxt.n_x * n_t * n_t * nxt.n_m**2
    q = min(order, lv.n_m)
    # Two 1-D prolongations per half per quantity: the first doubles one slope
    # axis, the second works on the already-doubled table.  Sum factorisation,
    # so O(q) per entry and not O(q^2).
    half = nxt.n_x * n_t * n_t
    pass1 = half * (nxt.n_m * lv.n_m) * 2 * q
    pass2 = half * (nxt.n_m * nxt.n_m) * 2 * q
    interp = 3 * 2 * (pass1 + pass2) / 2  # half the targets are old points
    return Counts(points=t_out, flops=t_out * _MERGE_FLOPS + interp,
                  exps=t_out, reads=3 * 2 * t_in, writes=3 * t_out)


def base3d_counts(grid: BD3Grid, gl_order: int = 6,
                  img_order: int = 4) -> Counts:
    lv = grid.levels[0]
    seg = lv.n_x * grid.n_t**2 * lv.n_m**2
    nodes = seg * gl_order
    sample = 2 * nodes * (2 * img_order**3 + 6 * img_order)
    sums = seg * (2 * 2 * gl_order + 2 * gl_order**2 + 3 * gl_order)
    return Counts(points=seg, flops=sample + sums, exps=nodes,
                  reads=2 * grid.n_t**3, writes=3 * seg)


def sweep3d_counts(grid: BD3Grid, *, order: int = 4, gl_order: int = 6,
                   img_order: int = 4) -> Counts:
    total = base3d_counts(grid, gl_order, img_order)
    for k in range(len(grid.levels) - 1):
        total = total + refine3d_counts(grid, k, order)
    return total


def peak_table_entries3d(grid: BD3Grid) -> int:
    return 3 * 3 * max(lv.n_x * grid.n_t**2 * lv.n_m**2 for lv in grid.levels)


# --- devices ---------------------------------------------------------------


@dataclass(frozen=True)
class Device:
    """Vendor peak figures.  Used only to turn counts into a lower bound."""

    name: str
    bandwidth: float  # GB/s
    fp64: float  # Gflop/s
    fp32: float  # Gflop/s

    def peak(self, dtype_bytes: int) -> float:
        return self.fp64 if dtype_bytes == 8 else self.fp32


#: Published peak numbers, for scale only.
DEVICES = (
    Device("CPU socket (~16 core)", bandwidth=100.0, fp64=500.0, fp32=1000.0),
    Device("RTX 4090", bandwidth=1008.0, fp64=1290.0, fp32=82600.0),
    Device("A100 80GB SXM", bandwidth=2039.0, fp64=9700.0, fp32=19500.0),
    Device("H100 SXM5", bandwidth=3350.0, fp64=34000.0, fp32=67000.0),
)


def roofline(counts: Counts, device: Device, dtype_bytes: int = 8) -> dict:
    """Time this many counts cannot beat on this device."""
    t_mem = counts.bytes_moved(dtype_bytes) / (device.bandwidth * 1e9)
    t_flop = counts.total_flops(dtype_bytes) / (device.peak(dtype_bytes) * 1e9)
    return {"t_mem": t_mem, "t_flop": t_flop, "t": max(t_mem, t_flop),
            "bound": "memory" if t_mem >= t_flop else "compute"}


def direct_counts(grid: BDGrid, n_image: int, *, families: int = 4,
                  img_order: int = 4, samples_per_h: float = 1.0) -> Counts:
    """Direct quadrature of the *same* lines, for a like-for-like roofline.

    One sample every ``h / samples_per_h`` of arc length along each top-level
    segment; at each sample the two fields are interpolated with an
    ``img_order`` tensor-product stencil, the optical depth is accumulated, and
    the integrand needs one exponential.

    The traffic charged is the two ``n_image``-square fields read once per
    family plus the output written once, i.e. every interpolation tap is
    assumed to be a cache hit.  That is the generous reading for the direct
    method -- an ``n``-square image is a few megabytes and does fit in a
    device's last-level cache -- and it is the point of the comparison: the
    direct method is arithmetic-bound, the recursion is bandwidth-bound, so the
    arithmetic saving does not convert one-for-one into time on hardware whose
    arithmetic is the cheap part.
    """
    lv = grid.top
    m = np.arange(-lv.rise, lv.rise + 1) * grid.h
    arc = np.sqrt(lv.length**2 + m**2)
    lines = grid.n_y * lv.n_m
    nodes = float(arc.sum()) * grid.n_y * samples_per_h / grid.h
    per_node = 2 * (2 * img_order**2 + 4 * img_order) + 4
    return Counts(points=families * lines,
                  flops=families * nodes * per_node,
                  exps=families * nodes,
                  reads=families * 2 * n_image**2,
                  writes=families * 3 * lines)
