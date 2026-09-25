"""Structural guarantees of the level hierarchy.

These are the invariants the exact merge depends on; if any of them breaks the
method silently loses its accuracy rather than failing loudly.
"""

import numpy as np
import pytest

from tomogrid.axis import Axis
from tomogrid.hierarchy import build_hierarchy, thetas_of

S_AXIS = Axis.from_endpoints(-1.0, 1.0, 33)


def make(n_theta_out=64, n_theta_min=16, n_levels=5, nu=2, s_axis=S_AXIS):
    return build_hierarchy(
        n_theta_out=n_theta_out,
        n_theta_min=n_theta_min,
        n_levels=n_levels,
        s_axis=s_axis,
        t_oversample=nu,
    )


@pytest.mark.parametrize("n", [4, 8, 16, 64])
def test_angular_grids_nest_by_doubling(n):
    assert np.allclose(thetas_of(2 * n)[0::2], thetas_of(n))


def test_angular_grid_is_periodic_and_uniform():
    th = thetas_of(8)
    assert np.allclose(np.diff(th), 2 * np.pi / 8)
    assert th[0] == 0.0 and th[-1] < 2 * np.pi


@pytest.mark.parametrize("nu", [2, 4])
def test_t_grids_nest_across_levels(nu):
    h = make(nu=nu)
    for l in range(1, h.n_levels + 1):
        fine, coarse = h.levels[l - 1].t_axis, h.levels[l].t_axis
        assert np.allclose(fine.nodes[0::2], coarse.nodes)


@pytest.mark.parametrize("nu", [2, 4])
def test_half_segment_centres_land_on_the_finer_grid(nu):
    """The reason merging needs no t-interpolation (docs sec. 4, step B)."""
    h = make(nu=nu)
    for l in range(1, h.n_levels + 1):
        fine, coarse = h.levels[l - 1], h.levels[l]
        for t in coarse.t_axis.nodes:
            for centre in (t - coarse.seg_len / 4, t + coarse.seg_len / 4):
                offset = (centre - fine.t_axis.origin) / fine.t_axis.h
                assert np.isclose(offset, round(offset), atol=1e-9), (l, t, centre)


def test_segment_lengths_double_and_cover_the_chord():
    h = make()
    for l in range(1, h.n_levels + 1):
        assert np.isclose(h.levels[l].seg_len, 2 * h.levels[l - 1].seg_len)
    assert np.isclose(h.top.seg_len, 2 * h.half_chord)
    assert h.top.t_axis.n == h.t_oversample + 1
    assert np.isclose(h.top.t_axis.nodes[h.t_oversample // 2], 0.0)


def test_t_grids_span_the_chord_exactly():
    h = make()
    for lv in h.levels:
        assert np.isclose(lv.t_axis.lo, -h.half_chord)
        assert np.isclose(lv.t_axis.hi, h.half_chord)


def test_refinement_schedule_is_bottom_loaded():
    """Angles double at the *short-segment* levels, then coast -- eq. (7)."""
    h = make(n_theta_out=64, n_theta_min=8, n_levels=6)
    counts = [lv.n_theta for lv in h.levels]
    assert counts == [8, 16, 32, 64, 64, 64, 64]
    assert h.refine_levels == (1, 2, 3)


def test_work_is_flat_while_refining_then_decays():
    h = make(n_theta_out=64, n_theta_min=8, n_levels=6)
    sizes = [lv.n_theta * lv.t_axis.n for lv in h.levels]
    refining = sizes[: len(h.refine_levels) + 1]
    assert max(refining) / min(refining) < 1.3, sizes
    for a, b in zip(sizes[len(h.refine_levels) + 1 :], sizes[len(h.refine_levels) + 2 :]):
        assert b < a


def test_angular_displacement_is_level_independent():
    """Eq. (8): every refinement faces the same criterion."""
    h = make(n_theta_out=64, n_theta_min=8, n_levels=6)
    vals = [
        (2 * np.pi / h.levels[i - 1].n_theta) * h.levels[i - 1].seg_len
        for i in h.refine_levels
    ]
    assert np.allclose(vals, vals[0])
    assert np.isclose(h.max_angular_displacement(), vals[0])
    assert np.isclose(h.angular_criterion(0.5), vals[0] / 0.5)


def test_no_refinement_when_min_equals_out():
    h = make(n_theta_out=32, n_theta_min=32)
    assert h.refine_levels == ()
    assert h.max_angular_displacement() == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(n_theta_out=48, n_theta_min=16),  # ratio not a power of two
        dict(n_theta_out=8, n_theta_min=16),  # out < min
        dict(n_theta_out=1024, n_theta_min=16, n_levels=3),  # cannot reach out
        dict(n_levels=-1),
        dict(nu=3),  # odd oversampling
        dict(nu=1),
        dict(n_theta_out=0),
    ],
)
def test_invalid_configurations_are_rejected(kwargs):
    with pytest.raises(ValueError):
        make(**kwargs)


def test_describe_mentions_every_level():
    text = make().describe()
    for l in range(6):
        assert f"L{l}:" in text
