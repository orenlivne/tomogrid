"""Cost: O(N log N) for the multilevel schedule vs O(N^1.5) for direct evaluation.

Asserted from operation counts rather than wall-clock, so the tests are
deterministic and can reach sizes that would take hours to actually run.
"""

import numpy as np
import pytest

from tomogrid import Options, build_hierarchy, estimate_work, forward, phantoms as ph
from tomogrid.axis import Axis
from tomogrid.image import sample_function, square_grid
from tomogrid.multilevel import default_n_levels

ORDERS = dict(gl_order=8, img_order=4, theta_order=4, s_order=4, t_order=4)


def work_for(n_grid, n_theta, n_theta_min, nu=2, sigma=2):
    h_img = 2.0 / (n_grid - 1)
    n_levels = default_n_levels(h_img, 1.0)
    s_axis = Axis.from_endpoints(-1.0, 1.0, sigma * (n_grid - 1) + 1)
    hier = build_hierarchy(
        n_theta_out=n_theta, n_theta_min=n_theta_min,
        n_levels=n_levels, s_axis=s_axis, t_oversample=nu,
    )
    return estimate_work(hier, **ORDERS)


def test_cost_model_matches_the_real_counters():
    n, n_theta, n_s = 33, 32, 33
    opt = Options(n_theta_min=8, n_levels=5, gl_order=6, s_oversample=2, t_oversample=2)
    xa, ya = square_grid(n)
    sg = forward(sample_function(ph.bump(radius=0.4), xa, ya),
                 sample_function(ph.zero(), xa, ya),
                 n_theta=n_theta, n_s=n_s, options=opt)
    want = estimate_work(sg.hierarchy, gl_order=opt.gl_order, img_order=opt.img_order,
                         theta_order=opt.theta_order, s_order=opt.s_order,
                         t_order=opt.t_order)
    for key in ("level0_nodes", "table_points", "interp_targets", "interp_gathers"):
        assert sg.work[key] == want[key], key


def test_table_size_is_flat_while_refining_then_decays():
    hier = build_hierarchy(n_theta_out=256, n_theta_min=16, n_levels=8,
                           s_axis=Axis.from_endpoints(-1, 1, 129), t_oversample=2)
    sizes = [lv.n_theta * lv.t_axis.n for lv in hier.levels]
    n_ref = len(hier.refine_levels)
    flat = sizes[: n_ref + 1]
    assert max(flat) / min(flat) < 1.3, sizes
    assert sizes[-1] < 0.2 * sizes[n_ref]


@pytest.mark.parametrize("n_theta_min", [8, 16, 32])
def test_multilevel_is_N_log_N(n_theta_min):
    """Doubling the grid must multiply work by ~4 (times a log), never by ~8."""
    ratios = []
    prev = None
    for n_grid in (65, 129, 257, 513, 1025):
        w = work_for(n_grid, n_theta=4 * (n_grid - 1), n_theta_min=n_theta_min)["gathers"]
        if prev is not None:
            ratios.append(w / prev)
        prev = w
    ratios = np.array(ratios)
    assert np.all(ratios < 6.0), f"N^1.5 would give ~8: {ratios}"
    assert np.all(ratios > 4.0), f"must grow at least like N: {ratios}"


def test_direct_evaluation_is_N_to_the_three_halves():
    ratios = []
    prev = None
    for n_grid in (65, 129, 257, 513):
        n_theta = 4 * (n_grid - 1)
        w = work_for(n_grid, n_theta=n_theta, n_theta_min=n_theta)["gathers"]
        if prev is not None:
            ratios.append(w / prev)
        prev = w
    assert np.all(np.array(ratios) > 7.5), f"expected ~8, got {ratios}"


def test_multilevel_beats_direct_and_the_gap_widens():
    gaps = []
    for n_grid in (129, 257, 513, 1025):
        n_theta = 4 * (n_grid - 1)
        ml = work_for(n_grid, n_theta, n_theta_min=16)["gathers"]
        direct = work_for(n_grid, n_theta, n_theta_min=n_theta)["gathers"]
        gaps.append(direct / ml)
    assert all(b > a for a, b in zip(gaps, gaps[1:])), gaps
    assert gaps[0] > 1.0 and gaps[-1] > 8.0, gaps


def test_cost_grows_at_most_linearly_in_n_theta_min():
    """The accuracy knob is paid for at most linearly -- in fact sublinearly,
    because raising n_theta_min also *removes* refinement levels (the schedule
    saturates at n_theta_out sooner)."""
    base = work_for(257, n_theta=1024, n_theta_min=8)["gathers"]
    ratios = []
    for factor in (2, 4, 8):
        got = work_for(257, n_theta=1024, n_theta_min=8 * factor)["gathers"]
        ratios.append(got / base)
        assert 1.0 < got / base <= 1.1 * factor, (factor, got / base)
    # sublinear: the last doubling buys less than a factor of two
    assert ratios[-1] / ratios[-2] < 2.0, ratios


@pytest.mark.parametrize("field,factor", [("s_oversample", 2), ("t_oversample", 2)])
def test_oversampling_costs_what_it_should(field, factor):
    kw = dict(nu=2, sigma=2)
    base = work_for(129, 512, 16, **kw)["gathers"]
    kw[{"s_oversample": "sigma", "t_oversample": "nu"}[field]] *= factor
    got = work_for(129, 512, 16, **kw)["gathers"]
    assert 0.8 * factor < got / base < 1.25 * factor


def test_refining_angles_alone_does_not_blow_up_the_cost():
    """Above n_theta_min the extra levels only coast, so work grows sublinearly
    in n_theta_out."""
    w = [work_for(129, n_theta=nt, n_theta_min=16)["gathers"] for nt in (64, 128, 256, 512)]
    for a, b in zip(w, w[1:]):
        assert 1.0 <= b / a < 1.5, w
