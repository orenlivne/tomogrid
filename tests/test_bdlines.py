"""Brandt & Dym reproduction: the ingredients, and the published behaviour."""

import numpy as np
import pytest

from bdlines import build_mesh, operation_count, optimal_order, true_integral
from bdlines.discrete import _spline_second_derivatives, discrete_integral
from bdlines.experiments import run_case, sample, sinusoid
from bdlines.recursion import base_level, double, run_recursion

from conftest import order_of


# --- discretisation --------------------------------------------------------


@pytest.mark.parametrize("kind", ["linear", "cubic", "spline"])
def test_discretisation_exact_on_a_linear_field(kind):
    g = np.tile(np.arange(64.0)[:, None], (1, 64))
    v = discrete_integral(g, np.asarray(8), np.asarray(20.0), 8, np.asarray(4.0), kind)
    assert np.isclose(v, 22.0)  # mean of rows 20..24


@pytest.mark.parametrize("kind", ["linear", "cubic", "spline"])
def test_discretisation_reproduces_a_constant(kind):
    g = np.full((64, 64), 3.5)
    v = discrete_integral(g, np.asarray(8), np.asarray(20.0), 8, np.asarray(3.0), kind)
    assert np.isclose(v, 3.5)


def test_spline_beats_cubic_beats_linear_on_smooth_data():
    g = sample(4, 129, 129)
    fun = sinusoid(4)
    rise = np.linspace(0.0, 8.0, 9)
    truth = true_integral(fun, 8, 60.0, 8, rise)
    errs = {k: np.abs(discrete_integral(g, np.asarray(8), np.asarray(60.0), 8,
                                        rise, k) - truth).max()
            for k in ("linear", "cubic", "spline")}
    assert errs["cubic"] < errs["linear"]
    assert errs["spline"] < errs["linear"]


# --- the hierarchy ---------------------------------------------------------


def test_table_size_is_independent_of_length():
    mesh = build_mesh(257, 257, sigma=1, max_length=128)
    sizes = [mesh.table_size(lv) for lv in mesh.levels]
    assert max(sizes) / min(sizes) < 1.3, sizes


def test_slope_count_doubles_and_positions_halve():
    mesh = build_mesh(257, 257, sigma=1, max_length=128)
    for a, b in zip(mesh.levels, mesh.levels[1:]):
        assert b.n_slope == 2 * a.n_slope - 1
        assert b.n_x == a.n_x // 2
        assert b.length == 2 * a.length


def test_slopes_span_the_shallow_family():
    mesh = build_mesh(129, 129, sigma=2, max_length=64)
    for lv in mesh.levels:
        s = mesh.slopes(lv)
        assert np.isclose(s.min(), -1.0) and np.isclose(s.max(), 1.0)


@pytest.mark.parametrize("bad", [dict(sigma=0), dict(length0=3),
                                 dict(max_length=1), dict(max_length=100)])
def test_invalid_meshes_rejected(bad):
    kw = dict(sigma=1, length0=2, max_length=64)
    kw.update(bad)
    with pytest.raises(ValueError):
        build_mesh(129, 129, **kw)


# --- the recursion ---------------------------------------------------------


def test_preexisting_directions_are_reproduced_exactly():
    """A direction present at the base level is merged, never interpolated, and
    the normalised trapezoid composes exactly -- so it must agree with the
    direct discretisation to roundoff."""
    n = 129
    g = sample(4, n, n)
    sm = _spline_second_derivatives(g)
    mesh = build_mesh(n, n, sigma=1, max_length=64)
    tables = run_recursion(g, mesh, disc_kind="spline", order=2, keep_all=True)
    y0 = 64.0
    jy = int(round(y0 / mesh.h))
    for lv, table in zip(mesh.levels, tables):
        step = lv.length // mesh.levels[0].length
        m = np.arange(0, lv.rise + 1, max(1, step))
        got = table[0, jy, m + lv.rise]
        want = discrete_integral(g, np.asarray(0), np.asarray(y0), lv.length,
                                 m * mesh.h, "spline", sm)
        assert np.abs(got - want).max() < 1e-12, lv.length


def test_constant_data_is_reproduced_exactly_at_every_level():
    n = 129
    g = np.full((n, n), 2.25)
    mesh = build_mesh(n, n, sigma=1, max_length=32)
    tables = run_recursion(g, mesh, disc_kind="cubic", order=4, keep_all=True)
    for lv, table in zip(mesh.levels, tables):
        interior = table[:, lv.rise + 2:mesh.n_y - lv.rise - 2, :]
        assert np.abs(interior - 2.25).max() < 1e-12, lv.length


@pytest.mark.parametrize("order,expect", [(2, 2.0), (4, 4.0)])
def test_error_falls_as_h_to_the_order(order, expect):
    """The paper's central claim: halving the integration mesh reduces the
    approximation error by 2^p."""
    errs = [run_case(4, s, order=order)[32]["approx_mean"] for s in (1, 2, 4)]
    obs = order_of(errs)
    assert obs.min() > expect - 0.6, f"order {order}: observed {obs} from {errs}"


def test_cubic_doubling_beats_linear():
    lin = run_case(2, 1, order=2)[32]["approx_mean"]
    cub = run_case(2, 1, order=4)[32]["approx_mean"]
    assert cub < 0.2 * lin, (lin, cub)


def test_approximation_can_be_driven_below_discretisation_error():
    """The design criterion of the paper: the approximation need only be as
    accurate as the discretisation it approximates."""
    r = run_case(4, 1, order=4)[32]
    assert r["approx_mean"] < r["disc_mean"]


def test_double_rejects_mismatched_table():
    mesh = build_mesh(129, 129, sigma=1, max_length=32)
    with pytest.raises(ValueError):
        double(np.zeros((3, 4, 5)), mesh, 0, 2)


# --- work ------------------------------------------------------------------


def test_operation_count_grows_like_n_log_n():
    """Doubling the image should multiply the work by a little more than four."""
    counts = []
    for q in (6, 7, 8, 9):
        n = 2**q + 1
        mesh = build_mesh(n, n, sigma=1, max_length=2**q)
        counts.append(operation_count(mesh, 2)["ops"])
    ratios = [b / a for a, b in zip(counts, counts[1:])]
    assert all(4.0 < r < 5.6 for r in ratios), ratios


def test_work_is_quadratic_in_the_mesh_refinement():
    """W ~ sigma^2, against an error ~ sigma^-p -- the trade the paper optimises."""
    base = operation_count(build_mesh(129, 129, sigma=1, max_length=64), 2)["ops"]
    for sigma in (2, 4):
        got = operation_count(build_mesh(129, 129, sigma=sigma, max_length=64),
                              2)["ops"]
        assert 0.8 * sigma**2 < got / base < 1.25 * sigma**2, sigma


def test_optimal_order_follows_the_paper():
    assert optimal_order(2**8) == max(2, round(2 * np.log(8)))
    assert optimal_order(2**16) >= optimal_order(2**8)
