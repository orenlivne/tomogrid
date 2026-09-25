import numpy as np
import pytest

from tomogrid.axis import Axis
from tomogrid.interp import (
    interp1d,
    interp2d,
    interp2d_stacked,
    lagrange_weights,
    periodic_midpoint_weights,
    stencil,
)

from conftest import order_of


@pytest.mark.parametrize("order", [1, 2, 3, 4, 6, 8])
def test_weights_are_a_partition_of_unity(order):
    u = np.linspace(-1.0, order + 1.0, 37)
    assert np.allclose(lagrange_weights(u, order).sum(axis=0), 1.0)


@pytest.mark.parametrize("order", [2, 3, 4, 6])
def test_weights_reproduce_nodes(order):
    """At node k the weights must be the k-th unit vector."""
    for k in range(order):
        w = lagrange_weights(np.asarray(float(k)), order)
        expect = np.zeros(order)
        expect[k] = 1.0
        assert np.allclose(w.ravel(), expect, atol=1e-12)


def test_midpoint_weights_match_textbook_values():
    assert np.allclose(periodic_midpoint_weights(2).ravel(), [0.5, 0.5])
    assert np.allclose(
        periodic_midpoint_weights(4).ravel(), [-1 / 16, 9 / 16, 9 / 16, -1 / 16]
    )


@pytest.mark.parametrize("order", [2, 3, 4, 6])
def test_exact_for_polynomials_below_order(order):
    ax = Axis.from_endpoints(-2.0, 3.0, 41)
    x = np.linspace(-1.0, 2.0, 23)
    for deg in range(order):
        vals = ax.nodes**deg
        assert np.allclose(interp1d(vals, ax, x, order), x**deg, atol=1e-9)


@pytest.mark.parametrize("order", [2, 4, 6])
def test_convergence_order(order):
    fun = lambda z: np.sin(1.7 * z) * np.exp(0.3 * z)
    x = np.linspace(-0.9, 0.9, 17) + 0.013  # deliberately off-node
    errs = []
    for n in (21, 41, 81, 161):
        ax = Axis.from_endpoints(-2.0, 2.0, n)
        errs.append(np.abs(interp1d(fun(ax.nodes), ax, x, order) - fun(x)).max())
    obs = order_of(errs)
    assert obs.min() > order - 0.45, f"order {order}: observed {obs}"


def test_zero_fill_outside_grid():
    ax = Axis.from_endpoints(0.0, 1.0, 11)
    vals = np.ones(11)
    far = np.array([-5.0, 5.0])
    assert np.allclose(interp1d(vals, ax, far, 4), 0.0)


def test_stencil_clamps_indices_and_zeroes_weights():
    ax = Axis.from_endpoints(0.0, 1.0, 11)
    idx, w = stencil(np.array([-0.5, 0.5, 1.5]), ax, 4)
    assert idx.min() >= 0 and idx.max() <= 10
    assert np.all(w[:, 0] == 0.0) and np.all(w[:, 2] == 0.0)
    assert not np.all(w[:, 1] == 0.0)


def test_interp2d_exact_on_tensor_polynomials():
    ax = Axis.from_endpoints(-1.0, 1.0, 17)
    ay = Axis.from_endpoints(-2.0, 2.0, 21)
    xx, yy = ax.nodes[:, None], ay.nodes[None, :]
    vals = (1 + xx + xx**2) * (2 - yy + 0.5 * yy**3)
    px = np.linspace(-0.8, 0.8, 11)
    py = np.linspace(-1.5, 1.5, 11)
    got = interp2d(vals, ax, ay, px[:, None], py[None, :], 4, 4)
    want = (1 + px[:, None] + px[:, None] ** 2) * (2 - py[None, :] + 0.5 * py[None, :] ** 3)
    assert np.allclose(got, want, atol=1e-9)


def test_interp2d_rejects_wrong_shape():
    ax = Axis.from_endpoints(0.0, 1.0, 5)
    with pytest.raises(ValueError):
        interp2d(np.zeros((4, 5)), ax, ax, np.array(0.5), np.array(0.5), 2, 2)


def test_interp2d_stacked_selects_the_right_slab():
    ax = Axis.from_endpoints(0.0, 1.0, 9)
    tables = np.stack([np.full((9, 9), float(k)) for k in range(4)])
    lead = np.array([0, 2, 3])[:, None]
    x = np.full((3, 1), 0.44)
    got = interp2d_stacked(tables, ax, ax, lead, x, x, 4, 4)
    assert np.allclose(got.ravel(), [0.0, 2.0, 3.0])
