"""The reference evaluators are themselves checked against closed form.

The trapezoidal rule is *spectrally* accurate on a smooth integrand that decays
to nothing inside the chord (every Euler-Maclaurin boundary term vanishes), so
on analytic phantoms :func:`naive_transform` reaches machine precision and is a
genuine ground truth.  It drops to second order on the piecewise-polynomial grid
interpolant, which is where :func:`richardson_transform` earns its keep.
"""

import numpy as np
import pytest

from tomogrid import phantoms as ph
from tomogrid.image import sample_function, square_grid
from tomogrid.reference import (
    analytic_gaussian,
    image_evaluator,
    naive_transform,
    richardson_transform,
)

from conftest import order_of, rel_err

THETAS = np.linspace(0.0, 2 * np.pi, 9)[:-1]
S_NODES = np.linspace(-0.6, 0.6, 13)
SIGMA, AMP, CENTER, H = 0.1, 1.3, (0.2, -0.15), 1.0


def gaussian_case(mu_const, sigma=SIGMA, center=CENTER):
    f = ph.gaussian(amp=AMP, center=center, sigma=sigma)
    mu = ph.constant(mu_const)
    want = analytic_gaussian(
        THETAS, S_NODES, amp=AMP, center=center, sigma=sigma,
        mu_const=mu_const, half_chord=H,
    )
    return f, mu, want


@pytest.mark.parametrize("mu_const", [0.0, 0.8, 4.0])
def test_naive_matches_closed_form_to_machine_precision(mu_const):
    f, mu, (wS, wE, wI) = gaussian_case(mu_const)
    S, E, I = naive_transform(f, mu, THETAS, S_NODES, H, n_samples=257)
    assert np.allclose(E, wE, rtol=1e-12)
    assert rel_err(S, wS) < 1e-12
    assert rel_err(I, wI) < 1e-11


def test_trapezoid_is_spectral_on_a_resolved_gaussian():
    """Halving the step must crush the error far faster than any fixed order."""
    f, mu, (_, _, wI) = gaussian_case(0.5, sigma=0.03, center=(0.0, 0.0))
    errs = [rel_err(naive_transform(f, mu, THETAS, S_NODES, H, n_samples=n)[2], wI)
            for n in (33, 65, 129)]
    assert order_of(errs).min() > 8.0, f"observed {order_of(errs)}"
    assert errs[-1] < 1e-13


def test_naive_drops_to_second_order_on_the_grid_interpolant():
    """The cubic Lagrange interpolant is only C^0, so its kinks cap the rule."""
    xa, ya = square_grid(65)
    f = sample_function(ph.bump(amp=1.0, center=(0.05, 0.0), radius=0.45), xa, ya)
    mu = sample_function(ph.plateau(amp=1.0, r_flat=0.3, r_zero=0.55), xa, ya)
    fe, me = image_evaluator(f), image_evaluator(mu)
    best = naive_transform(fe, me, THETAS, S_NODES, H, n_samples=16385)[2]
    errs = [rel_err(naive_transform(fe, me, THETAS, S_NODES, H, n_samples=n)[2], best)
            for n in (257, 513, 1025)]
    obs = order_of(errs)
    assert 1.5 < obs.min() < 3.5, f"expected ~2nd order, got {obs}"
    rich = richardson_transform(fe, me, THETAS, S_NODES, H, n_samples=513)[2]
    assert rel_err(rich, best) < errs[1]


def test_S_matches_the_analytic_radon_of_a_gaussian():
    f, mu, (wS, _, _) = gaussian_case(0.0)
    S, _, I = naive_transform(f, mu, THETAS, S_NODES, H, n_samples=257)
    assert rel_err(S, wS) < 1e-12
    assert rel_err(I, wS) < 1e-12  # no attenuation => I == S


def test_constant_attenuation_gives_constant_E():
    _, mu, _ = gaussian_case(1.7)
    _, E, _ = naive_transform(ph.zero(), mu, THETAS, S_NODES, H, n_samples=65)
    assert np.allclose(E, 1.7 * 2 * H, rtol=1e-12)


def test_attenuation_only_ever_reduces_the_signal():
    f = ph.bump(amp=1.0, radius=0.5)
    S, _, I = naive_transform(f, ph.constant(2.0), THETAS, S_NODES, H, n_samples=513)
    assert np.all(I <= S + 1e-12)
    lit = S > 1e-9
    assert lit.any() and np.all(I[lit] > 0)


def test_image_evaluator_agrees_with_direct_sampling():
    xa, ya = square_grid(33)
    img = sample_function(ph.bump(amp=1.0, radius=0.5), xa, ya)
    ev = image_evaluator(img)
    x = np.array([0.1, -0.2, 0.7])
    assert np.allclose(ev(x, x), img.sample(x, x))


def test_analytic_gaussian_shapes():
    S, E, I = analytic_gaussian(
        THETAS, S_NODES, amp=1.0, center=(0, 0), sigma=0.2, mu_const=0.5, half_chord=1.0
    )
    assert S.shape == E.shape == I.shape == (THETAS.size, S_NODES.size)
