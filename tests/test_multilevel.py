"""Correctness and physics of the multilevel forward transform.

Nothing here measures accuracy against a fine reference -- that is
test_accuracy.py.  These are the properties that must hold *exactly* (to
roundoff) or that encode the physics, so a regression shows up as a hard failure
rather than a slightly worse number.
"""

import numpy as np
import pytest

from tomogrid import Options, forward, phantoms as ph
from tomogrid.axis import Axis
from tomogrid.image import Image, sample_function, square_grid
from tomogrid.multilevel import default_n_levels
from tomogrid.reference import image_evaluator, richardson_transform

from conftest import rel_err

N_IMG, N_THETA, N_S = 33, 32, 33
FAST = Options(n_theta_min=8, n_levels=5, gl_order=6)

# Accuracy actually achieved by the multilevel path at these deliberately tiny
# test sizes (33^2 image, 8 base directions).  Inequalities that hold exactly in
# the continuum -- I <= S, I >= 0, monotonicity in mu -- hold only to within this,
# because the (s, t) table interpolation is a signed perturbation.
ALGO_TOL = 5e-3


def img(fun, n=N_IMG):
    xa, ya = square_grid(n)
    return sample_function(fun, xa, ya)


def run(f_fun, mu_fun, *, n_theta=N_THETA, n_s=N_S, options=FAST, n=N_IMG, **kw):
    return forward(img(f_fun, n), img(mu_fun, n), n_theta=n_theta, n_s=n_s,
                   options=options, **kw)


# --- shape and plumbing ----------------------------------------------------


def test_output_shape_and_grids():
    sg = run(ph.bump(radius=0.4), ph.plateau(r_flat=0.3, r_zero=0.55))
    assert sg.S.shape == sg.E.shape == sg.I.shape == (N_THETA, N_S)
    assert sg.thetas.shape == (N_THETA,)
    assert np.allclose(sg.thetas, 2 * np.pi * np.arange(N_THETA) / N_THETA)
    assert sg.s_axis.n == N_S
    assert np.isclose(sg.s_axis.lo, -1.0) and np.isclose(sg.s_axis.hi, 1.0)


def test_default_n_levels_puts_the_finest_segment_near_one_cell():
    for n in (33, 65, 129):
        h = 2.0 / (n - 1)
        L = default_n_levels(h, 1.0)
        assert 0.5 * h <= 2.0 / 2**L <= 2.0 * h


def test_pet_and_attenuated_accessors():
    sg = run(ph.bump(radius=0.4), ph.plateau(amp=1.0, r_flat=0.3, r_zero=0.55))
    assert np.allclose(sg.pet, np.exp(-sg.E) * sg.S)
    assert sg.attenuated is sg.I


def test_work_counters_are_populated():
    sg = run(ph.bump(radius=0.4), ph.zero())
    assert sg.work["table_points"] > 0
    assert sg.work["level0_nodes"] > 0
    assert sg.work["interp_gathers"] > 0


# --- exact properties ------------------------------------------------------


def test_zero_attenuation_makes_I_identical_to_S():
    sg = run(ph.bump(amp=1.0, center=(0.1, -0.1), radius=0.45), ph.zero())
    assert np.allclose(sg.E, 0.0, atol=1e-14)
    assert np.allclose(sg.I, sg.S, rtol=1e-12, atol=1e-14)
    assert np.allclose(sg.pet, sg.S, rtol=1e-12, atol=1e-14)


def test_zero_activity_gives_zero_signal():
    sg = run(ph.zero(), ph.plateau(amp=2.0, r_flat=0.3, r_zero=0.55))
    assert np.allclose(sg.S, 0.0, atol=1e-14)
    assert np.allclose(sg.I, 0.0, atol=1e-14)
    assert np.abs(sg.E).max() > 0.5


def test_linear_in_the_activity_at_fixed_attenuation():
    """I and S are linear in f; E does not depend on f at all."""
    mu = ph.plateau(amp=1.5, r_flat=0.3, r_zero=0.55)
    f1 = ph.bump(amp=1.0, center=(-0.2, 0.1), radius=0.25)
    f2 = ph.bump(amp=1.0, center=(0.2, -0.15), radius=0.3)
    a, b = 2.0, -0.75
    s1, s2 = run(f1, mu), run(f2, mu)
    sc = run(lambda x, y: a * f1(x, y) + b * f2(x, y), mu)
    assert np.allclose(sc.S, a * s1.S + b * s2.S, atol=1e-12)
    assert np.allclose(sc.I, a * s1.I + b * s2.I, atol=1e-12)
    assert np.allclose(sc.E, s1.E, atol=1e-12)


def test_plain_radon_parity():
    """S and E are even under (theta, s) -> (theta + pi, -s); I is not."""
    sg = run(ph.bump(amp=1.0, center=(0.15, -0.1), radius=0.35),
             ph.bump(amp=2.0, center=(-0.2, 0.15), radius=0.3))
    flip = lambda a: np.roll(a, N_THETA // 2, axis=0)[:, ::-1]
    assert rel_err(flip(sg.S), sg.S) < 1e-12
    assert rel_err(flip(sg.E), sg.E) < 1e-12
    assert rel_err(flip(sg.I), sg.I) > 1e-3  # attenuation is directional


def test_rotating_the_phantom_shifts_the_sinogram():
    """Covariance under a rotation by one angular step.

    Limited by *discretisation*, not by the algorithm: a Cartesian image grid is
    not rotation invariant, so the rotated phantom is a genuinely different
    discrete field.  The agreement improves with resolution
    (see test_accuracy.py::test_rotation_covariance_converges)."""
    alpha = 2 * np.pi / N_THETA
    f = ph.bump(amp=1.0, center=(0.25, 0.0), radius=0.25)
    mu = ph.bump(amp=1.5, center=(-0.1, 0.2), radius=0.3)

    def rot(fun):
        c, s = np.cos(alpha), np.sin(alpha)
        return lambda x, y: fun(c * x + s * y, -s * x + c * y)

    base = run(f, mu)
    turned = run(rot(f), rot(mu))
    for attr in ("S", "E", "I"):
        a, b = getattr(turned, attr), np.roll(getattr(base, attr), 1, axis=0)
        assert rel_err(a, b) < 5e-2, attr


# --- the physics -----------------------------------------------------------


def test_attenuation_downstream_of_the_source_factors_out():
    """If every emission point is upstream of all the attenuation, the exponent
    is constant along the ray and the SPECT model collapses to the PET one."""
    f = ph.bump(amp=1.0, center=(-0.35, 0.0), radius=0.15)
    mu = ph.bump(amp=3.0, center=(0.35, 0.0), radius=0.15)
    sg = run(f, mu)
    forward_dir = 0  # theta = 0, photons fly towards +x: f first, then mu
    assert rel_err(sg.I[forward_dir], sg.pet[forward_dir]) < 1e-10


def test_attenuation_upstream_of_the_source_does_nothing():
    """Same geometry seen from behind: the attenuator is already passed."""
    f = ph.bump(amp=1.0, center=(-0.35, 0.0), radius=0.15)
    mu = ph.bump(amp=3.0, center=(0.35, 0.0), radius=0.15)
    sg = run(f, mu)
    back = N_THETA // 2  # theta = pi, photons fly towards -x
    assert rel_err(sg.I[back], sg.S[back]) < 1e-10
    assert np.abs(sg.E[back]).max() > 0.5  # there *is* attenuation on the line


def test_more_attenuation_means_less_signal():
    """Monotone in mu, up to the algorithm's own (signed) interpolation error."""
    f = ph.bump(amp=1.0, center=(0.0, 0.0), radius=0.45)
    prev = None
    for amp in (0.0, 0.5, 1.5, 4.0):
        sg = run(f, ph.plateau(amp=amp, r_flat=0.35, r_zero=0.58))
        slack = ALGO_TOL * sg.S.max()
        assert np.all(sg.I <= sg.S + slack)
        if prev is not None:
            assert np.all(sg.I <= prev + slack)
        prev = sg.I


def test_signal_stays_bounded_under_extreme_attenuation():
    """E ~ 37 means the true signal is ~1e-16, i.e. far below the algorithm's
    noise floor.  What the method guarantees there is a small *absolute* error
    relative to the peak of S, not a small relative error on a quantity that has
    itself underflowed; the answer must stay finite and O(tol * peak)."""
    f = ph.bump(amp=1.0, radius=0.45)
    sg = run(f, ph.plateau(amp=40.0, r_flat=0.35, r_zero=0.58))
    assert np.all(np.isfinite(sg.I))
    assert sg.E.max() > 20.0
    floor = ALGO_TOL * sg.S.max()
    assert np.abs(sg.I).max() < floor
    assert sg.I.min() > -floor


# --- agreement with the independent reference ------------------------------


def test_direct_path_matches_the_naive_reference():
    """n_theta_min = n_theta performs zero interpolation, so this checks the
    level-0 kernel and the merge against a completely separate implementation."""
    f = ph.bump(amp=1.0, center=(0.1, -0.05), radius=0.45)
    mu = ph.plateau(amp=1.5, r_flat=0.35, r_zero=0.6)
    opt = Options(n_theta_min=N_THETA, n_levels=5, gl_order=8)
    sg = run(f, mu, options=opt, support_radius=0.6)
    fi, mi = img(f), img(mu)
    S, E, I = richardson_transform(
        image_evaluator(fi), image_evaluator(mi), sg.thetas, sg.s_axis.nodes,
        half_chord=1.0, n_samples=2049,
    )
    # Both sides evaluate the *same* piecewise-cubic interpolant, but with
    # different quadratures, and its C^0 kinks cap how closely they can agree.
    assert rel_err(sg.S, S) < 1e-4
    assert rel_err(sg.E, E) < 1e-4
    assert rel_err(sg.I, I) < 1e-4


# --- validation ------------------------------------------------------------


def test_mismatched_grids_are_rejected():
    xa, ya = square_grid(17)
    xb, yb = square_grid(21)
    f = sample_function(ph.zero(), xa, ya)
    mu = sample_function(ph.zero(), xb, yb)
    with pytest.raises(ValueError):
        forward(f, mu, n_theta=8, n_s=9, options=Options(n_theta_min=8, n_levels=3))


def test_support_larger_than_the_chord_is_rejected():
    with pytest.raises(ValueError):
        run(ph.bump(radius=0.4), ph.zero(), support_radius=1.5)


def test_support_wider_than_the_s_grid_is_rejected():
    s_axis = Axis.from_endpoints(-0.3, 0.3, 9)
    with pytest.raises(ValueError):
        forward(img(ph.bump(radius=0.4)), img(ph.zero()), n_theta=8,
                s_axis=s_axis, options=Options(n_theta_min=8, n_levels=3),
                support_radius=0.8)


def test_bad_oversampling_is_rejected():
    with pytest.raises(ValueError):
        run(ph.zero(), ph.zero(), options=Options(n_theta_min=8, n_levels=3, s_oversample=0))
    with pytest.raises(ValueError):
        run(ph.zero(), ph.zero(), options=Options(n_theta_min=8, n_levels=3, t_oversample=3))


def test_anisotropic_image_grid_is_rejected():
    f = Image(np.zeros((9, 11)), Axis.from_endpoints(-1, 1, 9), Axis.from_endpoints(-1, 1, 11))
    with pytest.raises(ValueError):
        forward(f, f, n_theta=8, n_s=9, options=Options(n_theta_min=8))


def test_chunking_is_numerically_invisible():
    """max_block only bounds working-array size; it must not change the answer."""
    f = ph.bump(amp=1.0, center=(0.1, -0.05), radius=0.4)
    mu = ph.plateau(amp=1.5, r_flat=0.3, r_zero=0.55)
    ref = run(f, mu, options=Options(n_theta_min=8, n_levels=5, gl_order=6))
    for block in (1 << 10, 1 << 14, 1 << 30):
        sg = run(f, mu, options=Options(n_theta_min=8, n_levels=5, gl_order=6,
                                        max_block=block))
        for attr in ("S", "E", "I"):
            assert np.array_equal(getattr(sg, attr), getattr(ref, attr)), (block, attr)
