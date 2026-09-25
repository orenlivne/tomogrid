"""Accuracy of the multilevel transform, over a battery of attenuation fields.

Two errors are kept apart throughout:

*algorithmic*  multilevel vs the same code run with ``n_theta_min = n_theta``,
               which performs no interpolation at all.  This isolates what the
               multilevel schedule costs in accuracy.
*total*        multilevel vs the continuum, i.e. algorithmic + discretisation.

Sizes here are deliberately small (33^2 image, 8 base directions) so the suite
stays quick; the errors are correspondingly generous.  The refinement tests are
the ones that show the method is actually convergent.
"""

import numpy as np
import pytest

from tomogrid import Options, forward, phantoms as ph
from tomogrid.image import sample_function, square_grid
from tomogrid.reference import analytic_gaussian, naive_transform

from conftest import order_of, rel_err

N_IMG, N_THETA, N_S, N_LEV = 33, 32, 33, 5
ML = Options(n_theta_min=8, n_levels=N_LEV, gl_order=6)
DIRECT = Options(n_theta_min=N_THETA, n_levels=N_LEV, gl_order=6)

# Measured worst case over all 36 phantom pairs at these sizes is 3.0e-3.
BATTERY_TOL = 5e-3


def images(f_fun, mu_fun, n=N_IMG):
    xa, ya = square_grid(n)
    return sample_function(f_fun, xa, ya), sample_function(mu_fun, xa, ya)


def pair(f_fun, mu_fun, support, n=N_IMG, n_theta=N_THETA, ml=ML, direct=DIRECT,
         n_s=None):
    fi, mi = images(f_fun, mu_fun, n)
    kw = dict(n_theta=n_theta, n_s=N_S if n_s is None else n_s, support_radius=support)
    return forward(fi, mi, options=ml, **kw), forward(fi, mi, options=direct, **kw)


def algorithmic_error(got, ref):
    """Errors in I are scaled by the peak of S -- the physically meaningful
    scale, since I itself underflows on heavily attenuated rays."""
    scale = np.abs(ref.S).max()
    return {
        "S": rel_err(got.S, ref.S),
        "E": rel_err(got.E, ref.E),
        "I": float(np.abs(got.I - ref.I).max() / scale),
    }


# --- the battery -----------------------------------------------------------


@pytest.mark.parametrize("mu_name", sorted(ph.ATTENUATIONS))
def test_every_attenuation_field(mu_name):
    mu, mu_r, _ = ph.ATTENUATIONS[mu_name]
    f, f_r = ph.ACTIVITIES["three_blobs"][:2]
    ml, direct = pair(f, mu, max(f_r, mu_r))
    err = algorithmic_error(ml, direct)
    assert max(err.values()) < BATTERY_TOL, (mu_name, err)


@pytest.mark.parametrize("f_name", sorted(ph.ACTIVITIES))
def test_every_activity_field(f_name):
    f, f_r = ph.ACTIVITIES[f_name][:2]
    mu, mu_r, _ = ph.ATTENUATIONS["high_contrast"]
    ml, direct = pair(f, mu, max(f_r, mu_r))
    err = algorithmic_error(ml, direct)
    assert max(err.values()) < BATTERY_TOL, (f_name, err)


@pytest.mark.slow
@pytest.mark.parametrize("case", ph.battery(), ids=lambda c: c.name)
def test_full_battery(case):
    ml, direct = pair(case.f, case.mu, case.support_radius)
    err = algorithmic_error(ml, direct)
    assert max(err.values()) < BATTERY_TOL, (case.name, err)


def test_strong_attenuation_does_not_degrade_accuracy():
    """The merge damps errors from upstream by exp(-E) <= 1, so a heavily
    attenuating phantom is *easier*, not harder."""
    f, f_r = ph.ACTIVITIES["three_blobs"][:2]
    errs = {}
    for name in ("zero", "very_strong"):
        mu, mu_r, _ = ph.ATTENUATIONS[name]
        ml, direct = pair(f, mu, max(f_r, mu_r))
        errs[name] = algorithmic_error(ml, direct)["I"]
    assert errs["very_strong"] < errs["zero"], errs


# --- the knobs -------------------------------------------------------------


def test_error_falls_when_the_base_angular_grid_is_refined():
    f, f_r = ph.ACTIVITIES["offcenter_bump"][:2]
    mu, mu_r, _ = ph.ATTENUATIONS["two_bumps"]
    support = max(f_r, mu_r)
    errs = []
    for n_min in (2, 4, 8):
        ml = Options(n_theta_min=n_min, n_levels=N_LEV, gl_order=6)
        a, b = pair(f, mu, support, ml=ml)
        errs.append(algorithmic_error(a, b)["I"])
    assert errs[0] > errs[1] > errs[2], errs


def _oversampling_error(sigma, nu):
    f, f_r = ph.ACTIVITIES["offcenter_bump"][:2]
    mu, mu_r, _ = ph.ATTENUATIONS["single_bump"]
    kw = dict(s_oversample=sigma, t_oversample=nu)
    ml = Options(n_theta_min=8, n_levels=N_LEV, gl_order=6, **kw)
    direct = Options(n_theta_min=N_THETA, n_levels=N_LEV, gl_order=6, **kw)
    a, b = pair(f, mu, max(f_r, mu_r), ml=ml, direct=direct)
    return algorithmic_error(a, b)["I"]


def test_transverse_oversampling_is_the_main_table_knob():
    """Resampling the tables in (s,t) when the frame rotates is the accuracy
    bottleneck -- not the angular interpolation the method is *about*.  Halving
    ds repeatedly pays off steeply."""
    errs = [_oversampling_error(sigma, 2) for sigma in (1, 2, 4)]
    assert errs[0] > 4.0 * errs[1], errs
    # The second halving buys less because the angular term (n_theta_min = 8)
    # is becoming the floor -- raise n_theta_min too to keep going.
    assert errs[1] > 1.5 * errs[2], errs


def test_along_ray_oversampling_saturates_at_two():
    """nu = 2 puts the table's Nyquist frequency exactly on the first zero of
    the box-average transfer function sinc(k*l/2); beyond that there is nothing
    left to resolve, and the measured error stops improving."""
    at_2 = _oversampling_error(4, 2)
    for nu in (4, 8):
        assert _oversampling_error(4, nu) > 0.8 * at_2, nu


def test_along_ray_undersampling_hurts():
    """The other side of the same coin: nu = 2 is a real requirement, not a
    free parameter -- it is the smallest even value, and the structure forbids
    odd ones (see test_segment.py)."""
    fine = _oversampling_error(4, 2)
    assert fine < _oversampling_error(1, 2)


# --- convergence -----------------------------------------------------------


@pytest.mark.slow
def test_algorithmic_error_converges_under_grid_refinement():
    """The decisive property: at *fixed* oversampling and fixed base angular
    grid, refining the image drives the multilevel error to zero."""
    f, f_r = ph.ACTIVITIES["offcenter_bump"][:2]
    mu, mu_r, _ = ph.ATTENUATIONS["single_bump"]
    support = max(f_r, mu_r)
    errs = []
    for n, lev in ((17, 4), (33, 5), (65, 6)):
        ml = Options(n_theta_min=8, n_levels=lev, gl_order=6)
        direct = Options(n_theta_min=N_THETA, n_levels=lev, gl_order=6)
        # n_s tracks the image: ds must shrink with h, or the transverse
        # sampling degrades as the grid refines and the error stops falling.
        a, b = pair(f, mu, support, n=n, ml=ml, direct=direct, n_s=n)
        errs.append(algorithmic_error(a, b)["I"])
    obs = order_of(errs)
    assert obs.min() > 1.2, f"observed orders {obs} from {errs}"


@pytest.mark.slow
def test_total_error_converges_to_the_continuum():
    """Against a machine-precision reference computed from the analytic fields,
    so this includes the discretisation error as well."""
    f_fun = ph.bump(amp=1.0, center=(0.1, -0.05), radius=0.45)
    mu_fun = ph.plateau(amp=1.5, r_flat=0.35, r_zero=0.6)
    errs = []
    for n, lev in ((17, 4), (33, 5), (65, 6)):
        opt = Options(n_theta_min=8, n_levels=lev, gl_order=8)
        fi, mi = images(f_fun, mu_fun, n)
        sg = forward(fi, mi, n_theta=N_THETA, n_s=n, options=opt, support_radius=0.6)
        _, _, want = naive_transform(
            f_fun, mu_fun, sg.thetas, sg.s_axis.nodes, half_chord=1.0, n_samples=4097
        )
        errs.append(rel_err(sg.I, want))
    assert order_of(errs).min() > 1.2, f"{order_of(errs)} from {errs}"
    assert errs[-1] < 5e-3


@pytest.mark.slow
def test_rotation_covariance_converges():
    """The residual in test_multilevel is discretisation, so it must vanish
    with the mesh."""
    alpha = 2 * np.pi / N_THETA
    f = ph.bump(amp=1.0, center=(0.25, 0.0), radius=0.25)
    mu = ph.bump(amp=1.5, center=(-0.1, 0.2), radius=0.3)

    def rot(fun):
        c, s = np.cos(alpha), np.sin(alpha)
        return lambda x, y: fun(c * x + s * y, -s * x + c * y)

    errs = []
    for n, lev in ((17, 4), (33, 5), (65, 6)):
        opt = Options(n_theta_min=8, n_levels=lev, gl_order=6)
        kw = dict(n_theta=N_THETA, n_s=n, options=opt, support_radius=0.6)
        base = forward(*images(f, mu, n), **kw)
        turned = forward(*images(rot(f), rot(mu), n), **kw)
        errs.append(rel_err(turned.I, np.roll(base.I, 1, axis=0)))
    assert order_of(errs).min() > 1.0, f"{order_of(errs)} from {errs}"


def test_unattenuated_transform_matches_the_closed_form_radon():
    """mu = 0 reduces the whole thing to the Radon transform of a Gaussian,
    which is a Gaussian -- and that is known exactly."""
    sigma, amp, centre = 0.12, 1.3, (0.15, -0.1)
    f_fun = ph.gaussian(amp=amp, center=centre, sigma=sigma)
    opt = Options(n_theta_min=8, n_levels=6, gl_order=8)
    fi, mi = images(f_fun, ph.zero(), 65)
    sg = forward(fi, mi, n_theta=N_THETA, n_s=N_S, options=opt)
    wS, _, wI = analytic_gaussian(
        sg.thetas, sg.s_axis.nodes, amp=amp, center=centre, sigma=sigma,
        mu_const=0.0, half_chord=1.0,
    )
    assert rel_err(sg.S, wS) < 5e-3
    assert rel_err(sg.I, wI) < 5e-3


# --- documented limitation -------------------------------------------------


def test_a_discontinuous_phantom_is_worse():
    """Zero-fill and every interpolation assume smoothness; a hard edge costs
    accuracy.  Recorded so the assumption stays visible."""
    mu, mu_r, _ = ph.ATTENUATIONS["uniform_soft"]
    smooth, _ = ph.ACTIVITIES["centered_bump"][:2]
    sharp = ph.sharp_disk(amp=1.0, center=(0.0, 0.0), radius=0.45)
    e_smooth = algorithmic_error(*pair(smooth, mu, mu_r))["I"]
    e_sharp = algorithmic_error(*pair(sharp, mu, mu_r))["I"]
    assert e_sharp > 2.5 * e_smooth, (e_smooth, e_sharp)


@pytest.mark.slow
def test_a_discontinuous_phantom_does_not_converge():
    """The smooth phantom's error falls with the mesh; the discontinuous one
    stalls, because the field it interpolates is not the smooth object the
    analysis assumes."""
    mu, mu_r, _ = ph.ATTENUATIONS["uniform_soft"]
    sharp = ph.sharp_disk(amp=1.0, center=(0.0, 0.0), radius=0.45)
    errs = []
    for n, lev in ((17, 4), (33, 5), (65, 6)):
        ml = Options(n_theta_min=8, n_levels=lev, gl_order=6)
        direct = Options(n_theta_min=N_THETA, n_levels=lev, gl_order=6)
        a, b = pair(sharp, mu, mu_r, n=n, ml=ml, direct=direct, n_s=n)
        errs.append(algorithmic_error(a, b)["I"])
    # It stalls: the last refinement buys essentially nothing, whereas the
    # smooth phantom keeps converging (see the test above).
    assert errs[-1] > 0.8 * errs[-2], f"expected stalling, got {errs}"
