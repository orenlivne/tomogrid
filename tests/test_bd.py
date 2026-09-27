"""The attenuated transform on the Brandt--Dym indexing, 2-D and 3-D.

The properties that must hold *exactly* are the point of this file: a direction
present at the base level is merged and never interpolated, and a constant field
is reproduced to roundoff at every level.  Both would fail if the indexing or
the arc-length normalisation were wrong, and both failed for the rotating-frame
formulation this replaces.
"""

import numpy as np
import pytest

from tomogrid import phantoms as ph
from tomogrid.bd import (SWEEPS, build_grid, build_level, forward, refine,
                         half_arc_lengths)
from tomogrid.bd3d import (SWEEPS_3D, build_grid3d, build_level3d, forward3d,
                           refine3d)
from tomogrid.image import (cube_grid, sample_function, sample_function3d,
                            square_grid)
from tomogrid.reference import naive_segments
from tomogrid.reference3d import naive_segments3d, image_evaluator3d

from conftest import rel_err

N2, N3 = 65, 17


def images2(f_fun, mu_fun, n=N2):
    xa, ya = square_grid(n)
    return sample_function(f_fun, xa, ya), sample_function(mu_fun, xa, ya)


def images3(f_fun, mu_fun, n=N3):
    xa, ya, za = cube_grid(n)
    return (sample_function3d(f_fun, xa, ya, za),
            sample_function3d(mu_fun, xa, ya, za))


# --- geometry --------------------------------------------------------------


def test_grid_halves_positions_and_doubles_slopes():
    g = build_grid(half_width=1.0, image_h=2 / 64, sigma=1)
    for a, b in zip(g.levels, g.levels[1:]):
        assert np.isclose(b.length, 2 * a.length)
        assert b.n_m == 2 * a.n_m - 1
        assert b.n_x == max(1, a.n_x // 2)
    assert g.top.n_x == 1
    assert np.isclose(g.top.length, 2.0)


def test_table_size_is_flat_across_levels():
    g = build_grid(half_width=1.0, image_h=2 / 128, sigma=1)
    sizes = [lv.n_x * g.n_y * lv.n_m for lv in g.levels]
    assert max(sizes) / min(sizes) < 1.3, sizes


def test_slopes_span_the_shallow_family():
    g = build_grid(half_width=1.0, image_h=2 / 64, sigma=2)
    for lv in g.levels:
        s = g.slopes(lv)
        assert np.isclose(s.min(), -1.0) and np.isclose(s.max(), 1.0)


def test_half_arc_lengths_use_the_target_geometry():
    g = build_grid(half_width=1.0, image_h=2 / 32, sigma=1)
    lv, nxt = g.levels[0], g.levels[1]
    half = half_arc_lengths(g, 0)
    M = np.arange(-nxt.rise, nxt.rise + 1)
    assert np.allclose(half, np.sqrt(lv.length**2 + (0.5 * M * g.h) ** 2))
    # the two halves must add up to the full segment
    full = np.sqrt(nxt.length**2 + (M * g.h) ** 2)
    assert np.allclose(2 * half, full)


def test_grid_rejects_bad_sigma():
    with pytest.raises(ValueError):
        build_grid(half_width=1.0, image_h=2 / 32, sigma=0)


# --- exactness, 2-D --------------------------------------------------------


def test_constant_field_is_exact_at_every_level_2d():
    n = 129
    xa, ya = square_grid(n)
    f = sample_function(lambda x, y: np.ones_like(x * y), xa, ya)
    mu = sample_function(lambda x, y: np.zeros_like(x * y), xa, ya)
    g = build_grid(half_width=1.0, image_h=xa.h, sigma=1)
    tri = build_level(f, mu, g, 0, gl_order=4)
    for k in range(g.levels[-1].index):
        tri = refine(tri, g, k, order=4)
        lv = g.levels[k + 1]
        nx, ny = tri.S.shape[0], tri.S.shape[1]
        px, pady = (1 if nx > 2 else 0), lv.rise + 3
        if nx - 2 * px < 1 or ny - 2 * pady < 1:
            continue
        sub = tri.S[px:nx - px, pady:ny - pady, :]
        assert np.abs(sub - 1.0).max() < 1e-12, lv.length


def test_zero_attenuation_gives_I_equal_S_2d():
    f, mu = images2(ph.ACTIVITIES["three_blobs"][0], ph.ATTENUATIONS["zero"][0])
    sw = forward(f, mu, sigma=1, order=4)
    for s in sw.values():
        assert np.abs(s.triple.E).max() < 1e-14
        assert rel_err(s.triple.I, s.triple.S) < 1e-12


def test_zero_activity_gives_zero_2d():
    f, mu = images2(ph.zero(), ph.ATTENUATIONS["uniform_soft"][0])
    sw = forward(f, mu, sigma=1, order=4)
    for s in sw.values():
        assert np.abs(s.triple.S).max() < 1e-14
        assert np.abs(s.triple.I).max() < 1e-14
    assert max(s.triple.E.max() for s in sw.values()) > 0.3


def test_linear_in_the_activity_2d():
    mu_fun = ph.ATTENUATIONS["uniform_soft"][0]
    f1 = ph.bump(amp=1.0, center=(-0.2, 0.1), radius=0.25)
    f2 = ph.bump(amp=1.0, center=(0.2, -0.15), radius=0.3)
    a, b = 2.0, -0.75
    s1 = forward(*images2(f1, mu_fun), sigma=1, order=4)["+x"]
    s2 = forward(*images2(f2, mu_fun), sigma=1, order=4)["+x"]
    sc = forward(*images2(lambda x, y: a * f1(x, y) + b * f2(x, y), mu_fun),
                 sigma=1, order=4)["+x"]
    assert np.abs(sc.triple.S - (a * s1.triple.S + b * s2.triple.S)).max() < 1e-12
    assert np.abs(sc.triple.I - (a * s1.triple.I + b * s2.triple.I)).max() < 1e-12


def test_all_four_families_are_produced():
    f, mu = images2(ph.bump(radius=0.4), ph.zero())
    sw = forward(f, mu, sigma=1, order=4)
    assert set(sw) == {name for name, _, _ in SWEEPS}
    for s in sw.values():
        assert s.p0.shape == s.p1.shape
        assert s.p0.shape[:-1] == s.triple.S.shape


def test_matches_an_independent_reference_2d():
    f_fun = ph.ACTIVITIES["three_blobs"][0]
    mu_fun = ph.ATTENUATIONS["high_contrast"][0]
    f, mu = images2(f_fun, mu_fun, 129)
    sw = forward(f, mu, sigma=2, order=4)
    for name, s in sw.items():
        t = s.integrals()
        S, E, I = naive_segments(f_fun, mu_fun, s.p0, s.p1, n_samples=1025)
        scale = float(np.abs(S).max())
        assert np.abs(t.S - S.reshape(t.S.shape)).max() / scale < 1e-2, name
        assert np.abs(t.I - I.reshape(t.I.shape)).max() / scale < 1e-2, name


def test_refining_the_integration_mesh_improves_accuracy_2d():
    f_fun = ph.ACTIVITIES["three_blobs"][0]
    mu_fun = ph.ATTENUATIONS["high_contrast"][0]
    f, mu = images2(f_fun, mu_fun, 129)
    errs = []
    for sigma in (1, 2):
        s = forward(f, mu, sigma=sigma, order=4)["+x"]
        t = s.integrals()
        _, _, I = naive_segments(f_fun, mu_fun, s.p0, s.p1, n_samples=1025)
        errs.append(rel_err(t.I, I.reshape(t.I.shape)))
    assert errs[1] < 0.5 * errs[0], errs


def test_refine_rejects_a_mismatched_table():
    from tomogrid.segment import Triple

    g = build_grid(half_width=1.0, image_h=2 / 32, sigma=1)
    with pytest.raises(ValueError):
        refine(Triple(*[np.zeros((2, 3, 4)) for _ in range(3)]), g, 0)


# --- exactness, 3-D --------------------------------------------------------


def test_constant_field_is_exact_at_every_level_3d():
    n = 33
    xa, ya, za = cube_grid(n)
    one = sample_function3d(
        lambda x, y, z: np.ones(np.broadcast(x, y, z).shape), xa, ya, za)
    zero = sample_function3d(
        lambda x, y, z: np.zeros(np.broadcast(x, y, z).shape), xa, ya, za)
    g = build_grid3d(half_width=1.0, image_h=xa.h, sigma=1)
    tri = build_level3d(one, zero, g, 0, gl_order=4)
    for k in range(g.levels[-1].index):
        tri = refine3d(tri, g, k, order=4)
        lv = g.levels[k + 1]
        nx, nt, pad = tri.S.shape[0], g.n_t, lv.rise + 2
        px = 1 if nx > 2 else 0
        if nt - 2 * pad < 1 or nx - 2 * px < 1:
            continue
        sub = tri.S[px:nx - px, pad:nt - pad, pad:nt - pad, :, :]
        assert np.abs(sub - 1.0).max() < 1e-12, lv.length


def test_grid3d_grows_by_two_per_level():
    """In 3-D directions quadruple while centres halve, so unlike 2-D the table
    grows and the work is dominated by the top level -- the output itself."""
    g = build_grid3d(half_width=1.0, image_h=2 / 32, sigma=1)
    sizes = [lv.n_x * g.n_t**2 * lv.n_m**2 for lv in g.levels]
    for a, b in zip(sizes, sizes[1:]):
        assert 1.5 < b / a < 2.5, sizes


def test_zero_attenuation_gives_I_equal_S_3d():
    f, mu = images3(ph.ACTIVITIES_3D["three_blobs"][0],
                    ph.ATTENUATIONS_3D["zero"][0])
    s = forward3d(f, mu, sigma=1, order=4, families=("+x",))["+x"]
    assert np.abs(s.triple.E).max() < 1e-14
    assert rel_err(s.triple.I, s.triple.S) < 1e-12


def test_zero_activity_gives_zero_3d():
    f, mu = images3(ph.zero3d(), ph.ATTENUATIONS_3D["uniform_soft"][0])
    s = forward3d(f, mu, sigma=1, order=4, families=("+x",))["+x"]
    assert np.abs(s.triple.S).max() < 1e-14
    assert np.abs(s.triple.I).max() < 1e-14
    assert s.triple.E.max() > 0.1


def test_all_six_families_are_produced_3d():
    f, mu = images3(ph.bump3d(radius=0.4), ph.zero3d())
    sw = forward3d(f, mu, sigma=1, order=2)
    assert set(sw) == {name for name, _, _ in SWEEPS_3D}
    for s in sw.values():
        assert s.p0.shape == s.p1.shape
        assert s.p0.shape[:-1] == s.triple.S.shape


@pytest.mark.slow
def test_matches_an_independent_reference_3d():
    f_fun = ph.ACTIVITIES_3D["centered_bump"][0]
    mu_fun = ph.ATTENUATIONS_3D["uniform_soft"][0]
    f, mu = images3(f_fun, mu_fun, 33)
    s = forward3d(f, mu, sigma=2, order=4, families=("+x",))["+x"]
    t = s.integrals()
    sl = (slice(None, None, 6), slice(None, None, 6),
          slice(None, None, 12), slice(None, None, 12))
    S, _, I = naive_segments3d(f_fun, mu_fun, s.p0[sl], s.p1[sl], n_samples=1025)
    scale = float(np.abs(S).max())
    assert np.abs(t.S[sl].ravel() - S).max() / scale < 5e-2
    assert np.abs(t.I[sl].ravel() - I).max() / scale < 5e-2
