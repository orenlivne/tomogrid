"""The three-dimensional transform: structure, exact properties, and accuracy.

Kept small: 3-D tables are four-dimensional and the frame-rotation resample
costs ``q^2 p^3`` per target, so these run at 9^3 and 17^3.
"""

import numpy as np
import pytest

from tomogrid import phantoms as ph
from tomogrid.axis import Axis
from tomogrid.directions import DirectionGrid
from tomogrid.image import cube_grid, sample_function3d
from tomogrid.multilevel3d import Options3D, build_hierarchy3d, forward3d
from tomogrid.reference3d import image_evaluator3d, naive_transform3d

from conftest import order_of, rel_err

N_IMG, N_S, M_OUT, N_LEV = 9, 9, 5, 3
FAST = Options3D(m_min=3, n_levels=N_LEV, gl_order=4)
DIRECT = Options3D(m_min=M_OUT, n_levels=N_LEV, gl_order=4)


def images(f_fun, mu_fun, n=N_IMG):
    xa, ya, za = cube_grid(n)
    return sample_function3d(f_fun, xa, ya, za), sample_function3d(mu_fun, xa, ya, za)


def run(f_fun, mu_fun, *, n=N_IMG, n_s=N_S, m_out=M_OUT, options=FAST, support=0.6):
    f, mu = images(f_fun, mu_fun, n)
    return forward3d(f, mu, m_out=m_out, n_s=n_s, options=options,
                     support_radius=support)


# --- hierarchy -------------------------------------------------------------


def test_hierarchy_nests_and_covers_the_chord():
    h = build_hierarchy3d(m_out=9, m_min=3, n_levels=4,
                          s_axis=Axis.from_endpoints(-1, 1, 9))
    assert [l.dirs.m for l in h.levels] == [3, 5, 9, 9, 9]
    assert h.refine_levels == (1, 2)
    for l in range(1, h.n_levels + 1):
        fine, coarse = h.levels[l - 1].t_axis, h.levels[l].t_axis
        assert np.allclose(fine.nodes[0::2], coarse.nodes)
        assert np.isclose(h.levels[l].seg_len, 2 * h.levels[l - 1].seg_len)
    assert np.isclose(h.top.seg_len, 2 * h.half_chord)


def test_direction_count_grows_by_four_while_refining():
    h = build_hierarchy3d(m_out=17, m_min=3, n_levels=5,
                          s_axis=Axis.from_endpoints(-1, 1, 9))
    counts = [h.levels[i].dirs.n_dir for i in range(4)]
    for a, b in zip(counts, counts[1:]):
        assert 2.7 < b / a <= 4.0


@pytest.mark.parametrize("kwargs", [
    dict(m_out=8, m_min=3),     # 8 is not (3-1)*2^k + 1
    dict(m_out=3, m_min=5),     # out < min
    dict(m_out=33, m_min=3, n_levels=2),  # too few levels to reach m_out
    dict(m_min=1),
])
def test_invalid_hierarchies_are_rejected(kwargs):
    base = dict(m_out=9, m_min=3, n_levels=4,
                s_axis=Axis.from_endpoints(-1, 1, 9))
    base.update(kwargs)
    with pytest.raises(ValueError):
        build_hierarchy3d(**base)


def test_odd_t_oversample_is_rejected():
    with pytest.raises(ValueError):
        build_hierarchy3d(m_out=9, m_min=3, n_levels=4,
                          s_axis=Axis.from_endpoints(-1, 1, 9), t_oversample=3)


# --- exact properties ------------------------------------------------------


def test_shape_and_grids():
    sg = run(ph.ACTIVITIES_3D["centered_bump"][0], ph.ATTENUATIONS_3D["zero"][0],
             support=0.55)
    assert sg.S.shape == sg.E.shape == sg.I.shape == (sg.dirs.n_dir, N_S, N_S)
    assert sg.dirs.m == M_OUT
    assert isinstance(sg.dirs, DirectionGrid)


def test_zero_attenuation_makes_I_identical_to_S():
    sg = run(ph.ACTIVITIES_3D["three_blobs"][0], ph.ATTENUATIONS_3D["zero"][0],
             support=0.55)
    assert np.allclose(sg.E, 0.0, atol=1e-14)
    assert rel_err(sg.I, sg.S) < 1e-12
    assert np.allclose(sg.pet, sg.S, rtol=1e-12, atol=1e-14)


def test_zero_activity_gives_zero_signal():
    sg = run(ph.zero3d(), ph.ATTENUATIONS_3D["uniform_soft"][0], support=0.62)
    assert np.allclose(sg.S, 0.0, atol=1e-14)
    assert np.allclose(sg.I, 0.0, atol=1e-14)
    assert sg.E.max() > 0.5


def test_linear_in_the_activity():
    mu = ph.ATTENUATIONS_3D["uniform_soft"][0]
    f1 = ph.bump3d(amp=1.0, center=(-0.2, 0.1, 0.05), radius=0.25)
    f2 = ph.bump3d(amp=1.0, center=(0.2, -0.15, -0.1), radius=0.3)
    a, b = 2.0, -0.75
    s1 = run(f1, mu, support=0.62)
    s2 = run(f2, mu, support=0.62)
    sc = run(lambda x, y, z: a * f1(x, y, z) + b * f2(x, y, z), mu, support=0.62)
    assert np.allclose(sc.S, a * s1.S + b * s2.S, atol=1e-12)
    assert np.allclose(sc.I, a * s1.I + b * s2.I, atol=1e-12)
    assert np.allclose(sc.E, s1.E, atol=1e-12)


def test_attenuation_never_increases_the_signal():
    f = ph.ACTIVITIES_3D["centered_bump"][0]
    sg = run(f, ph.ATTENUATIONS_3D["very_strong"][0], support=0.62)
    assert np.all(np.isfinite(sg.I))
    assert sg.E.max() > 8.0
    assert np.abs(sg.I).max() < sg.S.max()


def test_nested_directions_are_copied_not_interpolated():
    """With m_min = m_out nothing is interpolated at all, so the work counter
    must record no table gathers."""
    sg = run(ph.ACTIVITIES_3D["centered_bump"][0],
             ph.ATTENUATIONS_3D["uniform_soft"][0], options=DIRECT, support=0.62)
    assert sg.work.get("interp_gathers", 0) == 0


# --- agreement with the independent reference ------------------------------


def test_direct_path_matches_the_naive_reference():
    f_fun = ph.ACTIVITIES_3D["centered_bump"][0]
    mu_fun = ph.ATTENUATIONS_3D["uniform_soft"][0]
    sg = run(f_fun, mu_fun, options=DIRECT, support=0.62)
    fi, mi = images(f_fun, mu_fun)
    S, E, I = naive_transform3d(image_evaluator3d(fi), image_evaluator3d(mi),
                                sg.dirs, sg.s_axis.nodes, 1.0, n_samples=1025,
                                dir_chunk=8)
    scale = float(np.abs(S).max())
    assert np.abs(sg.S - S).max() / scale < 5e-3
    assert np.abs(sg.I - I).max() / scale < 5e-3


@pytest.mark.slow
def test_discretisation_converges_to_the_continuum():
    f_fun = ph.ACTIVITIES_3D["centered_bump"][0]
    mu_fun = ph.ATTENUATIONS_3D["uniform_soft"][0]
    errs = []
    for n, lev in ((9, 3), (17, 4), (33, 5)):
        opt = Options3D(m_min=M_OUT, n_levels=lev, gl_order=6)
        sg = run(f_fun, mu_fun, n=n, n_s=N_S, options=opt, support=0.62)
        _, _, I = naive_transform3d(f_fun, mu_fun, sg.dirs, sg.s_axis.nodes, 1.0,
                                    n_samples=4097, dir_chunk=4)
        errs.append(rel_err(sg.I, I))
    assert order_of(errs).min() > 2.5, f"{order_of(errs)} from {errs}"


@pytest.mark.slow
def test_transverse_sampling_dominates_in_3d_too():
    """As in 2-D, refining ds is what buys accuracy -- and 3-D needs it more,
    because there are two transverse directions rather than one."""
    f_fun = ph.ACTIVITIES_3D["three_blobs"][0]
    mu_fun = ph.ATTENUATIONS_3D["high_contrast"][0]
    errs = []
    for n_s in (9, 17, 33):
        ml = run(f_fun, mu_fun, n=17, n_s=n_s,
                 options=Options3D(m_min=3, n_levels=4, gl_order=4))
        direct = run(f_fun, mu_fun, n=17, n_s=n_s,
                     options=Options3D(m_min=M_OUT, n_levels=4, gl_order=4))
        errs.append(float(np.abs(ml.I - direct.I).max() / np.abs(direct.S).max()))
    assert errs[0] > errs[1] > errs[2], errs
    assert errs[-1] < 0.1 * errs[0], errs
