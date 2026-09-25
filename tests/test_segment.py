"""The exact-merge algebra -- the one identity the whole method rests on."""

import numpy as np
import pytest

from tomogrid.quadrature import gauss_legendre
from tomogrid.segment import Triple, merge, merge_level, quad_linear, quad_spectral

from conftest import order_of

F = lambda t: 1.0 + 0.7 * np.sin(3.0 * t) + 0.2 * t**2
MU = lambda t: 0.5 + 0.4 * np.cos(2.0 * t + 1.0)


def seg(fun_f, fun_mu, a, b, m=14):
    """Triple of ``[a, b]`` from analytic integrands, to near machine precision."""
    x, _ = gauss_legendre(m)
    t = 0.5 * (a + b) + 0.5 * (b - a) * x
    return quad_spectral(np.asarray(fun_f(t)), np.asarray(fun_mu(t)), b - a)


# --- the identity ----------------------------------------------------------


@pytest.mark.parametrize("c", [0.1, 0.25, 0.5, 0.73, 0.9])
def test_merge_reproduces_the_whole_segment(c):
    whole = seg(F, MU, 0.0, 1.0)
    got = merge(seg(F, MU, 0.0, c), seg(F, MU, c, 1.0))
    assert np.isclose(got.S, whole.S, atol=1e-12)
    assert np.isclose(got.E, whole.E, atol=1e-12)
    assert np.isclose(got.I, whole.I, atol=1e-12)


def test_merge_is_associative():
    a, b, c, d = 0.0, 0.3, 0.55, 1.0
    p, q, r = seg(F, MU, a, b), seg(F, MU, b, c), seg(F, MU, c, d)
    left = merge(merge(p, q), r)
    right = merge(p, merge(q, r))
    for attr in ("S", "E", "I"):
        assert np.isclose(getattr(left, attr), getattr(right, attr), atol=1e-13)


def test_chain_of_many_panels_converges_to_the_whole():
    whole = seg(F, MU, 0.0, 1.0)
    errs = []
    for K in (2, 4, 8, 16):
        edges = np.linspace(0.0, 1.0, K + 1)
        acc = None
        for i in range(K):
            p = seg(F, MU, edges[i], edges[i + 1], m=3)
            acc = p if acc is None else merge(acc, p)
        errs.append(abs(acc.I - whole.I))
    assert order_of(errs).min() > 4.0, f"observed {order_of(errs)}"


def test_zero_attenuation_makes_I_equal_S():
    t = seg(F, lambda z: np.zeros_like(z), 0.0, 1.0)
    assert np.isclose(t.E, 0.0, atol=1e-14)
    assert np.isclose(t.I, t.S, rtol=1e-13)


def test_merge_damps_error_in_the_left_half():
    """exp(-E_right) <= 1, so the recursion never amplifies -- the reason the
    end-referenced I of eq. (3) is the right variable to carry."""
    left = seg(F, MU, 0.0, 0.5)
    right = seg(F, MU, 0.5, 1.0)
    clean = merge(left, right)
    for bump in (1e-3, 1e-2):
        dirty = merge(Triple(S=left.S, E=left.E, I=left.I + bump), right)
        assert abs(dirty.I - clean.I) <= bump * np.exp(-right.E) * (1 + 1e-12)
        assert abs(dirty.I - clean.I) < bump


@pytest.mark.parametrize("scale", [1.0, 5.0, 20.0])
def test_strong_attenuation_stays_bounded(scale):
    t = seg(F, lambda z: scale * MU(z), 0.0, 1.0)
    assert 0.0 <= t.I <= seg(F, lambda z: np.zeros_like(z), 0.0, 1.0).S + 1e-12


# --- the level-0 kernel ----------------------------------------------------


@pytest.mark.parametrize("c", [0.0, 0.3, 3.0])
@pytest.mark.parametrize("length", [0.05, 0.5, 1.0])
def test_quad_spectral_uniform_case(c, length):
    m = 8
    t = quad_spectral(np.ones(m), np.full(m, c), length)
    exact = length if c == 0 else -np.expm1(-c * length) / c
    assert np.isclose(t.I, exact, rtol=1e-9)
    assert np.isclose(t.S, length, rtol=1e-13)
    assert np.isclose(t.E, c * length, rtol=1e-13)


def test_quad_spectral_order_increases_with_nodes():
    """A short segment is resolved to machine precision by a handful of nodes."""
    whole = seg(F, MU, 0.0, 0.05, m=16)
    errs = [abs(seg(F, MU, 0.0, 0.05, m=m).I - whole.I) for m in (2, 3, 4)]
    assert errs[0] > errs[1] > errs[2]
    assert errs[-1] < 1e-12


def test_quad_spectral_shape_mismatch_raises():
    with pytest.raises(ValueError):
        quad_spectral(np.ones(4), np.ones(5), 1.0)


@pytest.mark.parametrize("c", [0.0, 1e-14, 1e-9, 0.3, 5.0])
def test_quad_linear_is_stable_at_small_attenuation(c):
    d = 0.3
    t = quad_linear(np.array([1.0, 1.0]), np.asarray(float(c)), d)
    exact = d if c == 0 else -np.expm1(-c * d) / c
    assert np.isclose(t.I, exact, rtol=1e-10)


def test_quad_linear_exact_for_linear_f_without_attenuation():
    t = quad_linear(np.array([1.0, 3.0]), np.asarray(0.0), 2.0)
    assert np.isclose(t.S, 4.0) and np.isclose(t.I, 4.0)


def test_quad_linear_agrees_with_spectral_to_second_order():
    errs = []
    for d in (0.2, 0.1, 0.05):
        ref = seg(F, MU, 0.0, d, m=12)
        x = np.array([F(0.0), F(d)])
        approx = quad_linear(x, np.asarray(MU(d / 2)), d)
        errs.append(abs(approx.I - ref.I) / d)
    assert order_of(errs).min() > 1.6, f"observed {order_of(errs)}"


# --- merge_level: the nested t-grid bookkeeping ----------------------------


@pytest.mark.parametrize("nu", [2, 4, 8])
def test_merge_level_matches_directly_built_coarse_segments(nu):
    """The structural claim: level-l segment j is exactly the union of
    level-(l-1) segments 2j -/+ nu/2, so merging needs no interpolation."""
    H = 1.0
    fine_len = 0.25
    coarse_len = 2 * fine_len
    fine_nodes = -H + np.arange(int(2 * H / (fine_len / nu)) + 1) * (fine_len / nu)
    coarse_nodes = fine_nodes[::2]

    def build(nodes, length):
        parts = [seg(F, MU, t - length / 2, t + length / 2) for t in nodes]
        return Triple(
            S=np.array([p.S for p in parts]),
            E=np.array([p.E for p in parts]),
            I=np.array([p.I for p in parts]),
        )

    merged = merge_level(build(fine_nodes, fine_len), nu)
    direct = build(coarse_nodes, coarse_len)
    assert merged.shape == direct.shape
    # Interior only: the ends are zero-padded on purpose.
    half = nu // 2
    sl = slice(half, len(coarse_nodes) - half)
    for attr in ("S", "E", "I"):
        got, want = getattr(merged, attr)[sl], getattr(direct, attr)[sl]
        assert np.allclose(got, want, atol=1e-12), attr


@pytest.mark.parametrize("nu", [1, 3, 0, -2])
def test_merge_level_rejects_odd_oversampling(nu):
    tri = Triple(S=np.zeros(9), E=np.zeros(9), I=np.zeros(9))
    with pytest.raises(ValueError):
        merge_level(tri, nu)


def test_merge_level_rejects_even_length_axis():
    tri = Triple(S=np.zeros(8), E=np.zeros(8), I=np.zeros(8))
    with pytest.raises(ValueError):
        merge_level(tri, 2)


def test_merge_level_halves_the_grid():
    tri = Triple(*[np.zeros((3, 5, 17)) for _ in range(3)])
    assert merge_level(tri, 2).shape == (3, 5, 9)


def test_triple_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        Triple(S=np.zeros(3), E=np.zeros(4), I=np.zeros(3))


def test_pet_helper():
    t = Triple(S=np.array([2.0]), E=np.array([np.log(4.0)]), I=np.array([0.0]))
    assert np.allclose(t.pet(), [0.5])
