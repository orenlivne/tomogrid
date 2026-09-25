import numpy as np
import pytest

from tomogrid.quadrature import gauss_legendre, head_integral_matrix, tail_integral_matrix


@pytest.mark.parametrize("m", [1, 2, 3, 4, 6, 8, 10])
def test_gauss_legendre_exact_to_degree_2m_minus_1(m):
    x, w = gauss_legendre(m)
    for deg in range(2 * m):
        exact = 0.0 if deg % 2 else 2.0 / (deg + 1)
        assert np.isclose(w @ x**deg, exact, atol=1e-11)


@pytest.mark.parametrize("m", [2, 4, 6, 8])
def test_tail_matrix_exact_for_polynomials(m):
    x, _ = gauss_legendre(m)
    C = tail_integral_matrix(m)
    for deg in range(m):
        assert np.allclose(C @ x**deg, (1.0 - x ** (deg + 1)) / (deg + 1), atol=1e-10)


@pytest.mark.parametrize("m", [2, 4, 8])
def test_head_plus_tail_is_the_full_integral(m):
    _, w = gauss_legendre(m)
    total = head_integral_matrix(m) + tail_integral_matrix(m)
    assert np.allclose(total, np.broadcast_to(w, total.shape), atol=1e-12)


def test_rejects_zero_points():
    with pytest.raises(ValueError):
        gauss_legendre(0)
