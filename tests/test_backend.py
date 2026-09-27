"""The kernels must run on whatever array library they are handed.

There is no GPU in this container, so what can be checked here is the part that
actually breaks ports: that the kernels reach for the array library only through
their ``xp`` argument, and only for functions whose CuPy meaning matches NumPy's
(:data:`tomogrid.backend.PORTABLE`).  A recording namespace makes that testable
on the host.  Indexing and arithmetic operators are not recorded; they are
identical in the two libraries for the patterns used here.
"""

from __future__ import annotations

import numpy as np
import pytest

from tomogrid import bd, bd3d
from tomogrid.backend import PORTABLE, Recorder, resolve, to_host
from tomogrid.image import (Image, Image3D, cube_grid, sample_function,
                            sample_function3d, square_grid)


def _images(n=17):
    ax, ay = square_grid(n)
    f = sample_function(lambda x, y: np.exp(-6 * (x**2 + y**2)), ax, ay)
    mu = sample_function(lambda x, y: 0.4 * np.exp(-3 * (x**2 + y**2)), ax, ay)
    return f, mu


def _volumes(n=9):
    ax, ay, az = cube_grid(n)
    f = sample_function3d(lambda x, y, z: np.exp(-4 * (x**2 + y**2 + z**2)),
                          ax, ay, az)
    mu = sample_function3d(lambda x, y, z: 0.3 / (1 + x**2 + y**2 + z**2),
                           ax, ay, az)
    return f, mu


def test_resolve_numpy():
    assert resolve("numpy") is np
    assert resolve(np) is np
    with pytest.raises(ValueError):
        resolve("nonsense")


def test_resolve_auto_falls_back_without_a_device():
    # "auto" must never raise: it is what scripts use.
    assert resolve("auto").__name__ in ("numpy", "cupy")


def test_forward_2d_uses_only_portable_ops():
    f, mu = _images()
    rec = Recorder(np)
    bd.forward(f, mu, sigma=1, xp=rec)
    assert rec.used, "the recorder saw nothing -- xp is not being threaded"
    assert rec.unportable() == set(), (
        f"2-D kernels used non-portable array ops: {sorted(rec.unportable())}")


def test_forward_3d_uses_only_portable_ops():
    f, mu = _volumes()
    rec = Recorder(np)
    bd3d.forward3d(f, mu, sigma=1, xp=rec, families=("+x",))
    assert rec.used
    assert rec.unportable() == set(), (
        f"3-D kernels used non-portable array ops: {sorted(rec.unportable())}")


def test_backend_argument_does_not_change_the_answer_2d():
    f, mu = _images()
    a = bd.forward(f, mu, sigma=1)
    b = bd.forward(f, mu, sigma=1, xp=Recorder(np))
    for name in a:
        for q in ("S", "E", "I"):
            np.testing.assert_allclose(getattr(a[name].triple, q),
                                       getattr(b[name].triple, q), rtol=0,
                                       atol=0)


def test_backend_argument_does_not_change_the_answer_3d():
    f, mu = _volumes()
    a = bd3d.forward3d(f, mu, sigma=1, families=("+x",))
    b = bd3d.forward3d(f, mu, sigma=1, families=("+x",), xp=Recorder(np))
    for q in ("S", "E", "I"):
        np.testing.assert_allclose(getattr(a["+x"].triple, q),
                                   getattr(b["+x"].triple, q), rtol=0, atol=0)


def test_to_host_is_a_noop_on_numpy():
    a = np.arange(6.0)
    assert to_host(a) is a or np.shares_memory(to_host(a), a)


def test_portable_names_exist_in_numpy():
    missing = [n for n in PORTABLE if not hasattr(np, n)]
    assert missing == []


def test_image_to_numpy_is_identity():
    f, _ = _images(9)
    assert f.to(np) is f
    v, _ = _volumes(5)
    assert v.to(np) is v


def test_interior_slope_stencil_is_constant_coefficient():
    """Away from the ends of the slope range the doubling is a convolution.

    Both the offsets and the weights are the same for every target, so the
    interpolation step is a fixed four-tap stencil and not a gather with
    per-target indices -- which is what makes it a streaming kernel on a
    device.  The ends are one-sided and are the only special cases.
    """
    from tomogrid.bd import _slope_stencil

    q, n_in = 4, 17
    idx, w = _slope_stencil(n_in - 1, n_in, q)
    m = np.arange(n_in - 1)
    offsets = idx - m[None, :]
    interior = slice(1, n_in - 2)
    np.testing.assert_array_equal(
        offsets[:, interior],
        np.array([-1, 0, 1, 2])[:, None] + np.zeros_like(offsets[:, interior]))
    np.testing.assert_allclose(
        w[:, interior],
        np.array([-1 / 16, 9 / 16, 9 / 16, -1 / 16])[:, None]
        + np.zeros_like(w[:, interior]), rtol=0, atol=1e-15)
