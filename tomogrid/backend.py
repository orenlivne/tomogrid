"""Choosing the array namespace the kernels run on (NumPy on the host, CuPy on a
GPU), and a way to *prove* the kernels stay inside the portable subset.

The doubling recursion never loops over grid points: every level is three whole
arrays, and a step is an integer gather, a fixed-tap weighted sum along one
axis, and a pointwise merge.  So the same source runs on a device if it only
ever reaches for the array library through a namespace argument ``xp`` and only
calls functions whose NumPy and CuPy meanings agree.

:data:`PORTABLE` is that list.  :class:`Recorder` wraps a namespace and notes
what was touched, so a test can assert the kernels used nothing else -- which is
the part of "it will run on a GPU" that can be checked without a GPU.
"""

from __future__ import annotations

import types

import numpy as np

#: Names whose NumPy and CuPy versions agree on signature and semantics for the
#: uses in this package.  Anything outside this set has to be justified before
#: the kernels can claim to be device-ready.
PORTABLE = frozenset({
    "abs", "arange", "asarray", "ascontiguousarray", "broadcast_to", "clip",
    "concatenate", "cos", "einsum", "empty", "exp", "float64", "full",
    "int64", "linspace", "log", "log2", "matmul", "maximum", "minimum",
    "moveaxis", "newaxis", "ones", "pi", "sin", "sqrt", "stack", "sum",
    "where", "zeros", "zeros_like", "floor", "ndarray",
})


def resolve(name: str | types.ModuleType = "numpy"):
    """Return an array namespace.  ``"auto"`` prefers CuPy when a device is up."""
    if not isinstance(name, str):
        return name
    if name == "numpy":
        return np
    if name in ("cupy", "auto"):
        try:
            import cupy  # noqa: PLC0415

            if cupy.cuda.runtime.getDeviceCount() > 0:
                return cupy
        except Exception:
            if name == "cupy":
                raise
        return np
    raise ValueError(f"unknown backend {name!r}")


def is_gpu(xp) -> bool:
    return xp.__name__ == "cupy"


def to_device(xp, a):
    return a if xp is np else xp.asarray(a)


def to_host(a) -> np.ndarray:
    """Bring an array back to NumPy, whichever namespace produced it."""
    get = getattr(a, "get", None)
    return np.asarray(a) if get is None else get()


def sync(xp) -> None:
    """Wait for queued device work, so timings mean something."""
    if is_gpu(xp):
        xp.cuda.runtime.deviceSynchronize()


class Recorder(types.ModuleType):
    """A namespace that forwards to ``inner`` and records the names used."""

    def __init__(self, inner=np):
        super().__init__(f"recorder({inner.__name__})")
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "used", set())

    def __getattr__(self, name):
        if name in ("_inner", "used"):
            raise AttributeError(name)
        self.used.add(name)
        return getattr(self._inner, name)

    def unportable(self) -> set[str]:
        return {n for n in self.used if n not in PORTABLE}


_DEMOTE = ("zeros", "ones", "empty", "full", "asarray", "arange", "linspace",
           "zeros_like", "ones_like", "empty_like", "array")


class SinglePrecision(types.ModuleType):
    """A namespace whose float64 results come back as float32.

    A harness, not a production backend: it lets the whole pipeline be run in
    single precision without a ``dtype`` argument on every array constructor,
    which is how we check whether the recursion is accurate enough in fp32 --
    the precision most GPUs are fast in.  Integer results are untouched.
    """

    def __init__(self, inner=np):
        super().__init__(f"float32({inner.__name__})")
        object.__setattr__(self, "_inner", inner)

    def __getattr__(self, name):
        if name == "_inner":
            raise AttributeError(name)
        obj = getattr(self._inner, name)
        if name not in _DEMOTE:
            return obj

        def wrapper(*args, **kw):
            out = obj(*args, **kw)
            if getattr(out, "dtype", None) == self._inner.float64:
                return out.astype(self._inner.float32)
            return out

        return wrapper
