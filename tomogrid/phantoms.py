"""Smooth, compactly supported test fields, and the battery used by the tests.

Zero-fill outside the tables is exact only if the fields really vanish there, and
the *order* of the interpolation is only achieved if they vanish smoothly.  So the
building blocks here are ``C^infinity`` and compactly supported:

    bump      exp(1 - 1/(1-r^2))                  a single smooth blob
    plateau   smooth 1 -> 0 transition             a "uniform" region with soft edge

``SHARP_DISK`` is deliberately non-smooth and is used to *demonstrate* the
order loss, not to claim accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _psi(u: np.ndarray) -> np.ndarray:
    """``C^infinity`` step: 0 for ``u <= 0``, 1 for ``u >= 1``."""
    u = np.asarray(u, dtype=float)
    inside = (u > 0.0) & (u < 1.0)
    us = np.where(inside, u, 0.5)
    a = np.exp(-1.0 / us)
    b = np.exp(-1.0 / (1.0 - us))
    return np.where(u <= 0.0, 0.0, np.where(u >= 1.0, 1.0, a / (a + b)))


def _radius(x, y, center, scale=(1.0, 1.0), angle=0.0):
    cx, cy = center
    dx, dy = np.asarray(x) - cx, np.asarray(y) - cy
    ca, sa = np.cos(angle), np.sin(angle)
    u = (ca * dx + sa * dy) / scale[0]
    v = (-sa * dx + ca * dy) / scale[1]
    return np.sqrt(u * u + v * v)


def bump(amp=1.0, center=(0.0, 0.0), radius=0.3, scale=None, angle=0.0):
    """``C^infinity`` blob of peak ``amp``, identically zero outside ``radius``."""
    sc = (radius, radius) if scale is None else (radius * scale[0], radius * scale[1])

    def fun(x, y):
        r = _radius(x, y, center, sc, angle)
        inside = r < 1.0
        rs = np.where(inside, r, 0.0)
        return np.where(inside, amp * np.exp(1.0 - 1.0 / np.maximum(1.0 - rs**2, 1e-300)), 0.0)

    return fun


def plateau(amp=1.0, center=(0.0, 0.0), r_flat=0.35, r_zero=0.6, scale=None, angle=0.0):
    """Smoothly-edged "uniform" region: ``amp`` inside ``r_flat``, 0 outside ``r_zero``."""
    sc = (1.0, 1.0) if scale is None else scale

    def fun(x, y):
        r = _radius(x, y, center, sc, angle)
        return amp * _psi((r_zero - r) / (r_zero - r_flat))

    return fun


def gaussian(amp=1.0, center=(0.0, 0.0), sigma=0.15):
    def fun(x, y):
        cx, cy = center
        return amp * np.exp(-(((np.asarray(x) - cx) ** 2 + (np.asarray(y) - cy) ** 2)) / (2 * sigma**2))

    return fun


def constant(value=1.0):
    def fun(x, y):
        return value + 0.0 * np.asarray(x) * np.asarray(y)

    return fun


def zero():
    return constant(0.0)


def total(*funs):
    def fun(x, y):
        out = 0.0
        for f in funs:
            out = out + f(x, y)
        return out

    return fun


def modulated(base, kx=6.0, ky=0.0, depth=0.5):
    """``base * (1 + depth*sin(kx*x + ky*y))`` -- stresses the smoothness assumption."""

    def fun(x, y):
        return base(x, y) * (1.0 + depth * np.sin(kx * np.asarray(x) + ky * np.asarray(y)))

    return fun


def sharp_disk(amp=1.0, center=(0.0, 0.0), radius=0.5):
    """Discontinuous -- included only to show what non-smoothness costs."""

    def fun(x, y):
        return np.where(_radius(x, y, center, (radius, radius)) < 1.0, amp, 0.0)

    return fun


@dataclass(frozen=True)
class Case:
    name: str
    f: object
    mu: object
    support_radius: float
    note: str = ""


# --- attenuation battery ---------------------------------------------------
# Peak total attenuation along a chord is roughly amp * 2 * r_flat.

ATTENUATIONS = {
    "zero": (zero(), 0.0, "no attenuation: I must equal S exactly"),
    "uniform_soft": (plateau(amp=1.0, r_flat=0.4, r_zero=0.62), 0.62, "water-like disk"),
    "single_bump": (bump(amp=2.0, center=(0.1, -0.05), radius=0.5), 0.6, "one smooth blob"),
    "two_bumps": (
        total(bump(amp=1.5, center=(-0.22, 0.12), radius=0.3),
              bump(amp=2.5, center=(0.25, -0.18), radius=0.28)),
        0.55,
        "asymmetric: breaks the theta -> theta+pi symmetry of I",
    ),
    "high_contrast": (
        total(plateau(amp=1.0, r_flat=0.35, r_zero=0.6),
              bump(amp=8.0, center=(0.15, 0.1), radius=0.22)),
        0.6,
        "peak chord attenuation ~ 5; tests exponent conditioning",
    ),
    "anisotropic": (
        bump(amp=3.0, center=(0.0, 0.0), radius=0.55, scale=(1.0, 0.35), angle=0.6),
        0.6,
        "elongated: large |grad mu| in one direction",
    ),
    "gradient": (
        lambda x, y: plateau(amp=1.0, r_flat=0.38, r_zero=0.6)(x, y) * (1.5 + x + 0.7 * y),
        0.6,
        "nonzero grad mu everywhere in the support",
    ),
    "oscillatory": (
        modulated(plateau(amp=1.2, r_flat=0.38, r_zero=0.6), kx=8.0, ky=3.0, depth=0.6),
        0.6,
        "short-wavelength mu: stresses the smoothness assumption",
    ),
    "very_strong": (
        plateau(amp=12.0, r_flat=0.4, r_zero=0.6),
        0.62,
        "chord attenuation ~ 10: exp(-E) ~ 5e-5",
    ),
}

# --- activity battery ------------------------------------------------------

ACTIVITIES = {
    "centered_bump": (bump(amp=1.0, center=(0.0, 0.0), radius=0.5), 0.55),
    "offcenter_bump": (bump(amp=1.0, center=(0.2, -0.15), radius=0.3), 0.55),
    "three_blobs": (
        total(bump(amp=1.0, center=(-0.25, 0.2), radius=0.22),
              bump(amp=0.6, center=(0.3, 0.1), radius=0.18),
              bump(amp=1.4, center=(0.05, -0.28), radius=0.2)),
        0.55,
    ),
    "smooth_phantom": (
        total(plateau(amp=0.4, r_flat=0.3, r_zero=0.55),
              bump(amp=1.0, center=(-0.18, 0.15), radius=0.16),
              bump(amp=-0.25, center=(0.22, 0.05), radius=0.14),
              bump(amp=0.8, center=(0.0, -0.3), radius=0.13, scale=(1.6, 0.7), angle=0.4)),
        0.58,
        # signed, to make sure nothing secretly assumes f >= 0
    ),
}


def battery() -> list[Case]:
    """All (activity, attenuation) pairs used by the accuracy tests."""
    cases = []
    for mu_name, (mu, mu_r, note) in ATTENUATIONS.items():
        for f_name, entry in ACTIVITIES.items():
            f, f_r = entry[0], entry[1]
            cases.append(
                Case(
                    name=f"{f_name}|{mu_name}",
                    f=f,
                    mu=mu,
                    support_radius=max(f_r, mu_r),
                    note=note,
                )
            )
    return cases
