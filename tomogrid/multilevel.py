"""The multilevel forward transform.

``forward()`` evaluates, for every requested ``(theta, s)``,

    S = int f              (Radon transform of the activity)
    E = int mu             (Radon transform of the attenuation)
    I = int f exp(-int_t^b mu)      (attenuated / SPECT transform)
    PET = exp(-E) * S

in ``O(N log N)`` work, ``N`` = number of image pixels ~ number of sinogram bins.

Setting ``n_theta_min = n_theta_out`` disables every interpolation and makes this
the direct ``O(N^1.5)`` method -- which is how the tests obtain an exact
reference for the *same* discretisation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .axis import Axis
from .hierarchy import Hierarchy, Level, build_hierarchy, thetas_of
from .image import Image
from .interp import interp2d_stacked, lagrange_weights
from .quadrature import gauss_legendre
from .segment import Triple, merge_level, quad_spectral


@dataclass
class Options:
    """Accuracy/cost knobs.  ``n_theta_min`` is the important one."""

    n_theta_min: int = 16
    n_levels: int | None = None  # default: finest segment ~ one image cell
    half_chord: float = 1.0
    gl_order: int = 8  # Gauss-Legendre nodes per level-0 segment
    img_order: int = 4  # image interpolation (4 = cubic)
    theta_order: int = 4  # angular interpolation stencil
    s_order: int = 4  # transverse interpolation stencil
    t_order: int = 4  # along-ray interpolation stencil
    t_oversample: int = 2  # dt = seg_len / t_oversample; must be even (see hierarchy)
    s_oversample: int = 2  # internal ds = output ds / s_oversample
    max_block: int = 1 << 22  # cap on working-array elements; purely about memory


@dataclass
class Sinogram:
    """Result of :func:`forward`, indexed ``[i_theta, i_s]``."""

    S: np.ndarray
    E: np.ndarray
    I: np.ndarray
    thetas: np.ndarray
    s_axis: Axis
    hierarchy: Hierarchy | None = None
    work: dict = field(default_factory=dict)

    @property
    def attenuated(self) -> np.ndarray:
        """The SPECT / attenuated-Radon model, eq. (1)."""
        return self.I

    @property
    def pet(self) -> np.ndarray:
        """The PET model, eq. (2): ``exp(-E) * S``."""
        return np.exp(-self.E) * self.S

    def as_triple(self) -> Triple:
        return Triple(S=self.S, E=self.E, I=self.I)


def default_n_levels(image_h: float, half_chord: float) -> int:
    """Levels such that the finest segment is about one image cell long."""
    return max(0, int(math.ceil(math.log2(2.0 * half_chord / image_h))))


def _angle_chunks(n_angles: int, per_angle: int, max_block: int):
    """Split the direction axis so no working array exceeds ``max_block`` elements.

    Level-0 construction and angular prolongation both build arrays of shape
    ``(n_angles, n_s, n_t, ...)``, which at realistic sizes runs to many GB if
    materialised whole.  Chunking changes nothing numerically.
    """
    step = max(1, min(n_angles, int(max_block // max(1, per_angle))))
    for lo in range(0, n_angles, step):
        yield lo, min(lo + step, n_angles)


def _frame(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Direction and offset unit vectors, each shape ``theta.shape + (2,)``."""
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([c, s], axis=-1), np.stack([-s, c], axis=-1)


def _build_level0(
    f_img: Image, mu_img: Image, level: Level, s_axis: Axis, opt: Options
) -> Triple:
    """Direct evaluation of the finest-level segment triples."""
    gl_x, _ = gauss_legendre(opt.gl_order)
    w_dir, w_perp = _frame(level.thetas)  # (n_theta, 2)
    n_t = level.t_axis.n
    s = s_axis.nodes[None, :, None, None]
    t = level.t_axis.nodes[None, None, :, None] + (0.5 * level.seg_len) * gl_x[None, None, None, :]

    shape = (level.n_theta, s_axis.n, n_t)
    out = Triple(*[np.empty(shape) for _ in range(3)])
    per_angle = s_axis.n * n_t * opt.gl_order
    for lo, hi in _angle_chunks(level.n_theta, per_angle, opt.max_block):
        wd, wp = w_dir[lo:hi], w_perp[lo:hi]
        # positions[k, i_s, i_t, i_node] in x and y
        px = s * wp[:, 0, None, None, None] + t * wd[:, 0, None, None, None]
        py = s * wp[:, 1, None, None, None] + t * wd[:, 1, None, None, None]
        part = quad_spectral(
            f_img.sample(px, py, order=opt.img_order),
            mu_img.sample(px, py, order=opt.img_order),
            level.seg_len,
        )
        out.S[lo:hi], out.E[lo:hi], out.I[lo:hi] = part.S, part.E, part.I
    return out


def _prolong_angles(
    tri: Triple, s_axis: Axis, t_axis: Axis, n_old: int, n_new: int, opt: Options, work: dict
) -> Triple:
    """Interpolate the segment tables from ``n_old`` to ``n_new`` directions.

    ``n_new`` is either ``n_old`` (nothing to do -- exact) or ``2 * n_old``, in
    which case the even-indexed new directions are *copies* of the old ones
    (exact) and each odd one is interpolated across ``theta_order`` old
    directions.  Because the frames differ, each old direction is read at the
    *spatial* point of the target, hence the 2-D ``(s, t)`` interpolation.
    """
    if n_new == n_old:
        return tri
    if n_new != 2 * n_old:
        raise ValueError(f"angular grids must nest by doubling, got {n_old} -> {n_new}")

    q = min(opt.theta_order, n_old)
    a_w = lagrange_weights(np.asarray(q / 2 - 0.5), q)  # midpoint weights
    j = np.arange(n_old)

    th_old = thetas_of(n_old)
    dir_old, perp_old = _frame(th_old)
    th_new = thetas_of(n_new)[1::2]
    dir_new, perp_new = _frame(th_new)

    s = s_axis.nodes[None, :, None]
    t = t_axis.nodes[None, None, :]
    shape = (n_old, s_axis.n, t_axis.n)
    acc = [np.empty(shape) for _ in range(3)]

    per_angle = s_axis.n * t_axis.n * q * opt.s_order * opt.t_order
    for lo, hi in _angle_chunks(n_old, per_angle, opt.max_block):
        x0 = s * perp_new[lo:hi, 0, None, None] + t * dir_new[lo:hi, 0, None, None]
        x1 = s * perp_new[lo:hi, 1, None, None] + t * dir_new[lo:hi, 1, None, None]
        block = [np.zeros(x0.shape) for _ in range(3)]
        for r in range(q):
            idx = (j[lo:hi] - (q // 2 - 1) + r) % n_old
            lead = idx[:, None, None]
            sp = x0 * perp_old[idx, 0, None, None] + x1 * perp_old[idx, 1, None, None]
            tp = x0 * dir_old[idx, 0, None, None] + x1 * dir_old[idx, 1, None, None]
            for slot, table in enumerate((tri.S, tri.E, tri.I)):
                block[slot] += a_w[r] * interp2d_stacked(
                    table, s_axis, t_axis, lead, sp, tp, opt.s_order, opt.t_order
                )
        for slot in range(3):
            acc[slot][lo:hi] = block[slot]
    n_targets = n_old * s_axis.n * t_axis.n
    work["interp_targets"] = work.get("interp_targets", 0) + 3 * n_targets
    work["interp_gathers"] = (
        work.get("interp_gathers", 0) + 3 * n_targets * q * opt.s_order * opt.t_order
    )

    out = []
    for new_half, old_table in zip(acc, (tri.S, tri.E, tri.I)):
        full = np.empty((n_new,) + old_table.shape[1:], dtype=float)
        full[0::2] = old_table
        full[1::2] = new_half
        out.append(full)
    return Triple(S=out[0], E=out[1], I=out[2])


def forward(
    f_img: Image,
    mu_img: Image,
    *,
    n_theta: int,
    n_s: int | None = None,
    s_axis: Axis | None = None,
    options: Options | None = None,
    support_radius: float | None = None,
) -> Sinogram:
    """Multilevel evaluation of the attenuated Radon transform of ``f_img``.

    Parameters
    ----------
    f_img, mu_img
        Activity and attenuation on a common grid.
    n_theta
        Number of output directions, uniform on ``[0, 2*pi)``.  Must be
        ``options.n_theta_min * 2**k``.
    n_s, s_axis
        Output offset grid.  Defaults to the image grid spacing across
        ``[-half_chord, half_chord]``.
    support_radius
        If given, asserts that the tables extend past the support so that
        zero-fill outside them is exact.
    """
    opt = options or Options()
    if f_img.x_axis != mu_img.x_axis or f_img.y_axis != mu_img.y_axis:
        raise ValueError("f and mu must live on the same grid")

    if s_axis is None:
        if n_s is None:
            n_s = int(round(2 * opt.half_chord / f_img.h)) + 1
        s_axis = Axis.from_endpoints(-opt.half_chord, opt.half_chord, n_s)

    n_levels = opt.n_levels
    if n_levels is None:
        n_levels = default_n_levels(f_img.h, opt.half_chord)

    sigma = int(opt.s_oversample)
    if sigma < 1:
        raise ValueError(f"s_oversample must be >= 1, got {sigma}")
    work_s_axis = (
        s_axis
        if sigma == 1
        else Axis(origin=s_axis.origin, h=s_axis.h / sigma, n=sigma * (s_axis.n - 1) + 1)
    )

    hier = build_hierarchy(
        n_theta_out=n_theta,
        n_theta_min=min(opt.n_theta_min, n_theta),
        n_levels=n_levels,
        s_axis=work_s_axis,
        half_chord=opt.half_chord,
        t_oversample=opt.t_oversample,
    )

    if support_radius is not None:
        if not opt.half_chord > support_radius:
            raise ValueError(
                f"half_chord={opt.half_chord} must exceed support_radius="
                f"{support_radius} for zero-fill in t to be exact"
            )
        if not work_s_axis.hi >= support_radius:
            raise ValueError(
                f"s grid reaches {work_s_axis.hi} < support_radius={support_radius}; "
                "zero-fill in s would clip nonzero values"
            )

    work: dict = {"table_points": 0, "level0_nodes": 0}
    tri = _build_level0(f_img, mu_img, hier.levels[0], work_s_axis, opt)
    work["level0_nodes"] = tri.shape[0] * tri.shape[1] * tri.shape[2] * opt.gl_order
    work["table_points"] += int(np.prod(tri.shape))

    for l in range(1, hier.n_levels + 1):
        prev, cur = hier.levels[l - 1], hier.levels[l]
        tri = _prolong_angles(
            tri, work_s_axis, prev.t_axis, prev.n_theta, cur.n_theta, opt, work
        )
        tri = merge_level(tri, hier.t_oversample)
        work["table_points"] += int(np.prod(tri.shape))
        if tri.shape != (cur.n_theta, work_s_axis.n, cur.t_axis.n):
            raise AssertionError(
                f"level {l}: table {tri.shape} != "
                f"{(cur.n_theta, work_s_axis.n, cur.t_axis.n)}"
            )

    # The full-chord segment is the node at t = 0, i.e. index nu/2 of the top level.
    mid = hier.t_oversample // 2
    if abs(hier.top.t_axis.nodes[mid]) > 1e-12:
        raise AssertionError("top-level midpoint is not at t = 0")
    sub = slice(None, None, sigma)
    return Sinogram(
        S=tri.S[:, sub, mid],
        E=tri.E[:, sub, mid],
        I=tri.I[:, sub, mid],
        thetas=hier.top.thetas,
        s_axis=s_axis,
        hierarchy=hier,
        work=work,
    )
