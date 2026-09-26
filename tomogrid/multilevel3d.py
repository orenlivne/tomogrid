"""Three-dimensional multilevel evaluation of the attenuated X-ray transform.

Everything that is dimension-independent is reused: the segment triple and its
exact concatenation (:mod:`tomogrid.segment`), the Gauss-Legendre kernel, and
the nested ``t``-grid merge.  What changes is the angular variable: directions
live on the sphere (:mod:`tomogrid.directions`), each ray carries *two*
transverse offsets ``(s1, s2)``, and the criterion of eq. (11) becomes an area
criterion, so the number of directions grows as ``(l/h)^2`` while the number of
segment centres falls as ``h/l``.  Work per level therefore *grows* by two per
level rather than staying flat, and the total is dominated by the top level,
i.e. by the size of the output itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .axis import Axis
from .directions import DirectionGrid
from .image import Image3D
from .interp import interp3d_stacked, lagrange_weights, stencil
from .quadrature import gauss_legendre
from .segment import Triple, merge_level, quad_spectral


@dataclass
class Options3D:
    m_min: int = 3  # nodes per face axis of the coarsest direction grid
    n_levels: int | None = None
    half_chord: float = 1.0
    gl_order: int = 6
    img_order: int = 4
    dir_order: int = 4  # angular stencil, per face axis
    s_order: int = 4  # transverse stencils (both s1 and s2)
    t_order: int = 4
    t_oversample: int = 2
    max_block: int = 1 << 22


@dataclass
class Level3D:
    index: int
    seg_len: float
    t_axis: Axis
    dirs: DirectionGrid

    @property
    def table_size(self) -> int:
        return self.dirs.n_dir * self.t_axis.n


@dataclass
class Hierarchy3D:
    levels: tuple[Level3D, ...]
    s_axis: Axis
    half_chord: float
    t_oversample: int

    @property
    def n_levels(self) -> int:
        return len(self.levels) - 1

    @property
    def top(self) -> Level3D:
        return self.levels[-1]

    @property
    def refine_levels(self) -> tuple[int, ...]:
        return tuple(l.index for l in self.levels[1:]
                     if l.dirs.m != self.levels[l.index - 1].dirs.m)

    def max_angular_displacement(self) -> float:
        out = 0.0
        for i in self.refine_levels:
            prev = self.levels[i - 1]
            out = max(out, prev.dirs.max_angular_step() * prev.seg_len)
        return out

    def describe(self) -> str:
        rows = [
            f"  L{l.index}: seg_len={l.seg_len:.5g} n_t={l.t_axis.n:5d} "
            f"n_dir={l.dirs.n_dir:6d} table={l.table_size * self.s_axis.n ** 2:11d}"
            for l in self.levels
        ]
        return "\n".join([
            f"Hierarchy3D: {self.n_levels} levels, half_chord={self.half_chord}, "
            f"n_s={self.s_axis.n} (x2), ds={self.s_axis.h:.5g}",
            f"  refine at levels {self.refine_levels}, "
            f"max angular step * seg_len = {self.max_angular_displacement():.4g}",
            *rows,
        ])


def build_hierarchy3d(*, m_out: int, m_min: int, n_levels: int, s_axis: Axis,
                      half_chord: float = 1.0, t_oversample: int = 2) -> Hierarchy3D:
    """Levels with nested direction grids ``m_l = (m_min - 1) 2^l + 1``, clamped
    at ``m_out``.  ``m_out`` must lie in that sequence."""
    if m_min < 2:
        raise ValueError(f"m_min must be >= 2, got {m_min}")
    if m_out < m_min:
        raise ValueError(f"m_out={m_out} < m_min={m_min}")
    ratio = (m_out - 1) / (m_min - 1)
    k = round(math.log2(ratio)) if ratio > 0 else -1
    if k < 0 or abs(2.0**k - ratio) > 1e-12:
        raise ValueError(
            f"m_out must equal (m_min-1)*2^k + 1; got m_out={m_out}, m_min={m_min}")
    if k > n_levels:
        raise ValueError(f"need at least {k} levels to reach m_out={m_out}")
    nu = int(t_oversample)
    if nu < 2 or nu % 2:
        raise ValueError(f"t_oversample must be even >= 2, got {t_oversample}")

    levels = []
    for l in range(n_levels + 1):
        n_seg = 2 ** (n_levels - l)
        seg_len = 2.0 * half_chord / n_seg
        t_axis = Axis(origin=-half_chord, h=seg_len / nu, n=nu * n_seg + 1)
        m = min(m_out, (m_min - 1) * 2**l + 1)
        levels.append(Level3D(index=l, seg_len=seg_len, t_axis=t_axis,
                              dirs=DirectionGrid(m=m)))
    return Hierarchy3D(tuple(levels), s_axis, half_chord, nu)


@dataclass
class Sinogram3D:
    """Result indexed ``[i_dir, i_s1, i_s2]``."""

    S: np.ndarray
    E: np.ndarray
    I: np.ndarray
    dirs: DirectionGrid
    s_axis: Axis
    hierarchy: Hierarchy3D | None = None
    work: dict = field(default_factory=dict)

    @property
    def pet(self) -> np.ndarray:
        return np.exp(-self.E) * self.S

    @property
    def attenuated(self) -> np.ndarray:
        return self.I


def _chunks(n, per_item, budget):
    step = max(1, min(n, int(budget // max(1, per_item))))
    for lo in range(0, n, step):
        yield lo, min(lo + step, n)


def _build_level0(f_img, mu_img, level, s_axis, opt):
    gl_x, _ = gauss_legendre(opt.gl_order)
    e1, e2, d = level.dirs.frames()
    n_s, n_t = s_axis.n, level.t_axis.n
    s1 = s_axis.nodes[None, :, None, None, None]
    s2 = s_axis.nodes[None, None, :, None, None]
    t = (level.t_axis.nodes[None, None, None, :, None]
         + (0.5 * level.seg_len) * gl_x[None, None, None, None, :])

    shape = (level.dirs.n_dir, n_s, n_s, n_t)
    out = Triple(*[np.empty(shape) for _ in range(3)])
    per = n_s * n_s * n_t * opt.gl_order
    for lo, hi in _chunks(level.dirs.n_dir, per, opt.max_block):
        a1, a2, ad = e1[lo:hi], e2[lo:hi], d[lo:hi]
        pos = []
        for c in range(3):
            pos.append(s1 * a1[:, c, None, None, None, None]
                       + s2 * a2[:, c, None, None, None, None]
                       + t * ad[:, c, None, None, None, None])
        part = quad_spectral(
            f_img.sample(*pos, order=opt.img_order),
            mu_img.sample(*pos, order=opt.img_order),
            level.seg_len,
        )
        out.S[lo:hi], out.E[lo:hi], out.I[lo:hi] = part.S, part.E, part.I
    return out


def _prolong_directions(tri, s_axis, t_axis, old: DirectionGrid, new: DirectionGrid,
                        opt, work):
    """Interpolate the tables from ``old`` to ``new`` directions.

    Directions of ``old`` reappear in ``new`` and are copied exactly.  Each
    remaining direction is interpolated over a ``dir_order x dir_order`` stencil
    of old directions *within its own face*, each contributing a 3-D resample of
    its ``(s1, s2, t)`` table at the target's spatial points.
    """
    if new.m == old.m:
        return tri
    if new.m != 2 * old.m - 1:
        raise ValueError(f"direction grids must nest: {old.m} -> {new.m}")

    keep = new.nested_parent_slice()
    fresh = np.flatnonzero(~keep)
    nf, na, nb = new.indices()
    old_axis, q = old.axis, min(opt.dir_order, old.m)

    e1n, e2n, dn = new.frames()
    e1o, e2o, do = old.frames()
    a_t = new.axis.nodes[na[fresh]]
    b_t = new.axis.nodes[nb[fresh]]
    ia, wa = stencil(a_t, old_axis, q, mode="shift")
    ib, wb = stencil(b_t, old_axis, q, mode="shift")
    face = nf[fresh]

    n_s, n_t = s_axis.n, t_axis.n
    s1 = s_axis.nodes[None, :, None, None]
    s2 = s_axis.nodes[None, None, :, None]
    tt = t_axis.nodes[None, None, None, :]

    acc = [np.empty((fresh.size, n_s, n_s, n_t)) for _ in range(3)]
    per = n_s * n_s * n_t * q * q * opt.s_order**2 * opt.t_order
    for lo, hi in _chunks(fresh.size, per, opt.max_block):
        sl = slice(lo, hi)
        pos = []
        for c in range(3):
            pos.append(s1 * e1n[fresh[sl], c, None, None, None]
                       + s2 * e2n[fresh[sl], c, None, None, None]
                       + tt * dn[fresh[sl], c, None, None, None])
        block = [np.zeros(pos[0].shape) for _ in range(3)]
        for u in range(q):
            for v in range(q):
                src = (face[sl] * old.m + ia[u, sl]) * old.m + ib[v, sl]
                w = (wa[u, sl] * wb[v, sl])[:, None, None, None]
                lead = src[:, None, None, None]
                c1 = sum(pos[c] * e1o[src, c, None, None, None] for c in range(3))
                c2 = sum(pos[c] * e2o[src, c, None, None, None] for c in range(3))
                ct = sum(pos[c] * do[src, c, None, None, None] for c in range(3))
                for slot, table in enumerate((tri.S, tri.E, tri.I)):
                    block[slot] += w * interp3d_stacked(
                        table, s_axis, s_axis, t_axis, lead, c1, c2, ct,
                        opt.s_order, opt.s_order, opt.t_order)
        for slot in range(3):
            acc[slot][sl] = block[slot]

    n_targets = fresh.size * n_s * n_s * n_t
    work["interp_targets"] = work.get("interp_targets", 0) + 3 * n_targets
    work["interp_gathers"] = (work.get("interp_gathers", 0)
                              + 3 * n_targets * q * q
                              * opt.s_order**2 * opt.t_order)

    out = []
    for half, old_table in zip(acc, (tri.S, tri.E, tri.I)):
        full = np.empty((new.n_dir,) + old_table.shape[1:])
        full[keep] = old_table
        full[fresh] = half
        out.append(full)
    return Triple(*out)


def forward3d(f_img: Image3D, mu_img: Image3D, *, m_out: int, n_s: int | None = None,
              s_axis: Axis | None = None, options: Options3D | None = None,
              support_radius: float | None = None) -> Sinogram3D:
    """Multilevel attenuated X-ray transform of a 3-D field."""
    opt = options or Options3D()
    if s_axis is None:
        if n_s is None:
            n_s = int(round(2 * opt.half_chord / f_img.h)) + 1
        s_axis = Axis.from_endpoints(-opt.half_chord, opt.half_chord, n_s)
    n_levels = opt.n_levels
    if n_levels is None:
        n_levels = max(0, int(math.ceil(math.log2(2.0 * opt.half_chord / f_img.h))))

    hier = build_hierarchy3d(m_out=m_out, m_min=min(opt.m_min, m_out),
                             n_levels=n_levels, s_axis=s_axis,
                             half_chord=opt.half_chord,
                             t_oversample=opt.t_oversample)
    if support_radius is not None:
        if not opt.half_chord > support_radius:
            raise ValueError("half_chord must exceed support_radius")
        if not s_axis.hi >= support_radius:
            raise ValueError("s grid does not cover the support")

    work = {"table_points": 0}
    tri = _build_level0(f_img, mu_img, hier.levels[0], s_axis, opt)
    work["level0_nodes"] = int(np.prod(tri.shape)) * opt.gl_order
    work["table_points"] += int(np.prod(tri.shape))

    for l in range(1, hier.n_levels + 1):
        prev, cur = hier.levels[l - 1], hier.levels[l]
        tri = _prolong_directions(tri, s_axis, prev.t_axis, prev.dirs, cur.dirs,
                                  opt, work)
        tri = merge_level(tri, hier.t_oversample)
        work["table_points"] += int(np.prod(tri.shape))
        want = (cur.dirs.n_dir, s_axis.n, s_axis.n, cur.t_axis.n)
        if tri.shape != want:
            raise AssertionError(f"level {l}: {tri.shape} != {want}")

    mid = hier.t_oversample // 2
    return Sinogram3D(S=tri.S[..., mid], E=tri.E[..., mid], I=tri.I[..., mid],
                      dirs=hier.top.dirs, s_axis=s_axis, hierarchy=hier, work=work)
