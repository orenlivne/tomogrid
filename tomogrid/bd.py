"""Brandt--Dym length doubling for the attenuated transform (2-D).

See docs/brandt_dym_scheme.md.  Segments are indexed by a starting point on a
*fixed* mesh and by an integer rise, so directions are equispaced in slope and
every segment starts and ends on the mesh.  Doubling the length then needs no
spatial interpolation at all: existing directions merge exactly, and new ones
need only a one-dimensional interpolation in the slope index.

Near-horizontal family (|slope| <= 1), swept along x.  The steep family is the
same algorithm applied to the transposed image.

Quantities are stored **normalised by arc length**, as in the paper, i.e. as
mean values along the segment rather than integrals.  This matters: the
interpolation is in the slope index, and an un-normalised integral carries the
factor ``sqrt(L^2 + (m h)^2)``, so interpolating it interpolates the arc length
too and leaves a purely geometric error that has nothing to do with the data.
With mean values a constant field is reproduced exactly.

Level ``k``:  horizontal extent ``L_k = 2^k L_0``
              x starts spaced ``L_k``      (count halves per level)
              y starts spaced ``h``        (constant)
              rises ``m in [-R_k, R_k]``, ``R_k = L_k / h``  (count doubles)
so the table size is the same at every level, and there are ``log n`` levels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .axis import Axis
from .image import Image
from .interp import lagrange_weights
from .quadrature import gauss_legendre, tail_integral_matrix
from .segment import Triple


@dataclass(frozen=True)
class BDLevel:
    index: int
    length: float  # horizontal extent L_k
    n_x: int  # number of x start positions
    rise: int  # R_k; rises run -R_k .. R_k

    @property
    def n_m(self) -> int:
        return 2 * self.rise + 1


@dataclass(frozen=True)
class BDGrid:
    """Geometry of the whole hierarchy."""

    x0: float
    y0: float
    h: float  # integration mesh spacing, transverse to the sweep
    n_y: int
    levels: tuple[BDLevel, ...]

    @property
    def top(self) -> BDLevel:
        return self.levels[-1]

    def x_axis(self, lv: BDLevel) -> Axis:
        return Axis(origin=self.x0, h=lv.length, n=lv.n_x)

    @property
    def y_axis(self) -> Axis:
        return Axis(origin=self.y0, h=self.h, n=self.n_y)

    def slopes(self, lv: BDLevel) -> np.ndarray:
        return np.arange(-lv.rise, lv.rise + 1) * (self.h / lv.length)

    def table_points(self) -> int:
        return sum(lv.n_x * self.n_y * lv.n_m for lv in self.levels)

    def describe(self) -> str:
        rows = [
            f"  L{lv.index}: length={lv.length:.5g} n_x={lv.n_x:5d} "
            f"n_slope={lv.n_m:6d} table={lv.n_x * self.n_y * lv.n_m:9d}"
            for lv in self.levels
        ]
        return "\n".join([
            f"BDGrid: {len(self.levels) - 1} doublings, h={self.h:.5g}, "
            f"n_y={self.n_y}",
            *rows,
        ])


def build_grid(*, half_width: float, image_h: float, sigma: int = 1,
               n_levels: int | None = None) -> BDGrid:
    """Hierarchy over ``[-half_width, half_width]^2``.

    ``sigma`` refines the integration mesh against the data mesh, ``h = H/sigma``
    -- the paper's ``H/h``, which controls the interpolation error as
    ``(h/H)^p`` per level.
    """
    if sigma < 1:
        raise ValueError(f"sigma must be >= 1, got {sigma}")
    h = image_h / sigma
    width = 2.0 * half_width
    n_y = int(round(width / h)) + 1
    length0 = 2.0 * image_h
    if n_levels is None:
        n_levels = int(round(np.log2(width / length0)))
    if n_levels < 0:
        raise ValueError("image too coarse for a hierarchy")

    levels = []
    for k in range(n_levels + 1):
        length = length0 * 2**k
        n_x = max(1, int(round(width / length)))
        rise = int(round(length / h))
        levels.append(BDLevel(index=k, length=length, n_x=n_x, rise=rise))
    return BDGrid(x0=-half_width, y0=-half_width, h=h, n_y=n_y,
                  levels=tuple(levels))


def build_level(f_img: Image, mu_img: Image, grid: BDGrid, k: int = 0, *,
                gl_order: int = 8, img_order: int = 4,
                transpose: bool = False) -> Triple:
    """Direct quadrature of the level-``k`` segments, shape ``(n_x, n_y, n_m)``.

    Integrals are with respect to arc length, so that concatenation is additive.
    Only ``k = 0`` is used by the solver; higher ``k`` is how the tests check the
    doubling step against a directly computed answer.
    """
    lv = grid.levels[k]
    gl_x, w = gauss_legendre(gl_order)
    C = tail_integral_matrix(gl_order)

    xs = grid.x_axis(lv).nodes[:, None, None, None]
    ys = grid.y_axis.nodes[None, :, None, None]
    rise = (np.arange(-lv.rise, lv.rise + 1) * grid.h)[None, None, :, None]
    u = 0.5 * (gl_x + 1.0)[None, None, None, :]  # 0..1 along the segment

    px = xs + lv.length * u
    py = ys + rise * u
    if transpose:
        px, py = py, px
    f_nodes = f_img.sample(px, py, order=img_order)
    mu_nodes = mu_img.sample(px, py, order=img_order)

    # Arc length of the segment, per rise.
    seg_len = np.sqrt(lv.length**2 + (np.arange(-lv.rise, lv.rise + 1) * grid.h) ** 2)

    # Normalised: 0.5 * sum(w_j .) is the mean over the segment, independent of
    # its arc length.  The attenuation inside the exponential is *not*
    # normalised -- it is a physical optical depth -- so it carries seg_len.
    S = 0.5 * (f_nodes @ w)
    E = 0.5 * (mu_nodes @ w)
    tail = (0.5 * seg_len)[None, None, :, None] * (mu_nodes @ C.T)
    I = 0.5 * np.einsum("...i,...i->...", w * f_nodes, np.exp(-tail))
    return Triple(S=S, E=E, I=I)


def build_level0(f_img: Image, mu_img: Image, grid: BDGrid, **kw) -> Triple:
    return build_level(f_img, mu_img, grid, 0, **kw)


def _shear(table: np.ndarray, n_y: int) -> np.ndarray:
    """``G[i, j, m] = table[i, j + m, m]`` with zero outside the mesh.

    This is where the second half of each segment is read: it starts where the
    first half ended, so its y index moves with the rise index.  Integer shear,
    no interpolation.
    """
    n_m = table.shape[2]
    m_vals = np.arange(n_m) - (n_m - 1) // 2
    rows = np.arange(n_y)[:, None] + m_vals[None, :]
    ok = (rows >= 0) & (rows < n_y)
    cols = np.arange(n_m)[None, :]
    out = table[:, np.clip(rows, 0, n_y - 1), cols]
    return np.where(ok[None, :, :], out, 0.0)


def _slope_stencil(n_m_out_odd: int, n_m_in: int, order: int):
    """Indices and weights interpolating the half-rise ``m + 1/2``.

    Returns ``(idx, w)`` of shape ``(order, n_m_out_odd)``: the level-``k`` rise
    indices contributing to each odd-rise target, with the stencil slid inside
    the slope range at the ends (one-sided, never extrapolating).
    """
    q = min(order, n_m_in)
    w = lagrange_weights(np.asarray(q / 2 - 0.5), q).ravel()
    # Odd target M = 2m+1 has half-rise m+1/2; m runs over the level-k rises
    # except the last, so the base index is m and the stencil is m-(q//2-1)..
    m_idx = np.arange(n_m_out_odd)  # 0-based level-k index of m
    base = m_idx - (q // 2 - 1)
    base = np.clip(base, 0, n_m_in - q)
    idx = base[None, :] + np.arange(q)[:, None]
    # Local coordinate of the target within the (possibly shifted) stencil.
    u = (m_idx + 0.5) - base
    wts = lagrange_weights(u, q)
    return idx, wts


def half_arc_lengths(grid: BDGrid, k: int) -> np.ndarray:
    """Arc length of each half of every level-``k+1`` segment.

    A level-``k+1`` segment with rise ``M`` over horizontal extent ``2L`` has two
    halves of extent ``L`` and rise ``M/2``, so each has arc length
    ``sqrt(L^2 + (M h / 2)^2)`` -- the *target's* geometry, not the geometry of
    whichever level-``k`` segments were interpolated to approximate it.
    """
    lv, nxt = grid.levels[k], grid.levels[k + 1]
    M = np.arange(-nxt.rise, nxt.rise + 1)
    return np.sqrt(lv.length**2 + (0.5 * M * grid.h) ** 2)


def refine(tri: Triple, grid: BDGrid, k: int, *, order: int = 4) -> Triple:
    """One length doubling, level ``k`` -> ``k+1``."""
    lv, nxt = grid.levels[k], grid.levels[k + 1]
    n_y = grid.n_y
    if tri.shape != (lv.n_x, n_y, lv.n_m):
        raise ValueError(f"table {tri.shape} != {(lv.n_x, n_y, lv.n_m)}")

    first = [tri.S, tri.E, tri.I]
    second = [_shear(a, n_y) for a in first]

    out = [np.empty((nxt.n_x, n_y, nxt.n_m)) for _ in range(3)]
    # x index 2i supplies the first half, 2i+1 the second.
    lo = slice(0, 2 * nxt.n_x, 2)
    hi = slice(1, 2 * nxt.n_x, 2)

    half_len = half_arc_lengths(grid, k)

    # Even targets M = 2m: the direction already exists, so this is exact.
    Sa, Ea, Ia = (a[lo] for a in first)
    Sb, Eb, Ib = (b[hi] for b in second)
    depth = Eb * half_len[None, None, 0::2]  # optical depth of the second half
    out[0][:, :, 0::2] = 0.5 * (Sa + Sb)
    out[1][:, :, 0::2] = 0.5 * (Ea + Eb)
    out[2][:, :, 0::2] = 0.5 * (np.exp(-depth) * Ia + Ib)

    # Odd targets M = 2m+1: interpolate the slope index, then concatenate.
    n_odd = nxt.n_m - lv.n_m
    idx, wts = _slope_stencil(n_odd, lv.n_m, order)
    Aq = [np.zeros((nxt.n_x, n_y, n_odd)) for _ in range(3)]
    Bq = [np.zeros((nxt.n_x, n_y, n_odd)) for _ in range(3)]
    for a in range(idx.shape[0]):
        wa = wts[a][None, None, :]
        for s in range(3):
            Aq[s] += wa * first[s][lo][:, :, idx[a]]
            Bq[s] += wa * second[s][hi][:, :, idx[a]]
    depth = Bq[1] * half_len[None, None, 1::2]
    out[0][:, :, 1::2] = 0.5 * (Aq[0] + Bq[0])
    out[1][:, :, 1::2] = 0.5 * (Aq[1] + Bq[1])
    out[2][:, :, 1::2] = 0.5 * (np.exp(-depth) * Aq[2] + Bq[2])
    return Triple(S=out[0], E=out[1], I=out[2])


# --- driver ----------------------------------------------------------------

# The shallow family sweeps along x and covers directions within 45 degrees of
# +x.  Reflecting and transposing the image gives the other three quadrants.
# All four are needed: the attenuated transform is *directed*, since the
# exponent is referenced to the exit end, so theta and theta+pi differ.
SWEEPS = (
    ("+x", False, False),
    ("-x", True, False),
    ("+y", False, True),
    ("-y", True, True),
)


@dataclass
class BDSweep:
    """Top-level result of one sweep."""

    name: str
    triple: Triple  # normalised (mean) quantities, shape (n_y, n_slope)
    p0: np.ndarray  # (n_y, n_slope, 2) segment starts, in image coordinates
    p1: np.ndarray  # (n_y, n_slope, 2) segment ends
    arc: np.ndarray  # (n_slope,) arc lengths

    def integrals(self) -> Triple:
        """Un-normalised (S, E, I): mean times arc length."""
        a = self.arc[None, :]
        return Triple(S=self.triple.S * a, E=self.triple.E * a,
                      I=self.triple.I * a)

    @property
    def pet(self) -> np.ndarray:
        t = self.integrals()
        return np.exp(-t.E) * t.S


def _oriented(img: Image, flip: bool, transpose: bool) -> Image:
    v = img.values
    ax, ay = img.x_axis, img.y_axis
    if transpose:
        v = v.T
        ax, ay = ay, ax
    if flip:
        v = v[::-1]
        ax = Axis(origin=-(ax.origin + ax.h * (ax.n - 1)), h=ax.h, n=ax.n)
    return Image(values=np.ascontiguousarray(v), x_axis=ax, y_axis=ay)


def _to_image_frame(p: np.ndarray, flip: bool, transpose: bool) -> np.ndarray:
    """Map sweep coordinates back to the original image frame."""
    out = p.copy()
    if flip:
        out[..., 0] = -out[..., 0]
    if transpose:
        out = out[..., ::-1]
    return out


def sweep(f_img: Image, mu_img: Image, grid: BDGrid, *, flip: bool = False,
          transpose: bool = False, order: int = 4, gl_order: int = 8,
          img_order: int = 4, name: str = "") -> BDSweep:
    """One family: build the base level and double up to the full width."""
    fo = _oriented(f_img, flip, transpose)
    mo = _oriented(mu_img, flip, transpose)
    tri = build_level(fo, mo, grid, 0, gl_order=gl_order, img_order=img_order)
    for k in range(grid.levels[-1].index):
        tri = refine(tri, grid, k, order=order)

    lv = grid.top
    M = np.arange(-lv.rise, lv.rise + 1)
    y = grid.y_axis.nodes
    zeros = np.zeros((y.size, M.size))
    p0 = np.stack([zeros + grid.x0, y[:, None] + zeros], axis=-1)
    p1 = np.stack([zeros + grid.x0 + lv.length,
                   y[:, None] + (M * grid.h)[None, :]], axis=-1)
    arc = np.sqrt(lv.length**2 + (M * grid.h) ** 2)
    return BDSweep(name=name, triple=Triple(S=tri.S[0], E=tri.E[0], I=tri.I[0]),
                   p0=_to_image_frame(p0, flip, transpose),
                   p1=_to_image_frame(p1, flip, transpose), arc=arc)


def forward(f_img: Image, mu_img: Image, *, sigma: int = 1, order: int = 4,
            gl_order: int = 8, img_order: int = 4,
            n_levels: int | None = None,
            half_width: float = 1.0) -> dict[str, BDSweep]:
    """Attenuated transform over all four direction families.

    Returns one :class:`BDSweep` per family, each carrying the segment endpoints
    so the result can be checked line by line against any reference.
    """
    if f_img.x_axis != mu_img.x_axis or f_img.y_axis != mu_img.y_axis:
        raise ValueError("f and mu must live on the same grid")
    grid = build_grid(half_width=half_width, image_h=f_img.h, sigma=sigma,
                      n_levels=n_levels)
    return {
        name: sweep(f_img, mu_img, grid, flip=flip, transpose=tr, order=order,
                    gl_order=gl_order, img_order=img_order, name=name)
        for name, flip, tr in SWEEPS
    }
