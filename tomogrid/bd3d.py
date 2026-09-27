"""Brandt--Dym length doubling in three dimensions, for the attenuated transform.

The construction is the two-dimensional one with a second slope.  A segment of
the shallow family has horizontal extent ``L`` along ``x`` and starts at a mesh
point ``(x_i, y_j, z_k)``, ending at ``(x_i + L, y_j + m h, z_k + p h)`` with
``m, p`` integers.  So directions are equispaced in the pair of slopes
``(m h / L, p h / L)``, every segment starts and ends on the mesh, and doubling
the length again needs no spatial interpolation:

* ``M, P`` both even -- the direction exists at level ``k`` and the two halves
  are ``[2i, j, k, M/2, P/2]`` and ``[2i+1, j+M/2, k+P/2, M/2, P/2]``.  Exact.
* otherwise -- interpolate the slope indices.  The targets form a grid in
  ``(m, p)``, so the two-dimensional interpolation separates into two
  one-dimensional passes and still costs ``O(order)`` per entry, not
  ``O(order^2)``.

The table grows by two per level (directions quadruple while segment centres
halve), so unlike in two dimensions the work is dominated by the top level,
which is the size of the output: the three-dimensional X-ray transform is a
four-dimensional object.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .axis import Axis
from .backend import to_host
from .image import Image3D
from .interp import lagrange_weights
from .quadrature import gauss_legendre, tail_integral_matrix
from .segment import Triple


@dataclass(frozen=True)
class BD3Level:
    index: int
    length: float
    n_x: int
    rise: int

    @property
    def n_m(self) -> int:
        return 2 * self.rise + 1


@dataclass(frozen=True)
class BD3Grid:
    x0: float
    h: float
    n_t: int  # transverse mesh points, in each of the two transverse axes
    levels: tuple[BD3Level, ...]

    @property
    def top(self) -> BD3Level:
        return self.levels[-1]

    @property
    def t_axis(self) -> Axis:
        return Axis(origin=self.x0, h=self.h, n=self.n_t)

    def x_nodes(self, lv: BD3Level) -> np.ndarray:
        return self.x0 + np.arange(lv.n_x) * lv.length

    def table_points(self) -> int:
        return sum(lv.n_x * self.n_t**2 * lv.n_m**2 for lv in self.levels)

    def describe(self) -> str:
        rows = [f"  L{lv.index}: length={lv.length:.5g} n_x={lv.n_x:4d} "
                f"n_slope={lv.n_m:5d}^2 table={lv.n_x * self.n_t**2 * lv.n_m**2:11d}"
                for lv in self.levels]
        return "\n".join([f"BD3Grid: h={self.h:.5g}, n_transverse={self.n_t}",
                          *rows])


def build_grid3d(*, half_width: float, image_h: float, sigma: int = 1,
                 n_levels: int | None = None) -> BD3Grid:
    if sigma < 1:
        raise ValueError(f"sigma must be >= 1, got {sigma}")
    h = image_h / sigma
    width = 2.0 * half_width
    n_t = int(round(width / h)) + 1
    length0 = 2.0 * image_h
    if n_levels is None:
        n_levels = int(round(np.log2(width / length0)))
    levels = []
    for k in range(n_levels + 1):
        length = length0 * 2**k
        levels.append(BD3Level(index=k, length=length,
                               n_x=max(1, int(round(width / length))),
                               rise=int(round(length / h))))
    return BD3Grid(x0=-half_width, h=h, n_t=n_t, levels=tuple(levels))


def build_level3d(f_img: Image3D, mu_img: Image3D, grid: BD3Grid, k: int = 0, *,
                  gl_order: int = 6, img_order: int = 4,
                  axes: tuple[int, int, int] = (0, 1, 2),
                  flip: bool = False, xp=np) -> Triple:
    """Level-``k`` table, shape ``(n_x, n_t, n_t, n_m, n_m)``, normalised."""
    lv = grid.levels[k]
    gl_x, w = gauss_legendre(gl_order)
    gl_x, w = xp.asarray(gl_x), xp.asarray(w)
    C = xp.asarray(tail_integral_matrix(gl_order))
    rise = xp.arange(-lv.rise, lv.rise + 1) * grid.h
    u = 0.5 * (gl_x + 1.0)

    xs = xp.asarray(grid.x_nodes(lv))[:, None, None, None, None, None]
    ys = xp.asarray(grid.t_axis.nodes)[None, :, None, None, None, None]
    zs = xp.asarray(grid.t_axis.nodes)[None, None, :, None, None, None]
    ms = rise[None, None, None, :, None, None]
    ps = rise[None, None, None, None, :, None]
    uu = u[None, None, None, None, None, :]

    px = xs + lv.length * uu
    py = ys + ms * uu
    pz = zs + ps * uu
    if flip:
        px = -px
    coords = [None, None, None]
    for src, arr in zip(axes, (px, py, pz)):
        coords[src] = arr
    f_nodes = f_img.sample(*coords, order=img_order, xp=xp)
    mu_nodes = mu_img.sample(*coords, order=img_order, xp=xp)

    arc = xp.sqrt(lv.length**2 + rise[:, None] ** 2 + rise[None, :] ** 2)
    S = 0.5 * (f_nodes @ w)
    E = 0.5 * (mu_nodes @ w)
    tail = (0.5 * arc)[None, None, None, :, :, None] * (mu_nodes @ C.T)
    I = 0.5 * xp.einsum("...i,...i->...", w * f_nodes, xp.exp(-tail))
    return Triple(S=S, E=E, I=I)


def _move_pair(arr, a, b, xp=np):
    """Bring axes ``a`` (position) and ``b`` (slope) to the end, in that order."""
    return xp.moveaxis(arr, (a, b), (-2, -1))


def _shear_axis(arr: np.ndarray, pos_axis: int, slope_axis: int,
                n_t: int, xp=np) -> np.ndarray:
    """Shift a position axis by its paired slope axis: ``out[.., j, .., m] =
    arr[.., j + m - R, .., m]``, zero off the mesh.

    An integer gather, done one axis pair at a time so no large index array is
    ever formed.  This is where the second half of each segment is read.
    """
    n_m = arr.shape[slope_axis]
    shift = xp.arange(n_m) - (n_m - 1) // 2
    rows = xp.arange(n_t)[:, None] + shift[None, :]
    ok = (rows >= 0) & (rows < n_t)
    rows = xp.clip(rows, 0, n_t - 1)
    cols = xp.arange(n_m)[None, :]
    work = _move_pair(arr, pos_axis, slope_axis, xp=xp)
    out = xp.where(ok, work[..., rows, cols], 0.0)
    return xp.moveaxis(out, (-2, -1), (pos_axis, slope_axis))


def _odd_weights(n_odd: int, n_in: int, order: int):
    """Stencil interpolating the half-rises ``m + 1/2`` in a slope index."""
    q = min(order, n_in)
    m_idx = np.arange(n_odd)
    base = np.clip(m_idx - (q // 2 - 1), 0, n_in - q)
    idx = base[None, :] + np.arange(q)[:, None]
    return idx, lagrange_weights((m_idx + 0.5) - base, q)


def _prolong_slope(arr: np.ndarray, axis: int, order: int, xp=np) -> np.ndarray:
    """Refine one slope axis, ``n -> 2n-1``: keep the old values, interpolate
    the midpoints.  One-dimensional, so the two slope axes cost ``O(order)``
    each rather than ``O(order^2)`` together."""
    n_in = arr.shape[axis]
    work = xp.moveaxis(arr, axis, -1)
    out = xp.empty(work.shape[:-1] + (2 * n_in - 1,))
    out[..., 0::2] = work
    idx, w = _odd_weights(n_in - 1, n_in, order)
    idx, w = xp.asarray(idx), xp.asarray(w)
    acc = xp.zeros(work.shape[:-1] + (n_in - 1,))
    for a in range(idx.shape[0]):
        acc += w[a] * work[..., idx[a]]
    out[..., 1::2] = acc
    return xp.moveaxis(out, -1, axis)


def half_arc3d(grid: BD3Grid, k: int) -> np.ndarray:
    """Arc length of each half of every level-``k+1`` segment, ``(n_M, n_P)``."""
    lv, nxt = grid.levels[k], grid.levels[k + 1]
    r = np.arange(-nxt.rise, nxt.rise + 1) * (0.5 * grid.h)
    return np.sqrt(lv.length**2 + r[:, None] ** 2 + r[None, :] ** 2)


def refine3d(tri: Triple, grid: BD3Grid, k: int, *, order: int = 4,
             xp=np) -> Triple:
    """One length doubling in three dimensions."""
    lv, nxt = grid.levels[k], grid.levels[k + 1]
    want = (lv.n_x, grid.n_t, grid.n_t, lv.n_m, lv.n_m)
    if tri.shape != want:
        raise ValueError(f"table {tri.shape} != {want}")
    lo = slice(0, 2 * nxt.n_x, 2)
    hi = slice(1, 2 * nxt.n_x, 2)

    first, second = [], []
    for arr in (tri.S, tri.E, tri.I):
        a = arr[lo]
        b = _shear_axis(arr, 1, 3, grid.n_t, xp=xp)  # y shifted by the m slope
        b = _shear_axis(b, 2, 4, grid.n_t, xp=xp)  # z shifted by the p slope
        b = b[hi]
        for ax in (3, 4):
            a = _prolong_slope(a, ax, order, xp=xp)
            b = _prolong_slope(b, ax, order, xp=xp)
        first.append(a)
        second.append(b)

    half = xp.asarray(half_arc3d(grid, k))[None, None, None, :, :]
    depth = second[1] * half
    return Triple(S=0.5 * (first[0] + second[0]),
                  E=0.5 * (first[1] + second[1]),
                  I=0.5 * (xp.exp(-depth) * first[2] + second[2]))


# --- driver ----------------------------------------------------------------

# Six families: the shallow cone about each of +/-x, +/-y, +/-z.  Together they
# cover the sphere, and all six are needed because the attenuated transform is
# directed.  ``axes`` says which image axis each of (sweep, transverse1,
# transverse2) maps to.
SWEEPS_3D = (
    ("+x", (0, 1, 2), False), ("-x", (0, 1, 2), True),
    ("+y", (1, 0, 2), False), ("-y", (1, 0, 2), True),
    ("+z", (2, 0, 1), False), ("-z", (2, 0, 1), True),
)


@dataclass
class BD3Sweep:
    name: str
    triple: Triple  # (n_t, n_t, n_m, n_m), normalised
    p0: np.ndarray  # (..., 3) segment starts in image coordinates
    p1: np.ndarray
    arc: np.ndarray  # (n_m, n_m)

    def integrals(self) -> Triple:
        a = self.arc[None, None, :, :]
        return Triple(S=self.triple.S * a, E=self.triple.E * a,
                      I=self.triple.I * a)

    @property
    def pet(self) -> np.ndarray:
        t = self.integrals()
        return np.exp(-t.E) * t.S


def _endpoints(grid: BD3Grid, axes, flip: bool):
    lv = grid.top
    r = np.arange(-lv.rise, lv.rise + 1) * grid.h
    t = grid.t_axis.nodes
    shape = (t.size, t.size, r.size, r.size)
    y = np.broadcast_to(t[:, None, None, None], shape)
    z = np.broadcast_to(t[None, :, None, None], shape)
    m = np.broadcast_to(r[None, None, :, None], shape)
    p = np.broadcast_to(r[None, None, None, :], shape)
    x0 = np.full(shape, grid.x0)
    x1 = np.full(shape, grid.x0 + lv.length)
    if flip:
        x0, x1 = -x0, -x1
    starts = [None] * 3
    ends = [None] * 3
    for src, a, b in zip(axes, (x0, y, z), (x1, y + m, z + p)):
        starts[src], ends[src] = a, b
    return np.stack(starts, axis=-1), np.stack(ends, axis=-1)


def forward3d(f_img: Image3D, mu_img: Image3D, *, sigma: int = 1,
              order: int = 4, gl_order: int = 6, img_order: int = 4,
              n_levels: int | None = None, half_width: float = 1.0,
              families=None, xp=np) -> dict[str, BD3Sweep]:
    """Attenuated 3-D X-ray transform over all six direction families."""
    grid = build_grid3d(half_width=half_width, image_h=f_img.h, sigma=sigma,
                        n_levels=n_levels)
    f_img, mu_img = f_img.to(xp), mu_img.to(xp)
    lv = grid.top
    r = np.arange(-lv.rise, lv.rise + 1) * grid.h
    arc = np.sqrt(lv.length**2 + r[:, None] ** 2 + r[None, :] ** 2)
    wanted = SWEEPS_3D if families is None else [
        s for s in SWEEPS_3D if s[0] in families]

    out = {}
    for name, axes, flip in wanted:
        tri = build_level3d(f_img, mu_img, grid, 0, gl_order=gl_order,
                            img_order=img_order, axes=axes, flip=flip, xp=xp)
        for k in range(lv.index):
            tri = refine3d(tri, grid, k, order=order, xp=xp)
        p0, p1 = _endpoints(grid, axes, flip)
        top = Triple(S=to_host(tri.S[0]), E=to_host(tri.E[0]),
                     I=to_host(tri.I[0]))
        out[name] = BD3Sweep(name=name, triple=top, p0=p0, p1=p1, arc=arc)
    return out
