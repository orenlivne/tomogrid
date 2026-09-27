"""Independent dense-trapezoid reference for the 3-D transform (testing only)."""

from __future__ import annotations

import numpy as np

from .directions import DirectionGrid


def naive_transform3d(f_eval, mu_eval, dirs: DirectionGrid, s_nodes: np.ndarray,
                      half_chord: float = 1.0, n_samples: int = 1025,
                      dir_chunk: int = 4):
    """``(S, E, I)`` of shape ``(n_dir, n_s, n_s)`` by dense trapezoid.

    Shares no code with the solver beyond the direction grid itself.
    """
    s_nodes = np.asarray(s_nodes, dtype=float)
    n_s = s_nodes.size
    t = np.linspace(-half_chord, half_chord, n_samples)
    dt = t[1] - t[0]
    e1, e2, d = dirs.frames()

    S = np.empty((dirs.n_dir, n_s, n_s))
    E = np.empty_like(S)
    I = np.empty_like(S)
    s1 = s_nodes[None, :, None, None]
    s2 = s_nodes[None, None, :, None]
    tt = t[None, None, None, :]

    for lo in range(0, dirs.n_dir, dir_chunk):
        sl = slice(lo, lo + dir_chunk)
        a1, a2, ad = e1[sl], e2[sl], d[sl]
        pos = [s1 * a1[:, c, None, None, None] + s2 * a2[:, c, None, None, None]
               + tt * ad[:, c, None, None, None] for c in range(3)]
        fv = np.asarray(f_eval(*pos), dtype=float) + np.zeros(pos[0].shape)
        mv = np.asarray(mu_eval(*pos), dtype=float) + np.zeros(pos[0].shape)
        panel = 0.5 * (mv[..., :-1] + mv[..., 1:]) * dt
        tail = np.cumsum(panel[..., ::-1], axis=-1)[..., ::-1]
        A = np.concatenate([tail, np.zeros(tail.shape[:-1] + (1,))], axis=-1)

        def trapz(g):
            return dt * (g[..., 1:-1].sum(axis=-1) + 0.5 * (g[..., 0] + g[..., -1]))

        S[sl], E[sl], I[sl] = trapz(fv), A[..., 0], trapz(fv * np.exp(-A))
    return S, E, I


def image_evaluator3d(img, order: int = 4):
    def ev(x, y, z):
        return img.sample(x, y, z, order=order)

    return ev


def naive_segments3d(f_eval, mu_eval, p0, p1, n_samples: int = 1025,
                     chunk: int = 2048):
    """``(S, E, I)`` over arbitrary 3-D segments ``p0 -> p1`` by dense trapezoid.

    Arc-length integrals with the attenuation referenced to ``p1``; shares no
    code with the solver.
    """
    p0 = np.asarray(p0, dtype=float).reshape(-1, 3)
    p1 = np.asarray(p1, dtype=float).reshape(-1, 3)
    arc = np.linalg.norm(p1 - p0, axis=-1)
    u = np.linspace(0.0, 1.0, n_samples)
    du = u[1] - u[0]
    S = np.empty(p0.shape[0])
    E = np.empty_like(S)
    I = np.empty_like(S)
    for lo in range(0, p0.shape[0], chunk):
        sl = slice(lo, lo + chunk)
        a, b = p0[sl], p1[sl]
        pts = [a[:, c, None] + (b[:, c] - a[:, c])[:, None] * u[None, :]
               for c in range(3)]
        fv = np.asarray(f_eval(*pts), dtype=float) + np.zeros(pts[0].shape)
        mv = np.asarray(mu_eval(*pts), dtype=float) + np.zeros(pts[0].shape)
        ds = arc[sl][:, None] * du
        panel = 0.5 * (mv[:, :-1] + mv[:, 1:]) * ds
        tail = np.cumsum(panel[:, ::-1], axis=1)[:, ::-1]
        A = np.concatenate([tail, np.zeros((tail.shape[0], 1))], axis=1)

        def trapz(g):
            return (arc[sl] * du) * (g[:, 1:-1].sum(axis=1)
                                     + 0.5 * (g[:, 0] + g[:, -1]))

        S[sl], E[sl], I[sl] = trapz(fv), A[:, 0], trapz(fv * np.exp(-A))
    return S, E, I
