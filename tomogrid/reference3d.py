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
