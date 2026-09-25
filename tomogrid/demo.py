"""Worked example: ``python -m tomogrid.demo``."""

from __future__ import annotations

import time

import numpy as np

from . import phantoms as ph
from .cost import estimate_work
from .image import sample_function, square_grid
from .multilevel import Options, forward


def main() -> None:
    n_img, n_theta, n_s = 65, 64, 65
    xa, ya = square_grid(n_img)
    f_fun = ph.ACTIVITIES["three_blobs"][0]
    mu_fun = ph.ATTENUATIONS["high_contrast"][0]
    f = sample_function(f_fun, xa, ya)
    mu = sample_function(mu_fun, xa, ya)

    print(f"image {n_img}x{n_img}, sinogram {n_theta}x{n_s}\n")
    results = {}
    for label, n_min in (("multilevel", 16), ("direct", n_theta)):
        opt = Options(n_theta_min=n_min)
        t0 = time.perf_counter()
        sg = forward(f, mu, n_theta=n_theta, n_s=n_s, options=opt, support_radius=0.6)
        dt = time.perf_counter() - t0
        w = estimate_work(
            sg.hierarchy, gl_order=opt.gl_order, img_order=opt.img_order,
            theta_order=opt.theta_order, s_order=opt.s_order, t_order=opt.t_order,
        )
        results[label] = sg
        print(f"{label:>11}: {dt:6.2f} s   {w['gathers'] / 1e6:9.1f} M gathers")

    ml, direct = results["multilevel"], results["direct"]
    scale = np.abs(direct.S).max()
    print(
        f"\nalgorithmic error vs direct (same discretisation):\n"
        f"    S {np.abs(ml.S - direct.S).max() / scale:.2e}"
        f"    E {np.abs(ml.E - direct.E).max() / np.abs(direct.E).max():.2e}"
        f"    I {np.abs(ml.I - direct.I).max() / scale:.2e}"
    )
    print(f"\npeak chord attenuation E = {direct.E.max():.2f} "
          f"(exp(-E) = {np.exp(-direct.E.max()):.2e})")
    print(f"\n{ml.hierarchy.describe()}")


if __name__ == "__main__":
    main()
