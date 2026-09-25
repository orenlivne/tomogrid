# tomogrid

Multilevel `O(N log N)` evaluation of the **attenuated Radon transform** — the
emission-tomography (PET/SPECT) forward operator, where the attenuation sits in
the exponent:

```
                /                 /   /  inf                  \
 (R_mu f)(th,s) =  |  f(x(s,t)) exp | - |     mu(x(s,tau)) dtau |  dt
                /                 \   / t                      /
```

`N` is the number of image pixels (≈ the number of sinogram bins). Direct
evaluation is `O(N^1.5)`.

See **[docs/formulation.md](docs/formulation.md)** for the derivation: the
semigroup identity that makes the merge exact, the smoothness/sampling analysis
that sets the level schedule, and the measured accuracy behaviour.

## Quick start

```python
import numpy as np
from tomogrid import Options, forward, phantoms as ph
from tomogrid.image import sample_function, square_grid

xa, ya = square_grid(129)                      # 129^2 image on [-1,1]^2
f  = sample_function(ph.bump(amp=1.0, center=(0.1, -0.05), radius=0.45), xa, ya)
mu = sample_function(ph.plateau(amp=1.5, r_flat=0.35, r_zero=0.6), xa, ya)

sino = forward(f, mu, n_theta=128, n_s=129,
               options=Options(n_theta_min=16), support_radius=0.6)

sino.I            # attenuated (SPECT) transform     -- the hard one
sino.pet          # exp(-E) * S, the PET model
sino.S, sino.E    # plain Radon transforms of f and mu
print(sino.hierarchy.describe())
```

`python -m tomogrid.demo` prints a worked example with timings and errors.

## What the method does

Segments, not whole lines. For a segment `[a,b]` carry three numbers — `S` (the
plain integral of `f`), `E` (the plain integral of `mu`) and `I` (the attenuated
integral, referenced to the segment's **own far end**, so every weight is in
`(0,1]`). Then concatenation is *exact*:

```
E[a,b] = E[a,c] + E[c,b]
S[a,b] = S[a,c] + S[c,b]
I[a,b] = exp(-E[c,b]) * I[a,c] + I[c,b]
```

Segment length doubles level by level. A segment of length `l` resolves
direction only to `dtheta ~ h/l`, so *short* segments need *few* directions:
angles are doubled at the bottom levels in lockstep with the segment length, and
the top levels coast at full angular resolution doing nothing but exact merges.
Every level then costs the same, and there are `log N` of them.

Interpolating the right quantities is what makes it work: **`E`, never
`exp(-E)`** (the exponent is the smooth object; exponentiate afterwards), and
**`I` end-referenced, never midpoint-referenced** (which would introduce
unbounded `exp(+l||mu||/2)` factors).

## Accuracy knobs

| option | effect | cost |
|---|---|---|
| `n_theta_min` | base angular grid; controls the angular interpolation error | linear (in fact sublinear) |
| `s_oversample` | internal `ds = output ds / sigma`; the **dominant** knob | linear |
| `t_oversample` | `dt = seg_len / nu`; must be even, `nu = 2` is optimal | linear |
| `n_levels` | finest segment length; default ≈ one image cell | one level each |
| `gl_order`, `img_order`, `theta_order`, `s_order`, `t_order` | quadrature and stencil widths | — |

`n_theta_min = n_theta` disables every interpolation and recovers the direct
`O(N^1.5)` method, which is how the tests obtain an exact same-discretisation
reference.

## Status

Done: 2-D forward transform, full test suite, cost model.
Not done: 3-D (segment doubling is dimension-independent; the direction grid
becomes 2-D on the sphere), and the adjoint/backprojection (the transpose of
every step — the merge transposes to a scatter, the prolongation to
anterpolation).

## Tests

```
pip install numpy pytest
python -m pytest tests -q              # fast suite
python -m pytest tests -q --runslow    # + convergence studies and the full 36-case battery
```
