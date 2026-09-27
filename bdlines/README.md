# bdlines — a reproduction of Brandt & Dym (1999)

Plain line integrals, no attenuation, kept separate from `tomogrid` so that each
ingredient of the original method can be exercised on its own before any of it
is carried into the attenuated case.

| module | what it is |
|---|---|
| `mesh.py` | the three resolution rules and the level hierarchy |
| `discrete.py` | the Figure-1 discretisation `I_d`, with linear / cubic / spline interpolation along grid lines |
| `recursion.py` | the base level and the length doubling `I_a` |
| `truth.py` | the continuum integral `I_t` |
| `work.py` | operation counts |
| `experiments.py` | Tables 1 and 2 |

```
python -m bdlines.experiments 1     # Table 1
python -m bdlines.experiments 2     # Table 2
python -m pytest tests/test_bdlines.py -q
```

Results are in `results/table1.txt`, next to the published values.

## What reproduces

* **Exactness where the method claims it.** A direction already present at the
  base level is merged, never interpolated, and the normalised trapezoid rule
  composes exactly, so the recursion agrees with the direct discretisation to
  `1e-16`. Constant data is reproduced exactly at every level.
* **The convergence rate**, which is the paper's central claim: halving the
  integration mesh `h` reduces the approximation error by `2^p`. Measured over
  `H/h = 1, 2, 4`: linear doubling gives `3.8x, 3.9x`; cubic gives `16x, 19x`;
  quintic gives `2^6` until it reaches the discretisation floor.
* **The discretisation errors**, to within a few per cent of the published
  numbers: `0.085%` against `0.092%` at `w=4, L=8`, `0.019%` against `0.025%` at
  `L=32`, `0.0062%` against `0.008%` at `L=128`.
* **The near-independence of the approximation error from `L`**, and the
  `sigma^2` growth of the work against the `sigma^-p` fall in error, which is
  the trade the paper optimises to reach `O(n log n log log n)`.

## What does not

The absolute approximation error with **linear** doubling comes out about
`3.5x` larger than published (`0.45%` against `0.13%` at `w=4, L=8, H/h=1`),
with the same rates and the same dependence on `L`. Almost all of it is the
interpolation of the *second* half of each new direction, whose starting row
moves with the slope index; the first half contributes about five times less.
The published Figure 3(b) fixes which four length-`L` integrals are averaged,
and the text alone does not distinguish the reading used here from the
alternative. Both readings available from the text were implemented and give
*identical* results, because with linear interpolation the interpolation and
the averaging commute. (They do not commute for the attenuated merge, which is
why `tomogrid` interpolates the halves first and concatenates afterwards.)

The discrepancy does not affect the conclusions drawn from this package: with
cubic doubling the approximation error at `H/h = 1` is `0.015%`, already well
below the `0.085%` discretisation error, which is the criterion the paper sets.
