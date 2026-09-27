# What Brandt--Dym actually do, and what I had wrong

Source: A. Brandt and J. Dym, *Fast calculation of multiple line integrals*,
SIAM J. Sci. Comput. **20**(4):1417--1429, 1999.  Read in full.

## The indexing is the whole thing

Integrals are indexed by a **starting point on a fixed mesh** and a **slope**,
with directions equispaced in `tan θ` (`cot θ` for the steep family), so that

> "every integral starts and ends at a grid point (computing at equal intervals
> in `tan θ` ...; this is convenient for the length doubling stage)."

Write the near-horizontal family as: a segment of horizontal extent `L` starting
at mesh point `(x_i, y_j)` and ending at `(x_i + L, y_j + m h)`, where `h` is the
*integration* mesh spacing transverse to the sweep and `m` is an **integer**.
The slope is `m h / L`, so the slope set at level `k` has spacing `h / L_k` and
doubles in size when `L` doubles -- which is exactly the angular criterion,
derived in the paper from

```
 F(x, L, θ+ξ) = F(x, L, θ) + R,     |R| <= (ξ L / 4) max|grad g|
```

Their three resolution rules, all confirmed by my own analysis earlier:
angular resolution proportional to `L`; along-sweep point spacing inversely
proportional to `L`; transverse spacing independent of `L`, set by the data
bandwidth.  Hence **the number of integrals per level is independent of `L`**.

## The length-doubling step

For a level-`(k+1)` target with rise `M` over length `2L`:

* **`M` even** (`M = 2m`): the direction already exists at level `k`.  The two
  halves are the level-`k` segments `(x_{2i}, y_j, m)` and
  `(x_{2i+1}, y_{j+m}, m)` -- the second starts exactly where the first ends,
  both on the mesh.  **Exact, no interpolation.**
* **`M` odd**: the direction is new.  Interpolate the *direction* at length `L`,
  over `p` neighbouring integer rises, **at the same starting point**:

```
  A = sum_a w_a  T_k[2i,   j,      m_a]        (first half)
  B = sum_a w_a  T_k[2i+1, j+m_a,  m_a]        (second half)
```

  and then concatenate `A` and `B`.  The `w_a` are the midpoint Lagrange weights
  for the half-rise `M/2` among integers.

The second half's start point `y_{j+m_a}` moves *with* the rise index `m_a`,
because each direction's second half begins where its own first half ended.
That is the trick.  **No spatial interpolation occurs anywhere** -- only a
one-dimensional interpolation in the slope index, and an integer shear in
`(j, m)`.  Cost is `O(p)` per integral per level; the paper quotes about
`24 n log n` operations in total.

## Accuracy and work

Error per interpolation `~ (δ h / cH)^p`; over `l ~ log n` levels,

```
 l (δ h / cH)^p ≈ q          (q = target relative accuracy)
```

At `h = H` this gives `p ~ log(l/q)`, and since the work is `O(p n log n)`,

```
 W = O( n log n log(1/q) )
```

which is the form asked for.  (The paper instead optimises `p` against `h` and
reports `O(p n (log n)^{1+2/p})`, minimised at `p = 2 ln log n` to give
`O(n log n log log n)`.)

## What I had wrong

I indexed the tables by `(direction, s, t)` in a ray frame that **rotates with
the direction**.  Refining the direction then moves every sample point, so
recovering the table in the new frame is a scattered `d`-dimensional resample
costing `p^d` per entry -- 192 gathers in 2-D and 3072 in 3-D.  That single
choice produced both the dominant term in the error budget and the 3-D
crossover at `n ~ 200`.

Brandt--Dym's mesh does not rotate: it is fixed, denser than the data mesh in
one direction only, and the *slope* is what is refined.  The `p^d` never
appears.  My previous suggestion -- shear-factoring the rotation -- was treating
a symptom of an indexing choice that should not have been made.

## Carrying it to the attenuated transform

The concatenation identity acts elementwise in the line index and never mixes
lines, so it composes with the above unchanged.  Interpolate each half's triple
in the slope index, then concatenate exactly:

```
  S = A_S + B_S
  E = A_E + B_E
  I = exp(-B_E) * A_I + B_I
```

Interpolation is applied to `E` and to the end-referenced `I` -- the choice of
*what* to interpolate, established earlier, is unaffected; only the *indexing*
changes.  Cost: `3 fields x 2 halves x p` multiply-adds plus one exponential.

## Known costs of this formulation

* Directions are equispaced in slope, not angle, and the transverse spacing
  between neighbouring lines therefore varies with angle (`h` at `θ=0` against
  `h/sqrt 2` at `θ=π/4`).  The paper notes both the equispaced-grid variant and
  the simpler remedy: interpolate to an equispaced grid at the final stage only.
* Two families are needed, `|θ| <= π/4` swept along `x` and `|θ| >= π/4` swept
  along `y`, the second being the first applied to the transposed image.
