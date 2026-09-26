# Why the 3-D schedule loses to direct evaluation, and what fixes it

Two questions from the 3-D measurements: why is the multilevel method *slower*
than direct evaluation, and is that just an artefact of the small grids that fit
in this container?

**Short answer.** It is not the grid size. It is one constant, and the fix is a
change of line parameterisation taken from the Brandt--Dym lineage.

> **Provenance.** Brandt & Dym, *Fast calculation of multiple line integrals*,
> SIAM J. Sci. Comput. 20(4):1417--1429 (1999), could not be read directly: every
> host carrying it (the Weizmann site, Semantic Scholar, SIAM, arXiv, Springer)
> is blocked by this container's egress policy. What follows is built from
> secondary descriptions of the algorithm plus derivation, and should be checked
> against the paper.

---

## 1. The accounting

There are exactly two ways to obtain one entry of a segment table.

**Build it from the image.** Put `gl` Gauss--Legendre nodes on the segment and
sample `f` and `mu` at each with a `p_img`-wide stencil per axis:

```
  build  =  2 fields * gl * p_img^d
         =  2 * 8 * 4^2 =  256   gathers   (2-D, gl = 8)
         =  2 * 6 * 4^3 =  768   gathers   (3-D, gl = 6)
```

**Interpolate it from neighbouring directions.** This is the prolongation: an
angular stencil in `d-1` angular coordinates, each source direction requiring a
resample of its `(s..., t)` table because the ray frame has rotated:

```
  prolong = 3 fields * q^(d-1) * p_s^(d-1) * p_t
          = 3 * 4 * 4 * 4       =  192   gathers   (2-D)
          = 3 * 4^2 * 4^2 * 4   = 3072   gathers   (3-D)
```

| | build | prolongate | ratio |
|---|---|---|---|
| 2-D | 256 | 192 | **0.75** |
| 3-D | 768 | 3072 | **4.00** |

The multilevel method's entire saving is that level 0 is built at
`n_dir_min` directions instead of `n_dir_out`; every entry above level 0 is paid
for by prolongation. So it can only win if prolongating an entry is cheaper than
building it. In 2-D it is, barely. **In 3-D it is four times more expensive to
interpolate a table entry from its neighbours than to compute it from scratch.**

Measured split at `n = 33`, `m_out = 9`, `m_min = 5`:

```
  multilevel:  level0  31.6 G  +  prolongation 283.5 G  =  633 Gflop
  direct    :  level0 102.5 G  +  prolongation   0.0 G  =  215 Gflop
```

The 3.2x saving on level 0 is swamped by the prolongation.

## 2. Why a bigger grid does not help

Both terms are proportional to the same table size
`n_dir * n_s^(d-1) * n_t`, so refining the image does not move the ratio. The
cost model confirms it: with the direction count held fixed the ratio is `0.34`
at `n = 9` and still `0.34` at `n = 513`.

It improves only when the *angular* resolution tracks the spatial resolution, so
that `n_dir_out / n_dir_min` grows and more of level 0 is saved:

| n | 33 | 65 | 129 | 257 |
|---|---|---|---|---|
| direct/multilevel | 0.33 | 0.43 | 0.74 | **1.38** |

So the 3-D crossover sits near `n ~ 200`, against `n = 65` in 2-D. Reaching it
is not an option here: `257^3` is ~2.6 Pflop for direct evaluation, which at the
~26 M gathers/s this NumPy implementation sustains is of order a year. `33^3`
already costs ten minutes.

Going larger is the right instinct and the wrong lever. The lever is the 3072.

## 3. What the Brandt--Dym lineage does differently

The tables here are indexed by `(direction, s, t)` in a ray frame that **rotates
with the direction**. Refining the direction therefore rotates the frame, and
recovering the table at the new frame's sample points is a scattered
`d`-dimensional interpolation. That single decision produces the `p^d` factor --
and, separately, it is also the dominant term in the 2-D *error* budget.

The dyadic-block formulation avoids it. Lines are indexed by **slope and
intercept** against a fixed Cartesian block structure, and a line crossing a
block of height `2H` crosses its two stacked half-blocks at intercepts `b` and
`b + s*H`:

```
  R_2H[s, b]  =  R_H^bot[s, b]  +  R_H^top[s, b + s*H]
```

Choose the slope set so that `s*H` is a whole number of columns at every level --
which is exactly the rule that the angular resolution doubles when the
integration length doubles -- and the merge is a **pure indexed addition**. No
interpolation, no rotation, no multiplication. This is the structure behind the
discrete Radon transform of Gotz--Druckmuller and Brady, and behind the
pseudopolar/slant-stack transforms; the recursion is exact by construction.

## 4. It carries over to the attenuated case

The concatenation identity of the paper is **elementwise in the line index**: it
combines two sub-segments of the *same* line and never mixes lines. So it
transfers to the dyadic indexing untouched, with the top block read at the
shifted intercept:

```
  E[s, b]  <-  E_bot[s, b]  +  E_top[s, b + s*H]
  S[s, b]  <-  S_bot[s, b]  +  S_top[s, b + s*H]
  I[s, b]  <-  exp(-E_top[s, b + s*H]) * I_bot[s, b]  +  I_top[s, b + s*H]
```

Still exact. The exponential is the only multiplication. Cost per entry per
level drops from 3072 gathers to a handful of flops.

**What it costs.** The output lands on a slope-equispaced (pseudopolar) direction
set rather than a uniform angular grid, on digital rather than exact Euclidean
lines. Converting to uniform angles is a separate 1-D interpolation per offset.
Accuracy then rests on the level-0 quadrature and the digital-line model instead
of on interpolation between directions -- which changes the error analysis
completely, and is where the `log(1/eps)` question (currently answered
`W ~ eps^-0.61`, see `experiments/results/sampling.csv`) should be re-asked.

**What it is.** Not a patch. The segment-triple algebra, the level-0 kernel and
the quadrature all survive; the hierarchy, the direction grids and the
prolongation are replaced. In 3-D the same construction applies with two slopes
and two intercepts.

## 5. An intermediate option

If the dyadic rewrite is too large a step, the rotation can instead be factored
into shears -- a 2-D rotation is three 1-D shears, a 3-D rotation nine -- which
turns `p^d` into `9p` and, with the angular stencil applied as `d-1` successive
1-D passes, takes the 3-D constant from 3072 to about 864. Modelled effect:

| n | 33 | 65 | 129 | 257 |
|---|---|---|---|---|
| direct/multilevel, now | 0.33 | 0.43 | 0.74 | 1.38 |
| direct/multilevel, sheared | **0.92** | **1.40** | **2.51** | **4.81** |

That moves the 3-D crossover from `n ~ 200` to `n ~ 60` and makes 3-D runs at
reachable sizes worthwhile. It is a much smaller change than the dyadic rewrite,
and strictly weaker: it reduces the constant, where the dyadic formulation
removes the interpolation altogether.
