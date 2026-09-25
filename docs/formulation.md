# The transform, and the multilevel algorithm

This is a reconstruction-from-scratch of the forward-evaluation algorithm, written
before Oren's 2009 notes with Achi Brandt were available.  Everything here is
derived from first principles in the Brandt–Lubrecht / Brandt–Dym style; where a
choice had to be guessed it is flagged **[GUESS]** so it can be checked against
the notes.

---

## 1. The forward model

Two fields live on the domain (we use the square `[-1,1]^2`, support inside a disk
of radius `R < 1`):

| symbol | meaning |
|---|---|
| `f(x) >= 0` | tracer activity (emission density) — the unknown in reconstruction |
| `mu(x) >= 0` | linear attenuation coefficient at the photon energy |

Parameterise a line by a direction and a signed offset.  With

```
omega(th)      = ( cos th,  sin th)      # direction of flight
omega_perp(th) = (-sin th,  cos th)      # offset direction
x(s,t)         = s*omega_perp + t*omega
```

the line `L(th,s)` is `{x(s,t) : t in R}`, traversed in the direction of increasing `t`.

### 1.1 SPECT / attenuated X-ray transform

A single photon emitted at `x(s,t)` flying in direction `omega` survives with
probability `exp(-int_t^inf mu)`.  Hence

```
                   /                 /   /  inf            \
 (R_mu f)(th,s) =  |   f(x(s,t)) exp | - |      mu(x(s,tau)) dtau |  dt      (1)
                   /                 \   / t                /
```

This is the **attenuated Radon transform** (a.k.a. attenuated X-ray transform).
The exponent depends on the emission point `t`, so **attenuation does not factor
out**.  Novikov (2002) gave the explicit inversion.

### 1.2 PET

PET detects two collinear 511 keV photons in coincidence.  Both must escape, in
opposite directions, so the survival probability is

```
exp(-int_t^inf mu) * exp(-int_{-inf}^t mu) = exp(-int_R mu)
```

which is **independent of `t`**.  Therefore

```
                   /     /        \   /
 (P_mu f)(th,s) = exp| - | mu     |  |  f(x(s,t)) dt                          (2)
                   \     / L      /   /
```

i.e. plain Radon transform of `f` times a per-line *attenuation correction
factor* (ACF).  Classical PET attenuation is a multiplicative constant per line.

### 1.3 Which one is "the PET transform with attenuation in the exponent"?

**[GUESS]** The problem you describe — smoothness exploited *both in the exponent
and in the main kernel* — only has content when the exponent varies **along** the
line, i.e. for (1), not (2).  So this code computes (1) as the primary object.
Plausible reasons your notes say "PET" and still have a live exponent:

1. the attenuated transform (1) is used as the generic emission-tomography model,
   PET being the degenerate case;
2. time-of-flight PET, where the coincidence is localised along the line and the
   two one-sided attenuations no longer combine into a constant;
3. an iterative-reconstruction forward operator in which the ACF is folded in
   per-line but the *implementation* still wants fast line integrals.

The implementation carries **three** quantities per segment, so it delivers both
models at once and costs nothing extra to keep the option open:

```
 S = plain line integral of f     -> Radon f
 E = plain line integral of mu    -> Radon mu
 I = attenuated line integral (1) -> SPECT model
 PET model (2) = exp(-E) * S
```

Note that `S` and `E` are the *same* algorithm run on a trivial attenuation, so
they act as strong internal consistency checks on `I`.

---

## 2. The one identity the whole algorithm rests on

Work with **segments**, not whole lines.  For `a < b` along the ray `(th,s)` define

```
 E[a,b] = int_a^b mu(x(s,tau)) dtau
 S[a,b] = int_a^b f (x(s,t))   dt
 I[a,b] = int_a^b f(x(s,t)) exp( - int_t^b mu(x(s,tau)) dtau ) dt            (3)
```

Crucially `I` references the attenuation to the segment's **own far end `b`**, so
every weight lies in `(0,1]`.  Then for any `c` in `(a,b)`:

```
 E[a,b] = E[a,c] + E[c,b]
 S[a,b] = S[a,c] + S[c,b]
 I[a,b] = exp(-E[c,b]) * I[a,c] + I[c,b]                                     (4)
```

*Proof of the third line:* split the integral at `c`.  On `[a,c]` the exponent
`int_t^b = int_t^c + int_c^b`, and `int_c^b` is independent of `t`, so it pulls
out as `exp(-E[c,b])`; what remains is exactly `I[a,c]`.  On `[c,b]` the integral
is `I[c,b]` verbatim.  QED

Equivalently a 2x2 lower-triangular transfer matrix multiplies:

```
            [ exp(-E[a,b])   0 ]
 M[a,b] =   [ I[a,b]         1 ] ,        M[a,b] = M[a,c] . M[c,b]
```

`(4)` is **exact** — no quadrature, no interpolation.  This is the "semigroup"
that lets segment length double from level to level.  It is also *stabilising*:
a merge damps any error in the left half by `exp(-E[c,b]) <= 1`, so strong
attenuation makes the recursion better conditioned, not worse.

---

## 3. What is smooth, and how smooth

Fix a segment length `l` and a spatial centre `x`, and look at the segment
integral as a function of direction:

```
 Q(th) = int_{-l/2}^{l/2} f(x + tau*omega(th)) dtau
```

Take a single Fourier mode `f = exp(i k.y)` with `|k| = k`:

```
 Q(th) = exp(i k.x) * l * sinc( (l*k/2) cos(th - phi) ),    phi = arg k
```

The `th`-dependence enters only through `u(th) = (l k/2) cos(th-phi)`, and
`sinc` has unit scale in `u`, with `|du/dth| <= l k / 2`.  Hence `Q` varies in
`th` on the scale

```
 dth ~ 2 / (l k)         i.e.      dth * l * k ~ 1                          (5)
```

With the finest scale representable on an image grid of spacing `h` being
`k_max ~ pi/h`, the **angular sampling criterion** is

```
 dth_l * l_l  <~  c * h ,      c = O(1)                                      (6)
```

Two more axes, same reasoning:

* **transverse (`s`)**: `Q` inherits `f`'s transverse scale `h`, so `ds ~ h` at
  *every* level;
* **along-ray (`t`)**: `Q` is an *average over length `l`*, hence smooth on scale
  `l`, so `dt ~ l_l`.  This is the smoothing that buys the whole algorithm.

### 3.0 ...and where that counting is too optimistic

Both of those last two need a constant factor, and getting it wrong caps the
accuracy at ~1e-3 no matter what else is done.  Measured, not guessed
(`tests/test_accuracy.py`):

* **`dt = l` is not enough.**  The box average of length `l` multiplies the
  spectrum by `sinc(k l / 2)`, which is still `0.64` at the Nyquist frequency
  `k = pi/l` of a grid with `dt = l`.  The tables alias.  Taking
  `dt = l/2` (`t_oversample = 2`, i.e. segments that *overlap by half*) puts the
  Nyquist frequency exactly on the first **zero** of that sinc.  Measurements
  show that `nu = 2` is both necessary and sufficient: `nu = 4, 8` buy nothing.
* **`ds = h` is not enough either.**  Transverse to the ray there is no
  smoothing at all, so sampling the table at exactly the image spacing leaves an
  `O(1)` fraction of the content at Nyquist.  `s_oversample = sigma` refines the
  internal `s` grid (the requested output grid is recovered by subsampling).
  This turns out to be the *dominant* table knob.

Measured algorithmic error on a `33^2` image, `n_theta = 32`, `n_theta_min = 16`
(offcentre blob through a smooth attenuating blob):

| `sigma
u` | 2 | 4 | 8 |
|---|---|---|---|
| 1 | 9.2e-3 | | |
| 2 | 1.0e-3 | | |
| 4 | 1.4e-4 | 1.9e-4 | 2.0e-4 |

So: `nu = 2` always, and `sigma` is the knob.  Cost is linear in `sigma * nu`.

The most important measurement is that **the error converges under grid
refinement at fixed `sigma`, `nu` and fixed base angular grid** -- observed order
rising through 1.6, 1.9, 2.9 as `n` goes `17 -> 33 -> 65 -> 129`.  Raising the
stencil orders from 4 to 6 does *not* raise it, which says the limit is the
`C^0` kinks of the piecewise-cubic image interpolant rather than the stencils;
a `C^2` image representation (cubic B-spline) should restore fourth order and is
the obvious next improvement.

For the attenuated quantity `I` the same counting holds with an extra factor per
derivative coming from differentiating the exponent (`d_th E ~ l^2 |grad mu|`),
so the interpolation constant degrades with attenuation *contrast* but the order
of accuracy is unchanged.  This is measured in `tests/test_accuracy.py`.

### 3.1 The correct quantities to interpolate

This is where a naive implementation dies.

* Interpolate **`E`**, never `exp(-E)`.  `E` is an additive, `l`-smoothed line
  integral of `mu` with bounded derivatives; `exp(-E)` spans orders of magnitude
  and has enormous relative derivatives once total attenuation is `O(10)`.
  Interpolate the exponent, *then* exponentiate.  ("Smoothness in the exponent.")
* Interpolate **`I` as defined in (3)**, i.e. referenced to the segment's own far
  end.  Referencing to the midpoint (or to a global origin) introduces factors
  `exp(+l*||mu||/2)` that amplify without bound; end-referencing keeps all
  weights in `(0,1]`.  ("Smoothness in the main kernel.")
* Interpolate `S` directly (it is just a smoothed Radon integral).

---

## 4. The algorithm

Levels `l = 0..L`.  Level `l` stores, for every direction `th_k` on an angular
grid `Theta_l`, a 2-D table over `(s,t)` of the triple `(S,E,I)` for the segment
of length

```
 l_l = 2^l * l_0 ,       l_0 ~ h ,     l_L = 2*half_chord  (covers any chord)
```

centred at `x(s,t)`.  Grids: `ds = h` (all levels), `dt = l_l`,
`n_t^(l) = 2^(L-l)`, and `|Theta_l| = n_th^(l)`.

**Level 0 (finest segments), computed directly.**  Each segment is `~h` long.
Put `m` Gauss–Legendre nodes on it, sample `f` and `mu` there by tensor-product
Lagrange interpolation from the image grid, and use

```
 E = (l/2) sum_j w_j mu_j
 S = (l/2) sum_j w_j f_j
 A_i = (l/2) sum_j C_ij mu_j       ~ int_{t_i}^{b} mu   (C = exact tail-integral
                                     matrix of the Lagrange basis on the nodes)
 I = (l/2) sum_i w_i f_i exp(-A_i)
```

so the inner attenuation integral is obtained *spectrally* from the same `m`
samples.  Cost `O(m + m^2)` per segment, order `m` in `l_0`.

**Level `l-1` -> `l`, two sub-steps.**

*(A) angular prolongation.*  `Theta_{l-1}` is nested in `Theta_l` (exact
doubling), so half the directions are copied verbatim — **exactly**.  Each new
direction `th` is the midpoint of two old ones; its table is obtained by
interpolation in `th` over a stencil of `q` old directions, where for each old
direction `th'` the value is needed at the *spatial* point
`x = s*omega_perp(th) + t*omega(th)`, i.e. at ray coordinates
`s' = x.omega_perp(th')`, `t' = x.omega(th')` in the old frame — a `p_s x p_t`
tensor-product interpolation of the old table.  Interpolating in `th` at fixed
*ray* coordinates would be wrong: the segment centre would rotate on a circle of
radius `|x| = O(1)` and (5) would force `dth <~ h` at every level, destroying the
complexity.

*(B) merge.*  At each direction, merge `t`-neighbours with `(4)`.  With
`dt_l = l_l / nu` on the node grid `t_j = -H + j*dt_l`, the level-`l` segment
sitting at level-`(l-1)` index `2j` is exactly the union of the level-`(l-1)`
segments at `2j -/+ nu/2` -- an integer offset precisely because `nu` is **even**.
So **no `t`-interpolation and no approximation** occurs here.  (This is why odd
`nu`, and `nu = 1`, are rejected outright: they would put the half-segment
centres off-grid and force an interpolation into the one step that is exact.)

Zero-fill outside the tables is *exact*: if `|s'| > R` the whole line misses the
support, and if `|t'| > half_chord >= R + l_l/2` the segment does.

**Scheduling of the angular refinement.**  This is the part that is easy to get
backwards.  From (6), short segments need *few* directions and long segments need
*many*.  So take

```
 n_th^(l) = min( n_theta_out , n_theta_min * 2^l )                           (7)
```

All the angular refinement then happens at the **bottom** levels, in lockstep
with the doubling of `l_l`, and the criterion is *the same at every refinement*:

```
 dth_{l-1} * l_{l-1} = (2 pi / (n_theta_min 2^(l-1))) * (l_0 2^(l-1))
                     = 2 pi l_0 / n_theta_min                                (8)
```

— independent of `l`.  With `l_0 ~ h` this is `~ (2 pi / n_theta_min) h`, so
**`n_theta_min` alone controls the interpolation error**, at linear cost.  Once
`(7)` saturates at `n_theta_out`, the remaining top levels are pure exact merges:
no error is introduced there at all.  (Running with
`n_theta_min = n_theta_out` performs *zero* interpolation and reproduces the
direct method — which is how the reference in the tests is obtained.)

---

## 5. Complexity

Work at level `l` is proportional to the table size `n_th^(l) * n_s * n_t^(l)`:

```
 l <= l*  (refining):  (n_theta_min 2^l) * n_s * (M / 2^l) = n_theta_min * M * n_s
 l >  l*  (coasting):  n_theta_out * n_s * (M / 2^l)        -> geometric decay
```

with `M = 2^L ~ 2*half_chord/h` and `l* = log2(n_theta_out / n_theta_min)`.
Summing,

```
 total ~ n_theta_min * M * n_s * ( l* + 2 )
       = O( N log N ),      N ~ M * n_s ~ n_grid^2
```

against `O(n_theta_out * n_s * M) = O(N^1.5)` for direct evaluation.  Constant
per refinement level: `q * p_s * p_t` gathers.

---

## 6. Status / open questions for the notes

* **[GUESS]** SPECT-type exponent, as argued in §1.3.
* **[GUESS]** the level-0 kernel (spectral tail-integral on Gauss–Legendre nodes).
  A cheaper 2nd-order variant (piecewise-linear `f`, piecewise-constant `mu`, with
  the resulting `int (a+bt) exp(-mu(d-t)) dt` in closed form) is also implemented
  in `segment.py` for comparison.
* Not yet done: 3-D (straightforward — segment doubling is dimension-independent;
  the direction grid becomes 2-D on the sphere and (6) becomes an area criterion,
  so `n_dir^(l) ~ (l_l/h)^2` and the per-level work is again constant), and the
  adjoint/backprojection (the transpose of every step above; the merge transposes
  to a scatter, the prolongation to anterpolation).
