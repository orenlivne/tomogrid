# Running the recursion on a GPU

Short answer: yes, the algorithm ports to a GPU almost mechanically, and the
port is already in the source — every kernel takes an `xp` argument and uses
only array operations whose CuPy meaning matches NumPy's. What it does *not*
do is use tensor cores, and it should not try to. The recursion trades
arithmetic for memory traffic, and traffic is what a GPU has least of relative
to its arithmetic. That is the whole story, and the numbers below say how much
of the algorithmic saving survives it.

> **Nothing in this document was run on a GPU.** No CUDA device, no cloud
> instance, no CuPy — the container has none of them. Every device number
> below is a *roofline*: counted traffic or counted arithmetic divided by a
> vendor's published peak figure. It bounds what a good implementation could
> reach and says nothing about what this code does reach. The only measured
> numbers here are the host (NumPy) timings and the single-versus-double
> precision comparison, both labelled as such. `experiments/gpu_bench.py` is
> the script that would produce real numbers; see
> [Actually measuring it](#actually-measuring-it).

## What the kernels are

One doubling, level `k` to `k+1`, is three whole-array operations on tables of
shape `(n_x, n_y, n_slope)` (`tomogrid/bd.py:refine`):

1. **an integer shear** — `G[i, j, m] = T[i, j+m, m]`, zero off the mesh. A
   gather with indices known in advance, contiguous in the fastest axis for
   each fixed `m`.
2. **a `q`-tap weighted sum along the slope axis**, for the directions that do
   not exist at level `k`. Away from the ends of the slope range the stencil
   offsets and the weights are the *same for every target*: the interior of
   `_slope_stencil` is a constant-coefficient `q`-tap convolution. Only `O(q)`
   columns at each end are one-sided.
3. **a pointwise merge**, `I = (e^{-E_b ell} I_a + I_b)/2` with the averages of
   `S` and `E` — one exponential and eight flops per entry.

No scatter, no atomics, no data-dependent branching, no indirection that is not
known before the kernel launches, and no loop over grid points anywhere: the
Python loops run over the `q` stencil taps only. In three dimensions
(`tomogrid/bd3d.py:refine3d`) the two slope axes are refined one at a time —
sum factorisation, `O(q)` per entry instead of `O(q^2)` — which is the
tensor-product structure being exploited at the algorithm level.

The base level is different in kind: it interpolates the two fields at every
Gauss point of every short segment with a tensor-product Lagrange stencil, out
of an image that is a few megabytes and stays in cache. That part is
arithmetic-dense.

## The port

`tomogrid/backend.py` resolves a namespace, and every kernel takes `xp`:

```python
from tomogrid.backend import resolve
from tomogrid.bd import forward
sweeps = forward(f, mu, sigma=2, xp=resolve("auto"))   # cupy if a device is up
```

`resolve("auto")` returns CuPy when a device is present and NumPy otherwise,
so the same script runs either way. Tables live on the device for the whole
recursion; only the top level comes back.

Without a device, what can still be checked is the thing that actually breaks
ports — reaching for an array function that CuPy spells differently or means
differently. `tests/test_backend.py` runs the full 2-D and 3-D transforms
through a recording namespace and asserts that every array function they
touched is in `backend.PORTABLE`, and that threading `xp` changes no digit of
the answer.

## Single precision is enough

Consumer cards run fp64 at a 64th of their fp32 rate, so this decides whether
the algorithm is a datacenter-only affair. The merge is a damped combination —
its weight `e^{-E ell}` lies in `(0, 1]` — so roundoff is carried down the
levels, not amplified. Running the whole pipeline in fp32 against the fp64
answer (`experiments/gpu_roofline.py --fp32`):

| n | doublings | max abs. difference | relative |
|---|---|---|---|
| 65 | 5 | 9.84e-08 | 2.6e-07 |
| 129 | 6 | 1.02e-07 | 2.6e-07 |
| 257 | 7 | 1.20e-07 | 3.1e-07 |

Flat in the number of levels, and two to three orders of magnitude below the
discretisation error the method is run at. fp32 is safe, which halves the
traffic and puts the recursion on any card.

## The parallelism does not thin out with level

This is the part that separates the construction from a multigrid cycle,
where the coarse levels hold too few points to fill a device and the coarsest
are latency-bound. Here the table is the same size at every level by
construction: segment start positions halve exactly as directions double. At
n = 513, `sigma = 2` the nine levels hold between 2.10 and 2.36 million
entries per quantity per family — a spread of 12% — so every level offers
about 2.5e7 independent outputs across three quantities and four families,
and the last offers as many as the first. Occupancy never collapses.

## The adjoint keeps the same access pattern

Iterative reconstruction applies the adjoint as often as the forward
operator, and on a GPU the transpose of a gather is usually a scatter, which
means atomics. Not here. With `mu` fixed the operator is linear in `f` and
the merge coefficients `exp(-E l)` are computed once in the forward sweep.
The transpose of an integer shear is the opposite integer shear, and the
transpose of a banded operator is banded, so the anterpolation dual to the
slope interpolation is again an `O(p)`-tap gather over the transposed
stencil. Same three kernels, same access pattern, same cost. Not implemented
yet.

## Where it sits on the roofline

*Counted*, four direction families, `sigma = 2`, `order = 4`, fp64
(`experiments/gpu_roofline.py --counts`). No row is a measurement:

| n | lines | base GB | recursion GB | base flop/B | recursion flop/B | live GB |
|---|---|---|---|---|---|---|
| 65 | 132,612 | 0.005 | 0.049 | 35.7 | 0.70 | 0.011 |
| 129 | 527,364 | 0.018 | 0.235 | 35.7 | 0.71 | 0.043 |
| 257 | 2,103,300 | 0.074 | 1.090 | 35.7 | 0.71 | 0.170 |
| 513 | 8,400,900 | 0.294 | 4.962 | 35.7 | 0.71 | 0.680 |

0.71 flops per byte. Every datacenter GPU has a balance point between 4 and 10
flops per byte in fp64, so the recursion is bandwidth-bound by an order of
magnitude, and its time on a device is its traffic divided by HBM bandwidth.

The host implementation is nowhere near that bound: NumPy moves the same 273 MB
(one family, n = 257) in 0.38 s, an effective 0.7 GB/s against a socket that can
do a hundred times more, because every array expression is a separate pass with
its own temporaries. A fused device kernel is the point of the exercise; so is
a fused *host* kernel, and the gap is a fair measure of how much of the
reported CPU timings is implementation rather than algorithm.

## Against direct quadrature, on the same hardware

The comparison that matters is not flops but time, and the two methods are
bounded by different resources. Direct quadrature of the same lines reads an
image that fits in cache and is therefore arithmetic-bound; the recursion
streams tables and is bandwidth-bound. Rooflines at n = 513, 8.4 M lines. **Counted traffic and arithmetic divided
by vendor peak figures — no device was involved in producing any cell:**

| device | precision | recursion | direct | ratio |
|---|---|---|---|---|
| RTX 4090 | fp64 | 10.9 ms | 919 ms | 84 |
| RTX 4090 | fp32 | 2.6 ms | 12.7 ms | 4.9 |
| A100 80GB | fp64 | 2.6 ms | 122 ms | 47 |
| A100 80GB | fp32 | 1.3 ms | 53.7 ms | 42 |
| H100 SXM5 | fp64 | 1.6 ms | 34.9 ms | 22 |
| H100 SXM5 | fp32 | 0.8 ms | 15.6 ms | 20 |

Three things to read off it.

* The advantage is real and grows with `n` — it doubles with every refinement,
  as the `O(n / log n)` arithmetic ratio says it should.
* It is smaller than the arithmetic ratio, because the saving is in a resource
  the machine has in abundance. The clearest case is the 4090 in fp32: 82.6
  Tflop/s against 1 TB/s is a balance point of 82 flops per byte, so brute
  force is nearly free there and the recursion only wins by five. **The
  algorithm trades arithmetic for memory traffic, and GPU hardware has been
  moving the other way for a decade.**
* Which says where the remaining work is. At 0.71 flops per byte there is a
  lot of slack: fusing two doublings so a tile of the table is refined twice
  in registers halves the traffic, and recomputing rather than storing the
  sheared copy removes a third of it. Both trade flops for bytes, which is
  the right direction on this hardware.

## Why not tensor cores

The slope interpolation is a matrix applied along one axis, `X P^T`, so it is
tempting to make it a GEMM and put it on tensor cores. The arithmetic says no.
`P` is banded with `q = 4` nonzeros per row and `n_slope` reaches 2049, so the
dense product does about `n_slope / q ~ 500` times the arithmetic; TF32 tensor
cores run about 8 times the fp32 rate. Blocking the slope axis into tiles of
`b = 32` softens it to `(b + q - 1) / q ~ 8.75` times the arithmetic for 8
times the rate — a wash, at reduced precision, on a kernel that is bandwidth-
bound anyway. Keep the banded form and fuse.

The tensor-product structure does pay, twice, but at the algorithm level: the
3-D slope refinement separates into two 1-D passes, and the base level is a
tensor-product interpolation, which is the arithmetic-dense part that would
keep the units busy while the recursion waits on memory.

## What limits it: memory, in 3-D only

In two dimensions the working set at n = 513 is 0.68 GB in fp64, half that in
fp32 — nothing. In three dimensions the output is itself four-dimensional:
one direction family of six at n = 65, `sigma = 2` holds 79 GB in fp64 and 40
GB in fp32. That is the capacity of a large device for a sixth of the
directions of a small volume. Clinical three-dimensional work has to be
blocked over families and over slabs of the transverse plane, which the
construction permits — families are independent and the transverse index is
never interpolated — or streamed. Neither is implemented.

## Actually measuring it

`experiments/gpu_bench.py` runs the transform on whatever device is present
and reports wall time, counted traffic, and the effective bandwidth the two
imply. Run it on the host first; it prints `numpy (no device)` and works,
which is how you check it before renting anything.

```bash
python -m experiments.gpu_bench --grids 129 257            # host, fp64
pip install cupy-cuda12x                                   # on a CUDA box
python -m experiments.gpu_bench --grids 257 513 --fp32     # device, fp32
```

The number to read is the last column, the recursion's traffic over its wall
time, against the device's published bandwidth. That ratio says how much of
the hardware the kernels reach. A speed-up against the NumPy path would say
almost nothing: NumPy reaches under 1 GB/s of a socket's ~100, because every
array expression is a separate pass with its own temporaries, so any device
would look good against it.

Instances matching the three devices in the tables, for reference: an A100
80GB is `a2-ultragpu-1g` on GCP or `p4de.24xlarge` on AWS; an H100 is
`a3-highgpu-1g` or `p5.48xlarge`; a 4090 is not offered by either and comes
from the smaller GPU hosts. An hour on a single A100 is enough for the whole
2-D table.

## What is not done

No kernel has been fused, the adjoint is not implemented, the 3-D blocking is
not implemented, and nothing has run on a device. The port is written and the
portability is tested; that is all.
