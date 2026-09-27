# Running the recursion on a GPU

Short answer: yes, the algorithm ports to a GPU almost mechanically, and the
port is already in the source — every kernel takes an `xp` argument and uses
only array operations whose CuPy meaning matches NumPy's. What it does *not*
do is use tensor cores, and it should not try to. The recursion trades
arithmetic for memory traffic, and traffic is what a GPU has least of relative
to its arithmetic. That is the whole story, and the numbers below say how much
of the algorithmic saving survives it.

There is no GPU in the container these numbers were produced in, so nothing
here is a benchmark. Everything is either counted from array shapes or
measured on the host; the device figures are rooflines from published peak
bandwidth and peak rate, which is a bound and is labelled as one.

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

## Where it sits on the roofline

Counted from shapes, four direction families, `sigma = 2`, `order = 4`, fp64
(`experiments/gpu_roofline.py --counts`):

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
streams tables and is bandwidth-bound. Rooflines at n = 513, 8.4 M lines:

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

## What is not done

Measuring any of this on a device. The port is written and the portability is
tested, but no kernel has been fused and nothing has run on hardware, so the
rooflines above are bounds on what a good implementation could reach, not
claims about what this code does reach. On a machine with a GPU:

```bash
pip install cupy-cuda12x
python -c "
from tomogrid.backend import resolve
from tomogrid.bd import forward
import tomogrid.phantoms as ph
from tomogrid.image import sample_function, square_grid
xa, ya = square_grid(513)
f  = sample_function(ph.ACTIVITIES['three_blobs'][0], xa, ya)
mu = sample_function(ph.ATTENUATIONS['high_contrast'][0], xa, ya)
print(forward(f, mu, sigma=2, xp=resolve('auto'))['+x'].triple.I.shape)
"
```
