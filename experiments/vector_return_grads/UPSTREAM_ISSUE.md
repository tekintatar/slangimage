# [Bug] Silently WRONG input gradients when a `[Differentiable]` function contains a loop AND returns a vector — identical body with scalar return is exact

> FILED: https://github.com/shader-slang/slangpy/issues/1055 (2026-07-12). Repro: repro_vector_return_grads.py (verified 2026-07-12, L40S).
>
> **Status 2026-09-27:** closed upstream, fixed in the Slang compiler by
> [shader-slang/slang#12095](https://github.com/shader-slang/slang/pull/12095)
> (Slang issue [#12071](https://github.com/shader-slang/slang/issues/12071)).
> No released SlangPy wheel contains the fix. Reproduces on SlangPy 0.42.0. On
> 0.43.x the backward hangs instead
> ([#1167](https://github.com/shader-slang/slangpy/issues/1167)), because the
> loop below has a runtime trip count. Current status:
> [`docs/UPSTREAM_ISSUES.md`](../../docs/UPSTREAM_ISSUES.md).

## Summary

A `[Differentiable]` function that **both** contains a loop **and**
returns a vector (`float3`) produces silently wrong input gradients on
the torch autograd path. Forward values are correct; no error or
warning is raised — the backward results are simply garbage (wrong
magnitudes *and* wrong signs).

The exact same loop body with a **scalar** return gives exact
gradients. Both trigger conditions are necessary:

| loop | vector return | gradients |
|---|---|---|
| yes | yes | **WRONG** (silent) |
| yes | no (scalar) | exact — identical body, only the return differs |
| no | yes | exact (a loop-free float3-return function passes full `torch.autograd.gradcheck`) |

This is the worst failure class for a training library: a model built
on such a kernel trains on wrong gradients with no observable symptom
at the API level.

## Environment

- slangpy **0.42.0**, slangpy-torch 0.7.0 (native extension)
- torch 2.13.0+cu130, CUDA device via `spy.create_torch_device`
- Ubuntu 24.04, NVIDIA L40S, driver 580.126.16
- Python 3.12

## Reproduction

`python repro_vector_return_grads.py` — the script defines two kernels
with **identical bodies** (a toy 1D edge-aware filter: loop with
data-dependent weights). One returns `float3`, the other returns the
channel-weighted scalar sum. The losses built on them are the same
mathematical function of the input, which the script demonstrates by
printing both forward losses. Ground truth is central finite
differences **through the same slangpy forward** — no external
reference implementation is involved.

Observed output:

```text
forward loss (vec   ): 115.586731
forward loss (scalar): 115.586731

   element         FD     scalar        vec
 (0, 0, 0)     1.3477     1.3479     4.1044
 (2, 3, 1)     1.9127     1.9130    -3.2278
 (5, 6, 2)     3.0685     3.0682     3.4974
 (3, 0, 0)     1.0876     1.0878    -1.2643
 (1, 1, 2)     1.9489     1.9487    -2.6141

scalar-return max grad err: 3.56e-04  OK
vector-return max grad err: 5.14e+00  WRONG
```

Two provably-identical losses; the scalar-return gradient matches
finite differences to 3.6e-04, the vector-return gradient is wrong on
every probed element, including sign flips.

## Bisect notes (what it is NOT)

Found while implementing a differentiable bilateral filter; bisected
across five kernel variants:

- **Not** the vector *accumulator*: a variant carrying scalar
  accumulators `a0/a1/a2` through the loop and only assembling
  `float3(a0, a1, a2)` at the return is equally wrong.
- **Not** the data-dependent weights or the tensor indexing: the
  scalar-return control shares all of that code verbatim.
- **Not** loop bounds/`[MaxIters]`: same in both variants (and the
  loop is zero-based, so this is independent of #1051).
- A float3-return function **without** a loop (a LUT sampler with
  bilinear taps) passes full `torch.autograd.gradcheck`, so vector
  return alone is fine.

So the trigger is specifically: derivative propagation of a vector
return value out of a function whose body contains a loop. A plausible
suspect is the loop transposition/replay interacting with the
multi-component differential of the return value.

Tested on the torch autograd interop path on CUDA; not yet reduced on
the pure-Tensor `.bwds()` path.

## Expected

Exact gradients (as the scalar-return variant produces), or a compile
error if this shape of function is unsupported — not silently wrong
math.

## Workaround (for anyone else hitting this)

Restructure vector-valued neighborhood kernels as scalar-return
functions over an `(H, W, C)`-shaped dispatch grid — one thread per
output channel, each free to read all input channels (e.g. for a joint
bilateral range weight), accumulating and returning only its own
channel. Costs redundant weight computation per channel but produces
exact gradients (verified by gradcheck + finite differences).
