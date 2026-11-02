# [Bug] Backward dispatch segfaults when a [Differentiable] function has a loop with a negative runtime start value

> FILED: https://github.com/shader-slang/slangpy/issues/1051 (2026-07-12)
>
> **Status 2026-09-27:** closed upstream, fixed in the Slang compiler by
> [shader-slang/slang#12299](https://github.com/shader-slang/slang/pull/12299)
> (Slang issue [#12070](https://github.com/shader-slang/slang/issues/12070)).
> No released SlangPy wheel contains the fix. Upstream triage found that the
> trigger is any runtime (non-constant) loop start; a constant negative start
> is fine. On SlangPy 0.43.x this reproducer fails to compile the backward
> instead of crashing, and its `control` variant hangs
> ([#1167](https://github.com/shader-slang/slangpy/issues/1167)); use 0.42.0 to
> see the behavior described below. The script creates a Metal device: on
> Linux, change `spy.DeviceType.metal` in `main()` to `cuda` or `vulkan`.
> Current status: [`docs/UPSTREAM_ISSUES.md`](../../docs/UPSTREAM_ISSUES.md).

## Summary

Calling `Function.bwds(...)` — or triggering the backward through the
PyTorch autograd integration — **segfaults (SIGSEGV)** when the
`[Differentiable]` Slang function contains a loop whose start value is a
negative runtime expression, e.g. the canonical convolution form:

```slang
[MaxIters(17)]
for (int dx = -radius; dx <= radius; ++dx) { ... }   // radius is a no_diff int param
```

The **forward pass is always correct**, in both loop forms. Rewriting the
loop to a zero-based form with an offset produces correct gradients through
the same call path, so the crash appears to be in backward code generation
or its runtime handling of negative-start loops, not in the API usage.

## Environment

- slangpy **0.42.0** (pip), slangpy-torch 0.7.0 (crash reproduces with and
  without the native torch extension, and on the pure-Tensor path)
- Reproduced on **all three backends**:
  - Metal — macOS 26.5, Apple M3 Max
  - CUDA — Ubuntu 24.04, NVIDIA L40S, driver 580.126.16, CUDA toolkit 12.0
  - Vulkan — same L40S machine
- Python 3.12, numpy 2.x, torch 2.13 (torch not required to reproduce)

## Reproduction

Save the script below as `repro.py`.
`python repro.py` → forward prints correctly, then SIGSEGV at the `bwds()`
call (exit 139). `python repro.py control` runs the mathematically
identical zero-based-loop variant → completes with correct gradients.
Core of the trigger:

```slang
import slangpy;

// SEGFAULTS in backward:
[Differentiable]
float neg_start(no_diff int2 pix, IDiffTensor<float, 2> src,
                no_diff int radius, no_diff int width)
{
    float acc = 0.0;
    [MaxIters(17)]
    for (int dx = -radius; dx <= radius; ++dx)
        acc += src[uint2(clamp(pix.x + dx, 0, width - 1), pix.y)];
    return acc;
}

// WORKS (identical math, zero-based loop):
//   for (int i = 0; i < 2 * radius + 1; ++i) { int dx = i - radius; ... }
```

```python
src = spy.Tensor.from_numpy(device, arr).with_grads()
out = module.neg_start(spy.grid((H, W)), src, 2, W)     # forward: correct
out = out.with_grads()
out.grad.copy_from_numpy(np.ones((H, W), dtype=np.float32))
module.neg_start.bwds(spy.grid((H, W)), src, 2, W, _result=out)   # SIGSEGV
```

No Python traceback (hard native crash, exit 139). PyTorch path crashes at
`result.backward()` with the same trigger.

<details>
<summary>Full repro script (repro.py)</summary>

```python
# Minimal repro: backward dispatch of a [Differentiable] function segfaults
# when the function contains a loop with a NEGATIVE runtime start value,
# e.g. `for (int dx = -radius; dx <= radius; ++dx)`.
# Crashes on Metal, CUDA and Vulkan (slangpy 0.42.0); forward is correct.
#
#   python repro_bwds_segfault.py            # SIGSEGV at the bwds() call
#   python repro_bwds_segfault.py control    # 0-based loop, same math: works

import sys

import numpy as np
import slangpy as spy

SHADER = """
import slangpy;

// repro: negative runtime loop start -> bwds() SEGFAULTS
[Differentiable]
float neg_start(no_diff int2 pix, IDiffTensor<float, 2> src,
                no_diff int radius, no_diff int width)
{
    float acc = 0.0;
    [MaxIters(17)]
    for (int dx = -radius; dx <= radius; ++dx)
        acc += src[uint2(clamp(pix.x + dx, 0, width - 1), pix.y)];
    return acc;
}

// control: identical math, 0-based loop -> bwds() works, gradients correct
[Differentiable]
float zero_start(no_diff int2 pix, IDiffTensor<float, 2> src,
                 no_diff int radius, no_diff int width)
{
    float acc = 0.0;
    [MaxIters(17)]
    for (int i = 0; i < 2 * radius + 1; ++i)
    {
        int dx = i - radius;
        acc += src[uint2(clamp(pix.x + dx, 0, width - 1), pix.y)];
    }
    return acc;
}
"""


def main() -> None:
    # DeviceType.cuda and DeviceType.vulkan behave identically
    device = spy.create_device(type=spy.DeviceType.metal)
    module = spy.Module.load_from_source(device, "bwds_repro", SHADER)
    H = W = 8
    R = 2

    control = len(sys.argv) > 1 and sys.argv[1] == "control"
    fn = module.zero_start if control else module.neg_start

    src = spy.Tensor.from_numpy(device, np.random.rand(H, W).astype(np.float32)).with_grads()
    out = fn(spy.grid((H, W)), src, R, W)
    print("forward ok:", out.shape, "(values correct in both variants)")

    out = out.with_grads()
    out.grad.copy_from_numpy(np.ones((H, W), dtype=np.float32))
    print("calling bwds...")
    fn.bwds(spy.grid((H, W)), src, R, W, _result=out)   # <-- SIGSEGV in repro mode
    print("bwds ok, grad sum:", float(src.grad.to_numpy().sum()))


if __name__ == "__main__":
    main()
```

</details>

## Bisection matrix (all with IDiffTensor<float,2> + spy.grid)

| loop form                          | tensor index    | bwds() |
|------------------------------------|-----------------|--------|
| `for i in 0..3` (literal)          | fixed           | OK     |
| `for (i = 0; i < n; ++i)`          | fixed           | OK     |
| `for (i = 0; i < n; ++i)`          | loop-varying    | OK     |
| `for (i = 0; i <= n; ++i)`         | loop-varying    | OK     |
| `for (dx = -r; dx < r + 1; ++dx)`  | loop-varying    | **SIGSEGV** |
| `for (dx = -r; dx <= r; ++dx)`     | fixed           | **SIGSEGV** |
| `for (dx = -r; dx <= r; ++dx)`     | loop-varying    | **SIGSEGV** |

So: inclusive bounds are fine, varying indices are fine — the **negative
runtime start value** is the trigger. Per-element differentiable functions
(no loops) work correctly, including through torch autograd.

## Expected

Either correct gradients (as produced by the zero-based rewrite) or a
compile/runtime error — not a process crash.

## Workaround (for anyone else hitting this)

Rewrite differentiable loops to start at zero:

```slang
for (int i = 0; i < 2 * radius + 1; ++i) { int dx = i - radius; ... }
```

With that form, a full separable gaussian produces gradients matching
closed-form and finite-difference references to float32 precision on all
three backends, through both `.bwds()` and `torch.autograd`.
