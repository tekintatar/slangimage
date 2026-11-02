# [Bug] `Function.bwds()` hangs when runtime-dependent control flow encloses an `IDiffTensor` load

> FILED: https://github.com/shader-slang/slangpy/issues/1167 (2026-09-17)
>
> **Status 2026-09-27:** open; escalated upstream to
> [shader-slang/slang#13169](https://github.com/shader-slang/slang/issues/13169)
> (open). Still hangs on SlangPy 0.43.1 on CUDA and Vulkan; stack samples place
> the hang inside the Slang compiler (see `FINDINGS.md`). Current status:
> [`docs/UPSTREAM_ISSUES.md`](../../docs/UPSTREAM_ISSUES.md).

## Summary

On SlangPy **0.43.0 and 0.43.1**, `Function.bwds()` never returns when a
`[Differentiable]` function loads an `IDiffTensor` inside runtime-dependent
control flow. The smallest form I found is a zero-based loop with a runtime
upper bound:

```slang
[MaxIters(17)]
for (int i = 0; i < 2 * radius + 1; ++i)
    acc += src[...];
```

The forward call completes correctly. Backward leaves the process alive but
blocked indefinitely inside the native call; there is no exception or Python
traceback. A mathematically identical fixed-count control (`i < 5`, invoked
with `radius=2`) completes and returns the exact gradient.

This reproduces with SlangPy's direct `Tensor` and `.bwds()` APIs, without
PyTorch or `slangpy-torch`. It is a regression from SlangPy 0.42.0.

## Environment

- Ubuntu 24.04, Python 3.12.3, NumPy 2.5.3
- NVIDIA L40S, driver 580.126.16
- Reproduced on both CUDA and Vulkan
- SlangPy 0.43.1 from PyPI
  - bundled compiler: `libslang-compiler.so.0.2026.12`
- Also reproduced with SlangPy 0.43.0 / Slang 2026.12
- Control: SlangPy 0.42.0 / Slang 2026.5.2
- The reproducer does not import PyTorch or use `slangpy-torch`

## Reproduction

Save the script below as `repro.py`, then run each case under a timeout because
the affected call does not recover:

```bash
python -m pip install numpy slangpy==0.43.1

timeout -k 2s 12s python repro.py runtime cuda
echo $?  # 124: forward prints, then bwds() hangs

timeout -k 2s 12s python repro.py fixed cuda
echo $?  # 0: bwds() completes, gradient sum is 320
```

Replacing `cuda` with `vulkan` gives the same results.

Observed output for the runtime-count case:

```text
slangpy=0.43.1 backend=cuda variant=runtime
forward ok: shape=[8, 8]
calling bwds()
# no further output; timeout exits 124
```

Observed output for the fixed-count control:

```text
slangpy=0.43.1 backend=cuda variant=fixed
forward ok: shape=[8, 8]
calling bwds()
bwds ok: grad_sum=320.0
```

<details>
<summary>Full reproducer (repro.py)</summary>

```python
from __future__ import annotations

import sys

import numpy as np
import slangpy as spy


SHADER = r"""
import slangpy;

[Differentiable]
float runtime_count(no_diff int2 pix, IDiffTensor<float, 2> src,
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

[Differentiable]
float fixed_count(no_diff int2 pix, IDiffTensor<float, 2> src,
                  no_diff int radius, no_diff int width)
{
    float acc = 0.0;
    [MaxIters(17)]
    for (int i = 0; i < 5; ++i)
    {
        int dx = i - radius;
        acc += src[uint2(clamp(pix.x + dx, 0, width - 1), pix.y)];
    }
    return acc;
}
"""


def main() -> None:
    variant = sys.argv[1] if len(sys.argv) > 1 else "runtime"
    backend = sys.argv[2] if len(sys.argv) > 2 else "cuda"
    if variant not in {"runtime", "fixed"}:
        raise SystemExit("variant must be 'runtime' or 'fixed'")

    device_type = getattr(spy.DeviceType, backend)
    device = spy.create_device(type=device_type)
    module = spy.Module.load_from_source(device, "runtime_loop_autodiff_hang", SHADER)
    fn = module.runtime_count if variant == "runtime" else module.fixed_count

    height = width = 8
    radius = 2
    src = spy.Tensor.from_numpy(
        device, np.arange(height * width, dtype=np.float32).reshape(height, width)
    ).with_grads()
    out = fn(spy.grid((height, width)), src, radius, width).with_grads()
    print(f"slangpy={spy.__version__} backend={backend} variant={variant}", flush=True)
    print(f"forward ok: shape={out.shape}", flush=True)

    out.grad.copy_from_numpy(np.ones((height, width), dtype=np.float32))
    print("calling bwds()", flush=True)
    fn.bwds(spy.grid((height, width)), src, radius, width, _result=out)
    grad_sum = float(src.grad.to_numpy().sum())
    print(f"bwds ok: grad_sum={grad_sum}", flush=True)
    if grad_sum != 320.0:
        raise AssertionError(f"expected gradient sum 320.0, got {grad_sum}")


if __name__ == "__main__":
    main()
```

</details>

## Version and control matrix

All hanging cases were terminated by `timeout -k 2s 12s`.

| SlangPy | Slang compiler | Case | CUDA | Vulkan |
|---|---|---|---|---|
| 0.42.0 | 2026.5.2 | runtime-count loop | OK, gradient sum 320 | OK, gradient sum 320 |
| 0.43.0 | 2026.12 | runtime-count loop | **hang**, exit 124 | not tested |
| 0.43.1 | 2026.12 | fixed-count control | OK, gradient sum 320 | OK, gradient sum 320 |
| 0.43.1 | 2026.12 | runtime-count loop | **hang**, exit 124 | **hang**, exit 124 |
| 0.43.0 rebuilt against 2026.5.2 | 2026.5.2 | runtime-count loop | OK, gradient sum 320 | OK, gradient sum 320 |

The hybrid control was built from the official SlangPy `v0.43.0` tag (commit
`e1927ed90ef5e856bc4835ad5117d81840ce5c77`), selecting the older compiler
with:

```bash
CMAKE_ARGS="-DSGL_SLANG_VERSION=2026.5.2" python -m pip wheel --no-deps .
```

Changing only the compiler version makes the direct reproducer pass on both
backends. This points to the Slang 2026.12 compiler bundled by the 0.43 line,
rather than SlangPy's Python autograd hook, PyTorch integration or device API.

## Trigger boundary

Additional one-change controls produced this boundary on 0.43.1:

| Differentiable tensor load | Control flow around the load | `bwds()` |
|---|---|---|
| once, no loop | none | OK |
| loop-varying index | fixed literal count | OK |
| after loop | runtime-count scalar-only loop | OK |
| same element repeatedly | runtime-count loop | **hang** |
| loop-varying index | runtime-count loop | **hang** |
| inside runtime `if` | fixed maximum loop | **hang** |
| after runtime `break` condition | fixed maximum loop | **hang** |

Dynamic indexing alone is therefore not the trigger. The common property is
runtime-dependent control flow around a differentiable tensor load.

## Not a duplicate of #1051 / Slang #12070

#1051 concerns a loop whose **induction start** is a negative runtime
expression. Its working workaround is the literal-zero loop form used by this
reproducer. That form passes on SlangPy 0.42.0 but regresses on the 0.43 line.

This report instead uses a literal zero start and a runtime upper bound. It also
hangs with fixed maximum loops that put the load behind a runtime `if` or
`break`, so it is broader than negative-start loop lowering.

## Expected

`bwds()` should complete with the same gradients as the fixed-count control
(gradient sum 320 for this input), or fail with an actionable compiler/runtime
error. It should not block the process indefinitely.

## Workaround

Pinning SlangPy 0.42.0 avoids the regression. Replacing the runtime bound with a
literal count also works when the iteration count is genuinely static, but that
is not a general workaround for runtime-radius kernels.
