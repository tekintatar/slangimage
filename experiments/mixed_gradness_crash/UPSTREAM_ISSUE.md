# [Bug] CUDA illegal memory access (process abort) in backward when a `[Differentiable]` function has two tensor params and only one requires grad

> FILED: https://github.com/shader-slang/slangpy/issues/1056 (2026-07-12). Repro: repro_mixed_gradness.py (verified 2026-07-12, L40S).
>
> **Status 2026-09-27:** open; the fix PR
> [#1057](https://github.com/shader-slang/slangpy/pull/1057) is an unmerged
> draft. Still aborts on SlangPy 0.42.0 and 0.43.1. Current status:
> [`docs/UPSTREAM_ISSUES.md`](../../docs/UPSTREAM_ISSUES.md).

## Summary

Calling a `[Differentiable]` function with two `IDiffTensor` params
where only **one** of the torch inputs has `requires_grad=True` crashes
the process with `CUDA_ERROR_ILLEGAL_ADDRESS` during backward. When
**both** inputs require grad, backward works and the gradients are
exact. The kernel body is a single multiply — nothing exotic is needed.

The likely cause: the generated backward kernel scatters gradients for
*every* `IDiffTensor` param, but a grad buffer is only bound for inputs
that require grad — the write for the no-grad input goes through an
unbound/dangling address.

This makes the extremely common "frozen input, learnable parameter"
pattern (e.g. a fixed image and a learnable LUT/weight tensor) crash by
default.

Because the failure is asynchronous, without an explicit
`torch.cuda.synchronize()` it surfaces wherever the stream next syncs —
we first hit it as an `AcceleratorError` inside an Adam
`optimizer.step()`, far from the actual cause, which made it expensive
to trace.

## Environment

- slangpy **0.42.0**, slangpy-torch 0.7.0 (native extension)
- torch 2.13.0+cu130, CUDA device via `spy.create_torch_device`
- Ubuntu 24.04, NVIDIA L40S, driver 580.126.16
- Python 3.12

## Reproduction

Save the script below as `repro.py` (one case per process — the
illegal access poisons the CUDA context):

- `python repro.py both` — both inputs require grad: works, exact grads.
- `python repro.py first` — only the first input requires grad: **abort**.
- `python repro.py second` — only the second input requires grad: **abort**.

Observed output (crash cases — `first` and `second` are identical
modulo which input requires grad):

```text
case second: a.requires_grad=False, b.requires_grad=True
Traceback (most recent call last):
  File "repro.py", line 61, in <module>
    main()
  File "repro.py", line 49, in main
    torch.cuda.synchronize()
  ...
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
At /project/external/slang-rhi/src/cuda/cuda-command.cpp:1010
cuModuleUnload(m_module) failed: an illegal memory access was encountered (CUDA_ERROR_ILLEGAL_ADDRESS)
Assertion failed: CUDA call failed
At /project/external/slang-rhi/src/cuda/cuda-pipeline.cpp:23
Aborted (core dumped)   [exit code 134]
```

Observed output (control):

```text
case both: a.requires_grad=True, b.requires_grad=True
backward OK, gradients exact (a.grad set: True, b.grad set: True)
```

<details>
<summary>Full repro script (repro.py)</summary>

```python
import sys

import torch
import slangpy as spy

SHADER = """
import slangpy;

[Differentiable]
float mul2(no_diff int2 pix, IDiffTensor<float, 2> a, IDiffTensor<float, 2> b)
{
    return a[uint2(pix.x, pix.y)] * b[uint2(pix.x, pix.y)];
}
"""


def main() -> None:
    case = sys.argv[1] if len(sys.argv) > 1 else "second"
    assert case in ("both", "first", "second")

    device = spy.create_torch_device(type=spy.DeviceType.cuda)
    module = spy.Module.load_from_source(device, "mixed_grad_repro", SHADER)

    H, W = 32, 32
    torch.manual_seed(0)
    a = torch.rand(H, W, device="cuda", requires_grad=case in ("both", "first"))
    b = torch.rand(H, W, device="cuda", requires_grad=case in ("both", "second"))
    print(f"case {case}: a.requires_grad={a.requires_grad}, "
          f"b.requires_grad={b.requires_grad}")

    out = module.mul2(spy.grid((H, W)), a, b)
    out.sum().backward()
    torch.cuda.synchronize()

    # d(sum(a*b))/da = b and vice versa
    if a.grad is not None:
        torch.testing.assert_close(a.grad, b.detach())
    if b.grad is not None:
        torch.testing.assert_close(b.grad, a.detach())
    print(f"backward OK, gradients exact "
          f"(a.grad set: {a.grad is not None}, b.grad set: {b.grad is not None})")


if __name__ == "__main__":
    main()
```

</details>

## Case matrix

| `a.requires_grad` | `b.requires_grad` | result |
|---|---|---|
| yes | yes | OK — both gradients exact |
| yes | no | **illegal memory access, process abort** |
| no | yes | **illegal memory access, process abort** |

## Not a duplicate of #1052 / not fixed by #1054

Related to #1052 (input grad-ness not accounted for), but on a
different layer, and **not** fixed by PR #1054:

- #1054 adds the per-tensor grad bit to the call-data **cache
  signature** (`[Dn,Sm,Gk]`) so grad and no-grad calls route to
  distinct `CallData`. It changes routing only — per its own
  description, the compiled kernel and dispatch-time data binding are
  unchanged.
- This repro crashes on the **first-ever call in a fresh process**:
  there is no earlier call, so no cache entry to be mis-keyed. The
  mixed-grad call builds its own `CallData` (exactly the post-#1054
  state) and still aborts — the failure is in what the backward
  dispatch **binds** for a no-grad `IDiffTensor` input, not in which
  cached entry is selected.

## Expected

Backward computes gradients for the inputs that require them and skips
(or discards) the others — mixed grad-ness is the normal case when
training a parameter against frozen inputs.

## Workaround (for anyone else hitting this)

When at least one tensor argument requires grad, replace every no-grad
tensor argument with a throwaway alias that has a grad buffer:

```python
t_aliased = t.detach().requires_grad_(True)
```

The generated backward then has a real buffer to scatter into, and
torch simply discards the unused leaf gradient afterwards. (Costs one
gradient buffer per frozen input.)
