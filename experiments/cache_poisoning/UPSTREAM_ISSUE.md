# [Bug] PyTorch autograd hook silently dropped after a no-grad call to the same function (call-data cache ignores requires_grad)

> FILED: https://github.com/shader-slang/slangpy/issues/1052 (2026-07-12)
>
> **Status 2026-09-27:** open; the fix PR
> [#1054](https://github.com/shader-slang/slangpy/pull/1054) is an unmerged
> draft. Still reproduces on SlangPy 0.42.0 and 0.43.1. Current status:
> [`docs/UPSTREAM_ISSUES.md`](../../docs/UPSTREAM_ISSUES.md).

## Summary

When a slangpy function is first called with PyTorch tensors that do
**not** require grad, all **subsequent** calls to that function with
`requires_grad=True` tensors return outputs with `requires_grad=False`
and no `grad_fn` — the autograd hook is silently dropped. No error or
warning is raised at the call site; gradients simply stop existing.
`loss.backward()` then fails with
`element 0 of tensors does not require grad`, or worse, a partially
connected graph trains on wrong gradients.

The cause appears to be the per-function call-data cache: the generated
dispatch is cached by call signature **without accounting for the
inputs' grad-ness**, so the no-grad variant is reused for grad-requiring
calls.

This is easy to hit in practice: any warm-up/inference forward pass
before training poisons every function it touched for the rest of the
process.

## Environment

- slangpy **0.42.0**, slangpy-torch 0.7.0 (native extension; also
  reproduces with `SLANGPY_ALLOW_TORCH_FALLBACK=1`)
- torch 2.13.0+cu130, CUDA device via `spy.create_torch_device`
- Ubuntu 24.04, NVIDIA L40S, driver 580.126.16
- Python 3.12

## Reproduction

Save the script below as `repro.py`.

- `python repro.py` — no-grad call first, then a grad-requiring call:
  the second call returns `requires_grad=False`, backward fails.
- `python repro.py control` — grad-requiring call first: everything
  works (`grad_fn` attached, gradients exact).

Observed output (bug order):

```text
step 1: called square() with a no-grad tensor
step 2: grad-requiring call -> y.requires_grad=False, grad_fn=None
backward FAILED: element 0 of tensors does not require grad and does not have a grad_fn
```

<details>
<summary>Full repro script (repro.py)</summary>

```python
# Minimal repro: slangpy's call-data cache ignores whether torch inputs
# require grad. One no-grad call to a function permanently drops the
# autograd hook from all later grad-requiring calls to that function —
# silently (no error; outputs just have requires_grad=False).
#
#   python repro_cache_poisoning.py          # bug order: no-grad call first
#   python repro_cache_poisoning.py control  # grad call first: works
#
# Requires torch + CUDA (torch interop path). slangpy 0.42.0.

import sys

import torch
import slangpy as spy

SHADER = """
[Differentiable]
float square(float x)
{
    return x * x;
}
"""


def main() -> None:
    device = spy.create_torch_device(type=spy.DeviceType.cuda)
    module = spy.Module.load_from_source(device, "cache_repro", SHADER)

    control = len(sys.argv) > 1 and sys.argv[1] == "control"

    if not control:
        # the poisoning call: same function, input does NOT require grad
        plain = torch.tensor([5.0, 6.0], device="cuda")
        _ = module.square(x=plain)
        print("step 1: called square() with a no-grad tensor")

    x = torch.tensor([1.0, 2.0, 3.0], device="cuda", requires_grad=True)
    y = module.square(x=x)
    print(f"step 2: grad-requiring call -> y.requires_grad={y.requires_grad}, "
          f"grad_fn={y.grad_fn}")

    try:
        y.sum().backward()
        print(f"backward OK, x.grad = {x.grad.tolist()}   (expected [2, 4, 6])")
    except RuntimeError as e:
        print(f"backward FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

</details>

## Asymmetry (useful for the fix)

The contamination is one-directional:

| first call | later call | result |
|---|---|---|
| no-grad | grad-requiring | **broken** — output has no grad_fn |
| grad-requiring | grad-requiring | OK, gradients exact |
| grad-requiring | no-grad | OK — output correctly has requires_grad=False, values correct |

So call data built for grad-requiring inputs serves both cases
correctly; only the no-grad-built cache entry is unsafe for
grad-requiring reuse. Keying the cache on input grad-ness (or always
building the differentiable variant when the function is
`[Differentiable]`) would both fix it.

## Expected

Either correct autograd hookup regardless of call order, or an error —
not silent gradient loss.

## Workaround (for anyone else hitting this)

Load the module twice under different names and route calls by
`any(t.requires_grad for t in tensor_args)` — two module instances have
separate call-data caches, so the grad path never sees the poisoned
entry.
