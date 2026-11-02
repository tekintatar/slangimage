"""Reproduce the SlangPy 0.43 dynamic-control-flow backward hang.

The default runtime-count case completes on SlangPy 0.42.0 and hangs in
``Function.bwds()`` on 0.43.0/0.43.1.  The fixed-count control completes on
all three versions.

Run under an external timeout because an affected process does not recover::

    timeout 15s python repro_runtime_loop_autodiff_hang.py runtime cuda
    timeout 15s python repro_runtime_loop_autodiff_hang.py fixed cuda
"""

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
