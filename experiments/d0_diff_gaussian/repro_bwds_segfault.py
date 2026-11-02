# Minimal repro: backward dispatch of a [Differentiable] function segfaults
# when the function contains a loop with a NEGATIVE runtime start value,
# e.g. `for (int dx = -radius; dx <= radius; ++dx)`.
# Crashes on Metal, CUDA and Vulkan (slangpy 0.42.0); forward is correct.
#
#   python repro_bwds_segfault.py                 # SIGSEGV at the bwds() call
#   python repro_bwds_segfault.py control         # 0-based loop, same math: works
#   python repro_bwds_segfault.py repro vulkan    # choose the backend (default:
#                                                 # metal on macOS, cuda elsewhere)

import platform
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
    # Metal, CUDA and Vulkan behave identically
    default = "metal" if platform.system() == "Darwin" else "cuda"
    backend = sys.argv[2] if len(sys.argv) > 2 else default
    device = spy.create_device(type=getattr(spy.DeviceType, backend))
    module = spy.Module.load_from_source(device, "bwds_repro", SHADER)
    H = W = 8
    R = 2

    control = len(sys.argv) > 1 and sys.argv[1] == "control"
    fn = module.zero_start if control else module.neg_start

    src = spy.Tensor.from_numpy(device, np.random.rand(H, W).astype(np.float32)).with_grads()
    out = fn(spy.grid((H, W)), src, R, W)
    print("forward ok:", out.shape, "(values correct in both variants)", flush=True)

    out = out.with_grads()
    out.grad.copy_from_numpy(np.ones((H, W), dtype=np.float32))
    print("calling bwds...", flush=True)
    fn.bwds(spy.grid((H, W)), src, R, W, _result=out)   # <-- SIGSEGV in repro mode
    print("bwds ok, grad sum:", float(src.grad.to_numpy().sum()), flush=True)


if __name__ == "__main__":
    main()
