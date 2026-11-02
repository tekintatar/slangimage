# Minimal repro: a [Differentiable] function taking TWO tensor params
# crashes with a CUDA illegal memory access in backward when only ONE
# of the inputs requires grad. Both-inputs-require-grad works and gives
# exact gradients. The kernel body is a single multiply.
#
#   python repro_mixed_gradness.py both    # works, exact grads
#   python repro_mixed_gradness.py first   # only a requires grad -> ABORT
#   python repro_mixed_gradness.py second  # only b requires grad -> ABORT
#
# One case per process: the illegal access poisons the CUDA context.
# Without an explicit synchronize the failure surfaces asynchronously in
# whatever CUDA call happens next (we first hit it inside an Adam
# optimizer step, far from the cause).
#
# Requires torch + CUDA. slangpy 0.42.0.

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
