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
