# examples/06_differentiable_torch.py
# slangimage.diff — fused Slang kernels inside torch.autograd (CUDA only):
# chain differentiable ops, backprop through them, and learn a color grade
# as a LUT by gradient descent (the headline feature).
#
# Requires torch with CUDA; on other machines this script explains and exits.

import numpy as np

try:
    import torch
except ImportError:
    raise SystemExit(
        "slangimage.diff needs PyTorch, SlangPy's torch bridge and an NVIDIA CUDA "
        "GPU (Linux). Install them with the three steps in the README section "
        "'PyTorch extra'; a plain pip install \"slangimage[torch]\" cannot build "
        "the bridge.\n"
        "The torch extra does not install on macOS; use the non-differentiable "
        "si.ops path there."
    )

if not torch.cuda.is_available():
    raise SystemExit(
        "slangimage.diff needs a CUDA device (torch interop is CUDA-only).\n"
        "On Metal machines, use the non-differentiable si.ops path instead."
    )

import slangimage.diff as sid
from slangimage.ops import identity_lut_np

# --- 1. gradients flow through fused Slang kernels -------------------------
img = torch.rand(256, 256, 3, device="cuda", requires_grad=True)
out = sid.exposure(img, ev=0.5)
out = sid.bilateral(out, radius=5, sigma_spatial=2.5, sigma_range=0.15)
out = sid.sepia(out, intensity=0.8)
out.mean().backward()
print(f"chained bilateral pipeline: grad norm = {img.grad.norm().item():.4f}")

# --- 2. learn a color grade as a LUT ---------------------------------------
# Given (input, edited) pairs from ANY editor, recover the edit as a LUT.
rng = np.random.default_rng(0)
source = torch.tensor(rng.random((128, 128, 3), dtype=np.float64).astype(np.float32),
                      device="cuda")
target = sid.hue_rotate(sid.exposure(source, 0.3), 0.4).detach()  # the "unknown edit"

lut = torch.nn.Parameter(
    torch.tensor(identity_lut_np()[..., :3].copy(), device="cuda")
)
optim = torch.optim.Adam([lut], lr=5e-2)

for step in range(80):
    loss = (sid.lut(source, lut) - target).pow(2).mean()
    optim.zero_grad()
    loss.backward()
    optim.step()
    if step % 20 == 0:
        print(f"step {step:3d}  loss {loss.item():.6f}")

final = (sid.lut(source, lut) - target).pow(2).mean().item()
print(f"learned the edit as a LUT: final loss {final:.2e}")
print("the trained `lut` tensor is now a shareable color-grade preset")
