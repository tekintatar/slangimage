# experiments/d0_diff_gaussian/manual_bridge.py
# D0 workaround proof: differentiable gaussian WITHOUT slangpy's .bwds()
# plumbing (which segfaults when a differentiable loop starts at a runtime
# value, slangpy#1051, all backends, 0.42).
#
# Architecture — the D1 pattern:
#   torch.autograd.Function
#     forward  -> normal slangpy dispatch of the forward kernel
#     backward -> normal slangpy dispatch of a hand-written backward kernel
# Both go through the same production-hardened call path as si.ops.
#
# Run on a CUDA host prepared by scripts/cuda_box/setup.sh (which creates
# ~/cuda-shim and ~/venvs/slangimage):
#   CUDA_PATH=~/cuda-shim LD_LIBRARY_PATH=~/cuda-shim/lib64 \
#   ~/venvs/slangimage/bin/python experiments/d0_diff_gaussian/manual_bridge.py

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

import slangpy as spy

HERE = Path(__file__).resolve().parent
R, SIGMA = 9, 3.0

device = spy.create_torch_device(type=spy.DeviceType.cuda)
module = spy.Module.load_from_file(device, str(HERE / "diff_gaussian.slang"))


class GaussianH(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src: torch.Tensor, radius: int, sigma: float) -> torch.Tensor:
        H, W = src.shape
        ctx.meta = (radius, sigma, H, W)
        return module.gaussian_h(spy.grid((H, W)), src.detach(), radius, sigma, W)

    @staticmethod
    def backward(ctx, out_grad: torch.Tensor):
        radius, sigma, H, W = ctx.meta
        grad_src = module.gaussian_h_bwd(
            spy.grid((H, W)), out_grad.contiguous().detach(), radius, sigma, W
        )
        return grad_src, None, None


def gaussian_h(src, radius=R, sigma=SIGMA):
    return GaussianH.apply(src, radius, sigma)


def np_gaussian_h(img, radius, sigma):
    dx = np.arange(-radius, radius + 1)
    w = np.exp(-(dx * dx) / (2.0 * sigma * sigma))
    w /= w.sum()
    pad = np.pad(img, ((0, 0), (radius, radius)), mode="edge")
    out = np.zeros_like(img)
    for i, wi in enumerate(w):
        out += wi * pad[:, i : i + img.shape[1]]
    return out


def main():
    print(f"device: {device.info.adapter_name}")
    H = W = 96
    rng = np.random.default_rng(5)
    src_np = rng.random((H, W)).astype(np.float32)
    wgt_np = rng.random((H, W)).astype(np.float32)

    src = torch.tensor(src_np, device="cuda", requires_grad=True)
    out = gaussian_h(src)

    fwd_err = np.abs(out.detach().cpu().numpy() - np_gaussian_h(src_np, R, SIGMA)).max()
    print(f"forward vs numpy:            max err {fwd_err:.2e}  {'PASS' if fwd_err < 1e-5 else 'FAIL'}")

    (out * torch.tensor(wgt_np, device="cuda")).sum().backward()
    grad = src.grad.detach().cpu().numpy()

    # closed form in the interior (transpose of symmetric kernel = same blur)
    ref = np_gaussian_h(wgt_np, R, SIGMA)
    m = R + 1
    int_err = np.abs(grad[:, m:-m] - ref[:, m:-m]).max()
    print(f"backward interior vs closed: max err {int_err:.2e}  {'PASS' if int_err < 1e-4 else 'FAIL'}")

    # finite differences everywhere incl. edges (where clamp-transpose matters)
    def loss_np(s):
        return float((np_gaussian_h(s, R, SIGMA) * wgt_np).sum())

    eps, errs = 1e-2, []
    for (i, j) in [(0, 0), (5, 2), (50, 50), (95, 95), (10, 0), (0, 60), (95, 1)]:
        sp = src_np.copy(); sp[i, j] += eps
        sm = src_np.copy(); sm[i, j] -= eps
        fd = (loss_np(sp) - loss_np(sm)) / (2 * eps)
        errs.append(abs(fd - grad[i, j]))
    fd_max = max(errs)
    print(f"backward vs finite diff:     max err {fd_max:.2e}  {'PASS' if fd_max < 1e-2 else 'FAIL'}")

    # torch.autograd.gradcheck (float32: loosened tolerances)
    small = torch.rand(24, 24, device="cuda", requires_grad=True)
    ok = torch.autograd.gradcheck(
        lambda t: gaussianH_apply(t), (small,), eps=1e-2, atol=1e-2, rtol=1e-2,
        nondet_tol=1e-5,
    ) if (gaussianH_apply := lambda t: GaussianH.apply(t, 4, 2.0)) else False
    print(f"torch.autograd.gradcheck:    {'PASS' if ok else 'FAIL'}")

    # timing fwd+bwd
    N = 1024
    a = torch.rand(N, N, device="cuda", requires_grad=True)

    def fwd_bwd():
        o = gaussian_h(a)
        o.sum().backward()
        a.grad = None

    fwd_bwd(); torch.cuda.synchronize()
    ts = []
    for _ in range(10):
        t0 = time.perf_counter(); fwd_bwd(); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    print(f"fwd+bwd 1024^2 (1ch, r=9):   median {np.median(ts):.2f} ms")


if __name__ == "__main__":
    main()
