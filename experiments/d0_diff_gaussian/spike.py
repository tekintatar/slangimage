# experiments/d0_diff_gaussian/spike.py
# D0 spike: prove Slang autodiff -> slangpy -> torch.autograd end to end.
#
# stage 1: per-element polynomial, analytic gradient check (docs pattern)
# stage 2: neighborhood gaussian_h — forward vs numpy; backward vs
#          closed-form transpose (interior) and finite differences (edges)
# stage 3: fwd+bwd timing vs kornia gaussian_blur2d
#
# Run on a CUDA host prepared by scripts/cuda_box/setup.sh (which creates
# ~/cuda-shim and ~/venvs/slangimage):
#   CUDA_PATH=~/cuda-shim LD_LIBRARY_PATH=~/cuda-shim/lib64 \
#   ~/venvs/slangimage/bin/python experiments/d0_diff_gaussian/spike.py

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

import slangpy as spy

HERE = Path(__file__).resolve().parent
R, SIGMA = 9, 3.0


def np_gaussian_h(img: np.ndarray, radius: int, sigma: float) -> np.ndarray:
    dx = np.arange(-radius, radius + 1)
    w = np.exp(-(dx * dx) / (2.0 * sigma * sigma))
    w /= w.sum()
    pad = np.pad(img, ((0, 0), (radius, radius)), mode="edge")
    out = np.zeros_like(img)
    for i, wi in enumerate(w):
        out += wi * pad[:, i : i + img.shape[1]]
    return out


def main() -> None:
    device = spy.create_torch_device(type=spy.DeviceType.cuda)
    module = spy.Module.load_from_file(device, str(HERE / "diff_gaussian.slang"))
    print(f"device: {device.info.adapter_name}")

    # ---------------------------------------------------------- stage 1
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda", requires_grad=True)
    y = module.polynomial(a=2.0, b=8.0, c=-1.0, x=x)
    y.backward(torch.ones_like(y))
    expected = 2 * 2.0 * x.detach() + 8.0  # d/dx (2x^2 + 8x - 1)
    err = (x.grad - expected).abs().max().item()
    print(f"stage 1 polynomial grad err: {err:.2e}  {'PASS' if err < 1e-5 else 'FAIL'}")

    # ---------------------------------------------------------- stage 2
    H = W = 96
    rng = np.random.default_rng(5)
    src_np = rng.random((H, W)).astype(np.float32)

    src = torch.tensor(src_np, device="cuda", requires_grad=True)
    out = module.gaussian_h(spy.grid((H, W)), src, R, SIGMA, W)

    # forward correctness
    fwd_err = np.abs(out.detach().cpu().numpy() - np_gaussian_h(src_np, R, SIGMA)).max()
    print(f"stage 2 forward vs numpy: max err {fwd_err:.2e}  {'PASS' if fwd_err < 1e-5 else 'FAIL'}")

    # backward: loss = sum(out * Wgt); dL/dsrc = blur^T(Wgt)
    wgt_np = rng.random((H, W)).astype(np.float32)
    wgt = torch.tensor(wgt_np, device="cuda")
    (out * wgt).sum().backward()
    grad = src.grad.detach().cpu().numpy()

    # interior closed form: transpose of symmetric blur == same blur
    ref = np_gaussian_h(wgt_np, R, SIGMA)
    m = R + 1  # margin where clamp-padding transpose differs
    int_err = np.abs(grad[:, m:-m] - ref[:, m:-m]).max()
    print(f"stage 2 backward interior vs closed form: max err {int_err:.2e}  "
          f"{'PASS' if int_err < 1e-4 else 'FAIL'}")

    # finite differences at random entries incl. edges
    def loss_np(s):
        return float((np_gaussian_h(s, R, SIGMA) * wgt_np).sum())

    eps, fd_errs = 1e-3, []
    for (i, j) in [(0, 0), (5, 2), (50, 50), (95, 95), (10, 0), (0, 60)]:
        sp = src_np.copy(); sp[i, j] += eps
        sm = src_np.copy(); sm[i, j] -= eps
        fd = (loss_np(sp) - loss_np(sm)) / (2 * eps)
        fd_errs.append(abs(fd - grad[i, j]))
    fd_max = max(fd_errs)
    print(f"stage 2 backward vs finite differences (incl. edges): max err {fd_max:.2e}  "
          f"{'PASS' if fd_max < 1e-2 else 'FAIL'}")

    # ---------------------------------------------------------- stage 3
    try:
        import kornia.filters as KF
    except ImportError:
        print("stage 3 skipped (kornia not installed)")
        return

    N = 1024
    a = torch.rand(N, N, device="cuda", requires_grad=True)
    k = torch.rand(1, 3, N, N, device="cuda", requires_grad=True)

    def slang_fwd_bwd():
        o = module.gaussian_h(spy.grid((N, N)), a, R, SIGMA, N)
        o.sum().backward()
        a.grad = None

    def kornia_fwd_bwd():
        o = KF.gaussian_blur2d(k, (2 * R + 1, 2 * R + 1), (SIGMA, SIGMA))
        o.sum().backward()
        k.grad = None

    for name, fn in (("slangimage(1ch 1D)", slang_fwd_bwd), ("kornia(3ch 2D)", kornia_fwd_bwd)):
        fn(); torch.cuda.synchronize()
        ts = []
        for _ in range(10):
            t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
        print(f"stage 3 {name:20s} fwd+bwd median {np.median(ts):7.2f} ms  (not like-for-like; plumbing check)")


if __name__ == "__main__":
    main()
