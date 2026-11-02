# tests/test_nn.py
# slangimage.nn Module wrappers: functional parity, nn.Sequential
# composition, the learnable-LUT training loop, state_dict roundtrip,
# and mixed-precision (fp16 inputs / autocast) behavior of the diff ops.
# Requires torch + CUDA (skipped elsewhere), like tests/test_diff.py.

from __future__ import annotations

import os

import numpy as np
import pytest

if os.environ.get("SLANGIMAGE_BACKEND", "auto") not in {"auto", "cuda"}:
    pytest.skip("CUDA autograd tests excluded by SLANGIMAGE_BACKEND", allow_module_level=True)

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("slangimage.nn requires CUDA", allow_module_level=True)

import slangimage.diff as sid  # noqa: E402
import slangimage.nn as snn  # noqa: E402
from _fp32_gradcheck import assert_fp32_finite_differences  # noqa: E402


@pytest.fixture(scope="module")
def img():
    rng = np.random.default_rng(51)
    return torch.tensor(rng.random((32, 28, 3)).astype(np.float32), device="cuda")


# ------------------------------------------------------ functional parity
def test_modules_match_functionals(img):
    cases = [
        (snn.GaussianBlur(radius=4, sigma=1.5),
         lambda t: sid.gaussian_blur(t, radius=4, sigma=1.5)),
        (snn.Bilateral(radius=3, sigma_spatial=2.0, sigma_range=0.2),
         lambda t: sid.bilateral(t, radius=3, sigma_spatial=2.0, sigma_range=0.2)),
        (snn.UnsharpMask(radius=2, sigma=1.2, amount=0.8),
         lambda t: sid.unsharp_mask(t, radius=2, sigma=1.2, amount=0.8)),
        (snn.Resize(scale=0.5, mode="bicubic"),
         lambda t: sid.resize(t, scale=0.5, mode="bicubic")),
        (snn.Exposure(0.7), lambda t: sid.exposure(t, 0.7)),
        (snn.Sepia(0.6), lambda t: sid.sepia(t, 0.6)),
        (snn.HueRotate(0.9), lambda t: sid.hue_rotate(t, 0.9)),
    ]
    for module, fn in cases:
        torch.testing.assert_close(module(img), fn(img), atol=0, rtol=0)

    blend = snn.Blend(mode="overlay", opacity=0.7)
    torch.testing.assert_close(
        blend(img, img.flip(0)),
        sid.blend(img, img.flip(0), mode="overlay", opacity=0.7),
        atol=0, rtol=0,
    )


# --------------------------------------------------------- composition
def test_sequential_pipeline_backward(img):
    pipe = torch.nn.Sequential(
        snn.Exposure(0.3),
        snn.Bilateral(radius=3, sigma_range=0.3),
        snn.GaussianBlur(radius=3, sigma=1.2),
        snn.Sepia(0.5),
    )
    src = img.clone().requires_grad_(True)
    out = pipe(src)
    out.mean().backward()
    assert src.grad is not None and torch.isfinite(src.grad).all()
    assert float(src.grad.abs().sum()) > 0
    # scalar-hyperparameter modules expose no trainable state
    assert list(pipe.parameters()) == []


def test_module_fp32_finite_differences():
    small = torch.rand(8, 7, 3, device="cuda", requires_grad=True)
    mod = snn.Bilateral(radius=2, sigma_spatial=1.5, sigma_range=0.5)
    assert_fp32_finite_differences(mod, (small,))


# -------------------------------------------------------- learnable LUT
def test_lut_module_learns(img):
    target = sid.hue_rotate(sid.exposure(img, 0.3), 0.5).detach()
    lut_mod = snn.Lut().cuda()
    assert len(list(lut_mod.parameters())) == 1

    optim = torch.optim.Adam(lut_mod.parameters(), lr=5e-2)
    loss0 = None
    for _ in range(40):
        loss = (lut_mod(img) - target).pow(2).mean()
        if loss0 is None:
            loss0 = float(loss.detach())
        optim.zero_grad()
        loss.backward()
        optim.step()
    final = float((lut_mod(img) - target).pow(2).mean().detach())
    assert final < loss0 * 0.3, f"LUT module failed to learn: {loss0} -> {final}"


def test_lut_state_dict_roundtrip(img):
    a = snn.Lut().cuda()
    with torch.no_grad():
        a.lut.add_(torch.randn_like(a.lut) * 0.01)
    b = snn.Lut().cuda()
    b.load_state_dict(a.state_dict())
    torch.testing.assert_close(a(img), b(img), atol=0, rtol=0)

    fixed = snn.Lut(a.lut.detach().cpu(), learnable=False).cuda()
    assert list(fixed.parameters()) == []
    torch.testing.assert_close(fixed(img), a(img), atol=0, rtol=0)


@pytest.mark.parametrize("shape", [(8, 8, 3), (10, 10, 3), (16, 24, 3)])
def test_lut_module_rejects_invalid_layout(shape):
    with pytest.raises(ValueError, match="square, divisible by 8"):
        snn.Lut(torch.zeros(shape))


# ----------------------------------------------------- mixed precision
def test_fp16_inputs_upcast_and_grad(img):
    half = img.half().requires_grad_(True)
    for fn in (
        lambda t: sid.gaussian_blur(t, radius=3, sigma=1.5),
        lambda t: sid.bilateral(t, radius=2, sigma_range=0.3),
        lambda t: sid.sepia(t, 0.5),
        lambda t: sid.resize(t, scale=0.5),
        lambda t: sid.lut(t, torch.tensor(
            __import__("slangimage").ops.identity_lut_np()[..., :3].copy(),
            device="cuda")),
    ):
        half.grad = None
        out = fn(half)
        assert out.dtype == torch.float32  # kernels compute fp32
        out.float().mean().backward()
        assert half.grad is not None and half.grad.dtype == torch.float16
        assert torch.isfinite(half.grad).all()


def test_autocast_context(img):
    src = img.clone().requires_grad_(True)
    pipe = torch.nn.Sequential(snn.Bilateral(radius=2), snn.Sepia(0.5))
    with torch.autocast("cuda"):
        out = pipe(src)
        loss = out.mean()
    loss.backward()
    assert src.grad is not None and torch.isfinite(src.grad).all()

    # fp16 result of an autocast matmul feeding our ops
    with torch.autocast("cuda"):
        mixed = img @ torch.eye(3, device="cuda")
        out = sid.resize(mixed, scale=0.5)
    assert torch.isfinite(out).all()
