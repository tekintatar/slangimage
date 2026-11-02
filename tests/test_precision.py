# tests/test_precision.py
# fp16 pipeline correctness — the slangpy #648 regression test.
# Every RW output is annotated [format(SLANGIMAGE_FMT)]; CUDA writes raw bits
# without it. The four ops below run at fp16 and fp32 on the selected ctx
# backend. This compares storage precisions, not outputs across backends.

from __future__ import annotations

import numpy as np
import pytest

from slangimage.runtime.textures import np_to_texture, texture_to_np

from slangimage.kernels.gaussian import GaussianBlurOp
from slangimage.kernels.surface_blur import SurfaceBlurOp
from slangimage.kernels.vibrance import VibranceOp
from slangimage.kernels.unsharp_mask import UnsharpMaskOp

SIZE = 256

OPS = [
    ("gaussian", GaussianBlurOp, {"radius": 9, "sigma": 3.0}),
    ("surface_blur", SurfaceBlurOp, {"radius": 9, "bsigma": 0.1}),
    ("vibrance", VibranceOp, {"amount": 0.6}),
    ("unsharp_mask", UnsharpMaskOp, {"amount": 1.5, "radius": 8, "sigma": 3.0}),
]


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


@pytest.fixture(scope="session")
def src_np():
    rng = np.random.default_rng(7)
    y, x = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32) / SIZE
    img = np.stack(
        [
            0.5 + 0.5 * np.sin(8.0 * x) * np.cos(5.0 * y),
            np.clip(1.2 * y - 0.1, 0, 1),
            x,
            np.ones_like(x),
        ],
        axis=-1,
    )
    img[..., :3] += rng.normal(0.0, 0.02, size=(SIZE, SIZE, 3)).astype(np.float32)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


@pytest.mark.parametrize("name,op_cls,params", OPS, ids=[o[0] for o in OPS])
def test_fp16_matches_fp32(ctx, src_np, name, op_cls, params):
    device, modules = ctx
    op = op_cls(device, modules)

    t32 = np_to_texture(device, src_np, usage="read", precision="fp32")
    out32 = texture_to_np(op.apply(t32, **params))

    t16 = np_to_texture(device, src_np.astype(np.float16), usage="read", precision="fp16")
    out16 = texture_to_np(op.apply(t16, **params))

    assert np.isfinite(out16).all(), f"{name} fp16 produced non-finite values (#648 symptom)"
    # Readback itself is unclipped. This loose window catches #648 corruption;
    # it is not an operator contract (texture unsharp_mask clips its RGB).
    assert out16.min() >= -0.5 and out16.max() <= 1.5, (
        f"{name} fp16 out of range: [{out16.min()}, {out16.max()}] (#648 symptom)"
    )

    p = psnr(out16[..., :3], out32[..., :3])
    assert p > 40.0, f"{name}: fp16 vs fp32 PSNR {p:.1f} dB below threshold"
