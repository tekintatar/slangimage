# tests/test_smoke.py
# Plumbing smoke test: every effect runs end-to-end on the default device
# and produces sane output (correct shape, finite, non-constant).
# Uses a synthetic image so the repo carries no binary test assets.

from __future__ import annotations

import numpy as np
import pytest

from slangimage.runtime.textures import np_to_texture, texture_to_np

from slangimage.fx.gaussian_blur import GaussianBlurEffect
from slangimage.fx.soft_glow import SoftGlowEffect
from slangimage.fx.kuwahara_effect import KuwaharaPaintEffect

SIZE = 256

EFFECTS = [
    (GaussianBlurEffect, {}),
    (SoftGlowEffect, {}),
    (KuwaharaPaintEffect, {}),
]


@pytest.fixture(scope="session")
def src_np():
    # Deterministic synthetic photo-ish image: gradients + blobs + noise.
    rng = np.random.default_rng(42)
    y, x = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32) / SIZE
    r = 0.5 + 0.5 * np.sin(6.0 * x) * np.cos(3.0 * y)
    g = y
    b = x
    img = np.stack([r, g, b, np.ones_like(x)], axis=-1)
    img[..., :3] += rng.normal(0.0, 0.03, size=(SIZE, SIZE, 3)).astype(np.float32)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


@pytest.mark.parametrize("effect_cls,params", EFFECTS, ids=lambda e: getattr(e, "__name__", ""))
def test_effect_smoke(ctx, src_np, effect_cls, params):
    device, modules = ctx
    src_tex = np_to_texture(device, src_np, usage="read", precision="fp32")

    effect = effect_cls(device, modules)
    out_tex = effect.apply(src_tex, **params)

    out = texture_to_np(out_tex)
    assert out.shape[2] == 4
    assert out.shape[0] > 0 and out.shape[1] > 0
    assert np.isfinite(out).all(), f"{effect_cls.__name__} produced non-finite values"
    assert out[..., :3].std() > 1e-4, f"{effect_cls.__name__} produced a constant image"
