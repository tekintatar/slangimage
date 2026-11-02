# tests/test_public_api.py
# The promised public surface: si.load / from_numpy, Image handle,
# functional si.ops.*, pipe chaining, prewarm, Image-aware effects.

from __future__ import annotations

import numpy as np
import pytest

import slangimage as si


# These deliberately test the no-device-plumbing API. Explicit-backend prewarm
# and validation checks live in test_prewarm.py, not in this AUTO group.
pytestmark = pytest.mark.auto_device


@pytest.fixture(scope="module")
def img():
    rng = np.random.default_rng(31)
    arr = rng.random((96, 96, 4)).astype(np.float16)
    arr[..., 3] = 1.0
    return si.from_numpy(arr)


def test_from_numpy_infers_precision_and_wraps():
    arr16 = np.zeros((8, 8, 4), dtype=np.float16)
    arr32 = np.zeros((8, 8, 4), dtype=np.float32)
    assert si.from_numpy(arr16).precision == "fp16"
    assert si.from_numpy(arr32).precision == "fp32"
    im = si.from_numpy(arr16)
    assert isinstance(im, si.Image)
    assert (im.width, im.height) == (8, 8)


def test_load_save_roundtrip(tmp_path):
    rng = np.random.default_rng(32)
    arr = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
    from PIL import Image as PILImage

    p = tmp_path / "in.png"
    PILImage.fromarray(arr, "RGB").save(p)

    im = si.load(p)  # fp16 default
    assert im.precision == "fp16"
    out_p = tmp_path / "out.png"
    im.save(out_p)

    back = np.asarray(PILImage.open(out_p).convert("RGB"))
    np.testing.assert_allclose(back, arr, atol=1)  # fp16 is lossless for u8


def test_functional_op_no_plumbing(img):
    out = si.ops.vibrance(img, amount=0.6)
    assert isinstance(out, si.Image)
    assert (out.width, out.height) == (img.width, img.height)
    assert out.precision == img.precision == "fp16"  # fp16 in, fp16 out


def test_pipe_chaining(img):
    out = (
        img.pipe(si.ops.surface_blur, radius=5, bsigma=0.1)
        .pipe(si.ops.vibrance, amount=0.5)
        .pipe(si.ops.sharpen_luma, radius=3.0, sharpness=0.8)
    )
    arr = out.to_numpy()
    assert np.isfinite(arr).all()
    assert arr[..., :3].std() > 1e-4


def test_two_input_and_generator():
    noise = si.ops.random_noise(64, 64, seed=2.0)
    assert isinstance(noise, si.Image) and noise.precision == "fp16"
    blended = si.ops.blend(noise, si.ops.invert(noise), mode="overlay", opacity=0.8)
    assert isinstance(blended, si.Image)


def test_resize_aspect(img):
    half = si.ops.resize(img, scale=0.5)
    assert (half.width, half.height) == (48, 48)
    w_only = si.ops.resize(img, width=32)
    assert (w_only.width, w_only.height) == (32, 32)
    with pytest.raises(ValueError):
        si.ops.resize(img)


def test_raw_texture_passthrough(img):
    out_tex = si.ops.invert(img.texture)  # raw in, raw out
    import slangpy as spy

    assert isinstance(out_tex, spy.Texture)


def test_effect_accepts_image(img):
    from slangimage.fx.soft_glow import SoftGlowEffect

    device = img.device
    eff = SoftGlowEffect(device, si.ModuleCache(device))
    out = eff.apply(img, glow_strength=0.6)
    assert isinstance(out, si.Image)
    out_tex = eff.apply(img.texture, glow_strength=0.6)
    assert not isinstance(out_tex, si.Image)


def test_data_readback_unclipped(img):
    arr = si.ops.normalize(img, mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)).to_numpy()
    assert np.isfinite(arr).all()
    # normalized data is not color; it must read back without clipping
    assert arr[..., :3].min() < 0.0 and arr[..., :3].max() > 1.0
