# tests/test_color_aug.py
# Property tests for the augmentation color-op parity set: solarize,
# posterize, grayscale, normalize (pointwise) + equalize, autocontrast
# (histogram-based). Numpy references replicate each kernel's exact math.

from __future__ import annotations

import numpy as np
import pytest

from slangimage.runtime.textures import np_to_texture, texture_to_np
from slangimage.kernels.color_aug import ColorAugOps
from slangimage.kernels.hist_color import HistColorOps

ATOL = 2e-3


@pytest.fixture(scope="module")
def img_and_tex(ctx):
    device, _ = ctx
    rng = np.random.default_rng(31)
    img = rng.random((64, 48, 4)).astype(np.float32)
    img[..., 3] = 1.0
    return img, np_to_texture(device, img, usage="read", precision="fp32")


@pytest.fixture(scope="module")
def aug(ctx):
    device, modules = ctx
    return ColorAugOps(device, modules)


@pytest.fixture(scope="module")
def hist(ctx):
    device, modules = ctx
    return HistColorOps(device, modules)


# ---------------------------------------------------------------- solarize
def test_solarize_matches_numpy_and_identity(aug, img_and_tex):
    img, t = img_and_tex
    out = texture_to_np(aug.solarize(t, threshold=0.5))
    rgb = img[..., :3]
    ref = np.where(rgb >= 0.5, 1.0 - rgb, rgb)
    np.testing.assert_allclose(out[..., :3], ref, atol=ATOL)

    # threshold above every value -> identity
    out_id = texture_to_np(aug.solarize(t, threshold=2.0))
    np.testing.assert_allclose(out_id[..., :3], rgb, atol=ATOL)


# --------------------------------------------------------------- posterize
def test_posterize_matches_numpy(aug, img_and_tex):
    img, t = img_and_tex
    for bits in (1, 3, 4, 8):
        out = texture_to_np(aug.posterize(t, bits=bits))
        step = float(1 << (8 - bits))
        v = np.floor(np.clip(img[..., :3], 0, 1) * 255.0 + 0.5)
        ref = (v - np.mod(v, step)) / 255.0
        np.testing.assert_allclose(out[..., :3], ref, atol=ATOL)


def test_posterize_bits_validation(aug, img_and_tex):
    _, t = img_and_tex
    with pytest.raises(ValueError):
        aug.posterize(t, bits=0)


# --------------------------------------------------------------- grayscale
def test_grayscale_luma_and_channel_equality(aug, img_and_tex):
    img, t = img_and_tex
    out = texture_to_np(aug.grayscale(t))
    ref = img[..., :3] @ np.array([0.299, 0.587, 0.114])
    for ch in range(3):
        np.testing.assert_allclose(out[..., ch], ref, atol=ATOL)


# --------------------------------------------------------------- normalize
def test_normalize_matches_numpy_and_roundtrips(aug, img_and_tex):
    img, t = img_and_tex
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    out = texture_to_np(aug.normalize(t, mean, std))
    ref = (img[..., :3] - np.array(mean)) / np.array(std)
    # output goes negative — lossless readback must preserve it
    assert out[..., :3].min() < 0
    np.testing.assert_allclose(out[..., :3], ref, atol=5e-3)

    # invert the affine map: x * std + mean == identity
    back = texture_to_np(
        aug.normalize(
            np_to_texture(aug.device, out, usage="read", precision="fp32"),
            tuple(-m / s for m, s in zip(mean, std)),
            tuple(1.0 / s for s in std),
        )
    )
    np.testing.assert_allclose(back[..., :3], img[..., :3], atol=5e-3)


def test_normalize_validation(aug, img_and_tex):
    _, t = img_and_tex
    with pytest.raises(ValueError):
        aug.normalize(t, (0.5, 0.5, 0.5), (0.5, 0.0, 0.5))


# ---------------------------------------------------------------- equalize
def np_equalize(rgb):
    out = np.empty_like(rgb)
    for ch in range(3):
        v = np.clip(rgb[..., ch], 0, 1)
        bins = np.minimum((v * 255.0 + 0.5).astype(np.int64), 255)
        hist = np.bincount(bins.ravel(), minlength=256).astype(np.float64)
        nz = np.nonzero(hist)[0]
        cdf = np.cumsum(hist)
        cdf_min = hist[nz[0]] if len(nz) else 0.0
        denom = cdf[-1] - cdf_min
        if denom <= 0:
            lut = np.arange(256) / 255.0
        else:
            lut = np.clip((cdf - cdf_min) / denom, 0.0, 1.0)
        out[..., ch] = lut[bins]
    return out


def test_equalize_matches_numpy(hist, img_and_tex):
    img, t = img_and_tex
    out = texture_to_np(hist.equalize(t))
    np.testing.assert_allclose(out[..., :3], np_equalize(img[..., :3]), atol=ATOL)


def test_equalize_constant_image_passthrough(hist, ctx):
    device, _ = ctx
    img = np.full((16, 16, 4), 0.42, dtype=np.float32)
    img[..., 3] = 1.0
    t = np_to_texture(device, img, usage="read", precision="fp32")
    out = texture_to_np(hist.equalize(t))
    # constant channel: identity LUT -> value maps to its own bin
    np.testing.assert_allclose(out[..., :3], np.round(0.42 * 255) / 255, atol=ATOL)


# ------------------------------------------------------------ autocontrast
def test_autocontrast_matches_numpy(hist, ctx):
    device, _ = ctx
    rng = np.random.default_rng(32)
    # compressed range, off-center — a real stretch, incl. negatives
    img = (rng.random((40, 40, 4)).astype(np.float32) * 0.5 - 0.1)
    img[..., 3] = 1.0
    t = np_to_texture(device, img, usage="read", precision="fp32")
    out = texture_to_np(hist.autocontrast(t))
    rgb = img[..., :3].astype(np.float64)
    lo = rgb.min(axis=(0, 1))
    hi = rgb.max(axis=(0, 1))
    ref = (rgb - lo) / (hi - lo)
    np.testing.assert_allclose(out[..., :3], ref, atol=ATOL)
    # exact min/max reduction: full [0, 1] range is hit
    assert out[..., :3].min() == pytest.approx(0.0, abs=1e-6)
    assert out[..., :3].max() == pytest.approx(1.0, abs=1e-6)


def test_autocontrast_constant_channel_passthrough(hist, ctx):
    device, _ = ctx
    rng = np.random.default_rng(33)
    img = rng.random((16, 16, 4)).astype(np.float32)
    img[..., 0] = 0.7  # constant red channel
    img[..., 3] = 1.0
    t = np_to_texture(device, img, usage="read", precision="fp32")
    out = texture_to_np(hist.autocontrast(t))
    np.testing.assert_allclose(out[..., 0], 0.7, atol=ATOL)


# ------------------------------------------------------- functional wrappers
def test_functional_wrappers_roundtrip(ctx):
    import slangimage as si

    rng = np.random.default_rng(34)
    arr = rng.random((32, 32, 4)).astype(np.float32)
    arr[..., 3] = 1.0
    img = si.from_numpy(arr, precision="fp32", device=ctx[0])
    for out in (
        si.ops.solarize(img, 0.6),
        si.ops.posterize(img, 3),
        si.ops.grayscale(img),
        si.ops.normalize(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        si.ops.equalize(img),
        si.ops.autocontrast(img),
    ):
        a = out.to_numpy()
        assert a.shape[:2] == (32, 32) and np.isfinite(a).all()
