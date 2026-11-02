# tests/test_color_adjust.py
# Property tests for the color utility set.

from __future__ import annotations


import numpy as np
import pytest

from slangimage.runtime.textures import np_to_texture, texture_to_np
from slangimage.kernels.color_adjust import ColorAdjustOps

ATOL = 2e-3
LUM = np.array([0.299, 0.587, 0.114], dtype=np.float64)




@pytest.fixture(scope="module")
def img_and_tex(ctx):
    device, _ = ctx
    rng = np.random.default_rng(21)
    img = rng.random((64, 64, 4)).astype(np.float32) * 0.8 + 0.1  # avoid clip
    img[..., 3] = 1.0
    return img, np_to_texture(device, img, usage="read", precision="fp32")


@pytest.fixture(scope="module")
def ops(ctx):
    device, modules = ctx
    return ColorAdjustOps(device, modules)


def ctx_from_ops(ops):
    return ops.device, None


# ---------------------------------------------------------------- exposure
def test_exposure_identity_and_stops(ops, img_and_tex):
    img, t = img_and_tex
    out0 = texture_to_np(ops.exposure(t, ev=0.0))
    np.testing.assert_allclose(out0[..., :3], img[..., :3], atol=ATOL)

    out1 = texture_to_np(ops.exposure(t, ev=1.0))
    ref = np.clip(img[..., :3] * 2.0, 0, 1)
    np.testing.assert_allclose(out1[..., :3], ref, atol=ATOL)

    outm = texture_to_np(ops.exposure(t, ev=-1.0))
    np.testing.assert_allclose(outm[..., :3], img[..., :3] * 0.5, atol=ATOL)


# ------------------------------------------------------------------- sepia
def test_sepia_intensity_zero_identity_and_matrix(ops, img_and_tex):
    img, t = img_and_tex
    out0 = texture_to_np(ops.sepia(t, intensity=0.0))
    np.testing.assert_allclose(out0[..., :3], img[..., :3], atol=ATOL)

    out1 = texture_to_np(ops.sepia(t, intensity=1.0))
    m = np.array(
        [[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]]
    )
    ref = np.clip(img[..., :3] @ m.T, 0, 1)
    np.testing.assert_allclose(out1[..., :3], ref, atol=ATOL)
    # sepia of any image is warm: r >= g >= b
    assert (out1[..., 0] >= out1[..., 1] - 1e-4).all()
    assert (out1[..., 1] >= out1[..., 2] - 1e-4).all()


# --------------------------------------------------------------------- hue
def test_hue_identity_full_turn_and_luma_preserved(ops, img_and_tex):
    img, t = img_and_tex
    out0 = texture_to_np(ops.hue(t, angle=0.0))
    np.testing.assert_allclose(out0[..., :3], img[..., :3], atol=ATOL)

    out_turn = texture_to_np(ops.hue(t, angle=2 * np.pi))
    np.testing.assert_allclose(out_turn[..., :3], img[..., :3], atol=5e-3)

    # luminance preservation holds while colors stay in gamut, so test it
    # on a low-chroma image (clamping breaks it for saturated inputs, as in CI)
    device, _ = ctx_from_ops(ops)
    rng = np.random.default_rng(23)
    soft = np.full((32, 32, 4), 0.5, dtype=np.float32)
    soft[..., :3] += (rng.random((32, 32, 3)).astype(np.float32) - 0.5) * 0.3
    soft[..., 3] = 1.0
    ts = np_to_texture(device, soft, usage="read", precision="fp32")
    out_pi = texture_to_np(ops.hue(ts, angle=np.pi))
    lum_in = soft[..., :3].astype(np.float64) @ LUM
    lum_out = out_pi[..., :3].astype(np.float64) @ LUM
    np.testing.assert_allclose(lum_out, lum_in, atol=1e-2)
    # and the rotation actually changed chroma
    assert np.abs(out_pi[..., :3] - soft[..., :3]).max() > 0.1


# ------------------------------------------------------------- false color
def test_false_color_endpoints(ops, ctx):
    device, _ = ctx
    img = np.zeros((2, 2, 4), dtype=np.float32)
    img[0, :, :3] = 0.0   # luma 0
    img[1, :, :3] = 1.0   # luma 1
    img[..., 3] = 1.0
    t = np_to_texture(device, img, usage="read", precision="fp32")

    out = texture_to_np(ops.false_color(t, color0=(0.2, 0.0, 0.4), color1=(1.0, 0.9, 0.6)))
    np.testing.assert_allclose(out[0, 0, :3], [0.2, 0.0, 0.4], atol=ATOL)
    np.testing.assert_allclose(out[1, 0, :3], [1.0, 0.9, 0.6], atol=ATOL)


# --------------------------------------------------------- highlight/shadow
def test_highlight_shadow_defaults_identity(ops, img_and_tex):
    img, t = img_and_tex
    out = texture_to_np(ops.highlight_shadow(t, shadow_amount=0.0, highlight_amount=1.0))
    np.testing.assert_allclose(out[..., :3], img[..., :3], atol=ATOL)


def test_shadow_lift_brightens_darks_only(ops, ctx):
    device, _ = ctx
    img = np.zeros((2, 2, 4), dtype=np.float32)
    img[0, :, :3] = 0.15  # dark
    img[1, :, :3] = 0.9   # bright
    img[..., 3] = 1.0
    t = np_to_texture(device, img, usage="read", precision="fp32")

    out = texture_to_np(ops.highlight_shadow(t, shadow_amount=1.0))
    assert out[0, 0, 0] > 0.15 + 0.05, "dark pixel not lifted"
    np.testing.assert_allclose(out[1, 0, :3], 0.9, atol=2e-2)  # bright ~unchanged

    # negative amount deepens shadows
    out_neg = texture_to_np(ops.highlight_shadow(t, shadow_amount=-0.8))
    assert out_neg[0, 0, 0] < 0.15 - 0.02, "dark pixel not deepened"


def test_highlight_compression_darkens_brights_only(ops, ctx):
    device, _ = ctx
    img = np.zeros((2, 2, 4), dtype=np.float32)
    img[0, :, :3] = 0.15
    img[1, :, :3] = 0.9
    img[..., 3] = 1.0
    t = np_to_texture(device, img, usage="read", precision="fp32")

    out = texture_to_np(ops.highlight_shadow(t, highlight_amount=0.0))
    assert out[1, 0, 0] < 0.9 - 0.05, "bright pixel not compressed"
    np.testing.assert_allclose(out[0, 0, :3], 0.15, atol=2e-2)  # dark ~unchanged


# ------------------------------------------------------------- fp16 smoke
def test_color_adjust_fp16_smoke(ops, ctx):
    device, _ = ctx
    rng = np.random.default_rng(22)
    img = rng.random((32, 32, 4)).astype(np.float16)
    img[..., 3] = 1.0
    t = np_to_texture(device, img, usage="read", precision="fp16")

    outs = [
        ops.exposure(t, 0.5),
        ops.sepia(t, 0.36),
        ops.hue(t, 0.15),
        ops.false_color(t),
        ops.highlight_shadow(t, 0.5, 0.7),
    ]
    for out in outs:
        arr = texture_to_np(out)
        assert np.isfinite(arr).all()
        assert arr.min() >= -1.0 and arr.max() <= 2.0  # unclipped readback
